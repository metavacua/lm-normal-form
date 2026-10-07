// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The candle runtime of batch 0014. It loads a Llama checkpoint from a Hugging Face directory (config.json, model.safetensors) with
//! `candle-transformers` and, given token ids in a JSON file (the same file that the other runtimes read), writes:
//!   OUT.logits.f32   the logits of every position of every test sequence (little-endian float32, positions x vocabulary), by stepping one token at a
//!                    time with a key/value cache, because `Llama::forward` returns only the last position's logits
//!   OUT.json         load time, greedy generations (32 new tokens, stopping after a token of `eos`), the negative log-likelihood of the perplexity windows,
//!                    prefill speed (512 tokens, five repeats), decode speed (64 tokens after a prompt of 16, three repeats), and the peak resident memory
//! Usage: lmnf-candle MODEL_DIR INPUTS.json OUT DTYPE FULL      DTYPE is f32 or bf16; FULL is 1 for everything, 0 for load time and speed alone.
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Ids {
    test: Vec<Vec<u32>>,
    plain: Vec<Vec<u32>>,
    #[serde(default)]
    chat: Vec<Vec<u32>>,
    #[serde(default)]
    eos: Vec<u32>,
}

#[derive(Deserialize)]
struct Inputs {
    ids: Ids,
    wins: Vec<Vec<u32>>,
}

struct Runner<'a> {
    model: &'a Llama,
    config: &'a Config,
    dtype: DType,
    device: &'a Device,
}

impl<'a> Runner<'a> {
    fn cache(&self) -> Result<Cache> {
        Ok(Cache::new(true, self.dtype, self.config, self.device)?)
    }

    /// The logits (float32) after feeding `tokens` at positions pos, pos + 1, ...: those of the last token.
    fn logits(&self, tokens: &[u32], pos: usize, cache: &mut Cache) -> Result<Vec<f32>> {
        let input = Tensor::new(tokens, self.device)?.unsqueeze(0)?;
        let l = self.model.forward(&input, pos, cache)?;
        Ok(l.squeeze(0)?.to_vec1::<f32>()?)
    }

    fn generate(&self, prompts: &[Vec<u32>], eos: &[u32]) -> Result<Vec<Vec<u32>>> {
        let mut all = Vec::new();
        for p in prompts {
            let mut cache = self.cache()?;
            let mut l = self.logits(p, 0, &mut cache)?;
            let mut pos = p.len();
            let mut out: Vec<u32> = Vec::new();
            for _ in 0..32 {
                let nxt = argmax(&l);
                out.push(nxt);
                if eos.contains(&nxt) {
                    break;
                }
                l = self.logits(&[nxt], pos, &mut cache)?;
                pos += 1;
            }
            all.push(out);
        }
        Ok(all)
    }
}

/// The index of the first largest value, as torch.argmax gives it.
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

/// -log softmax(logits)[target], in float64.
fn nll(logits: &[f32], target: u32) -> f64 {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = logits.iter().map(|&x| ((x as f64) - m).exp()).sum();
    -((logits[target as usize] as f64) - m - s.ln())
}

/// VmHWM of /proc/self/status, in MiB.
fn peak_rss_mb() -> f64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse::<f64>().ok()))
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        bail!("usage: lmnf-candle MODEL_DIR INPUTS.json OUT DTYPE(f32|bf16) FULL(0|1)");
    }
    let (dir, inputs_path, out, full) = (&a[1], &a[2], &a[3], a[5] == "1");
    let dtype = match a[4].as_str() {
        "f32" => DType::F32,
        "bf16" => DType::BF16,
        other => bail!("dtype {other} is not f32 or bf16"),
    };
    let device = Device::Cpu;
    let inputs: Inputs = serde_json::from_slice(&fs::read(inputs_path)?)?;

    let t = Instant::now();
    let cfg: LlamaConfig = serde_json::from_slice(&fs::read(Path::new(dir).join("config.json"))?)?;
    let config = cfg.into_config(false);
    let weights = Path::new(dir).join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
    let model = Llama::load(vb, &config)?;
    let load_s = t.elapsed().as_secs_f64();
    let r = Runner { model: &model, config: &config, dtype, device: &device };

    let mut res = serde_json::Map::new();
    res.insert("load_s".into(), load_s.into());

    if full {
        let mut bin = std::io::BufWriter::new(fs::File::create(format!("{out}.logits.f32"))?);
        let (mut positions, mut vocab) = (0usize, 0usize);
        for seq in &inputs.ids.test {
            let mut cache = r.cache()?;
            for (pos, &tok) in seq.iter().enumerate() {
                let l = r.logits(&[tok], pos, &mut cache)?;
                vocab = l.len();
                for x in &l {
                    bin.write_all(&x.to_le_bytes())?;
                }
                positions += 1;
            }
        }
        bin.flush()?;
        res.insert("positions".into(), positions.into());
        res.insert("vocab".into(), vocab.into());

        let mut gen = serde_json::Map::new();
        gen.insert("plain".into(), serde_json::to_value(r.generate(&inputs.ids.plain, &inputs.ids.eos)?)?);
        if !inputs.ids.chat.is_empty() {
            gen.insert("chat".into(), serde_json::to_value(r.generate(&inputs.ids.chat, &inputs.ids.eos)?)?);
        }
        res.insert("gen".into(), gen.into());

        let (mut nll_sum, mut ntok) = (0.0f64, 0usize);
        for w in &inputs.wins {
            let mut cache = r.cache()?;
            for pos in 0..w.len() - 1 {
                let l = r.logits(&[w[pos]], pos, &mut cache)?;
                nll_sum += nll(&l, w[pos + 1]);
                ntok += 1;
            }
        }
        res.insert("ppl_nll".into(), nll_sum.into());
        res.insert("ppl_tokens".into(), ntok.into());
    }

    let w0 = &inputs.wins[0];
    {
        let mut c = r.cache()?;
        r.logits(w0, 0, &mut c)?;
    }
    let mut pre = Vec::new();
    for _ in 0..5 {
        let mut c = r.cache()?;
        let t = Instant::now();
        r.logits(w0, 0, &mut c)?;
        pre.push(w0.len() as f64 / t.elapsed().as_secs_f64());
    }
    let p16: Vec<u32> = inputs.ids.plain[0].iter().cloned().chain(w0.iter().cloned()).take(16).collect();
    let mut dec = Vec::new();
    for _ in 0..3 {
        let mut c = r.cache()?;
        let mut l = r.logits(&p16, 0, &mut c)?;
        let mut pos = p16.len();
        let t = Instant::now();
        for _ in 0..64 {
            let nxt = argmax(&l);
            l = r.logits(&[nxt], pos, &mut c)?;
            pos += 1;
        }
        dec.push(64.0 / t.elapsed().as_secs_f64());
    }
    res.insert("prefill_tps".into(), pre.into());
    res.insert("decode_tps".into(), dec.into());
    res.insert("peak_rss_mb".into(), peak_rss_mb().into());
    fs::write(format!("{out}.json"), serde_json::to_string_pretty(&res)?)?;
    Ok(())
}
