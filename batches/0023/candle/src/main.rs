// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batch 0023, the candle runtime for a checkpoint whose whole input space is run: it loads a Llama checkpoint from a Hugging Face directory (config.json, model.safetensors) with
//! `candle-transformers` and writes, for every sequence of INPUTS.json ({"sequences": [[token ids], ...]}), the logits of every position as little-endian float32 (OUT.logits.f32, positions x
//! vocabulary), by feeding one token at a time with a key/value cache, because `Llama::forward` returns the logits of the last position only. Same loading code as batch 0014's lmnf-candle.
//! Usage: lmnf-candle-logits MODEL_DIR INPUTS.json OUT DTYPE      (DTYPE: f32)
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Deserialize)]
struct Inputs {
    sequences: Vec<Vec<u32>>,
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        bail!("usage: lmnf-candle-logits MODEL_DIR INPUTS.json OUT DTYPE(f32)");
    }
    let (dir, inputs_path, out) = (&a[1], &a[2], &a[3]);
    let dtype = match a[4].as_str() {
        "f32" => DType::F32,
        other => bail!("dtype {other} is not f32"),
    };
    let device = Device::Cpu;
    let inputs: Inputs = serde_json::from_slice(&fs::read(inputs_path)?)?;
    let cfg: LlamaConfig = serde_json::from_slice(&fs::read(Path::new(dir).join("config.json"))?)?;
    let config = cfg.into_config(false);
    let weights = Path::new(dir).join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
    let model = Llama::load(vb, &config)?;
    let mut bin = std::io::BufWriter::new(fs::File::create(format!("{out}.logits.f32"))?);
    let mut positions = 0usize;
    for seq in &inputs.sequences {
        let mut cache = Cache::new(true, dtype, &config, &device)?;
        for (pos, &tok) in seq.iter().enumerate() {
            let input = Tensor::new(&[tok][..], &device)?.unsqueeze(0)?;
            let l = model.forward(&input, pos, &mut cache)?.squeeze(0)?.to_vec1::<f32>()?;
            for x in &l {
                bin.write_all(&x.to_le_bytes())?;
            }
            positions += 1;
        }
    }
    bin.flush()?;
    println!("{positions} positions written to {out}.logits.f32");
    Ok(())
}
