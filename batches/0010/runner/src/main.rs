// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later

// Runs the Rust that burn-onnx generated (src/model.rs) from a graph whose three inputs are
// int64 [batch, seq] tensors and whose output is float logits [batch, seq, vocab]. For each
// IDS argument (token ids, comma separated) it writes the logits of every position as
// little-endian float32 to OUTDIR/N.f32 (N counts from 0), as batches/0010/logits.py does.
// INPUT_ORDER names the graph's inputs in order, because the generated `forward` takes them
// by position.
// Usage: INPUT_ORDER=input_ids,attention_mask,position_ids runner WEIGHTS OUTDIR IDS [IDS ...]
use burn::prelude::*;
use std::time::Instant;

mod model;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let order = std::env::var("INPUT_ORDER").expect("INPUT_ORDER names the graph's inputs in order");
    let order: Vec<&str> = order.split(',').collect();
    assert_eq!(order.len(), 3, "the generated forward is called with three inputs");
    let device: Device = Default::default();
    let start = Instant::now();
    let m = model::Model::from_file(&a[1], &device);
    eprintln!("model loaded in {:.2?}", start.elapsed());
    for (k, ids) in a[3..].iter().enumerate() {
        let ids: Vec<i32> = ids.split(',').map(|x| x.parse().unwrap()).collect();
        let n = ids.len();
        let input = |name: &str| {
            let v: Vec<i32> = match name {
                "input_ids" => ids.clone(),
                "attention_mask" => vec![1; n],
                "position_ids" => (0..n as i32).collect(),
                _ => panic!("no value for the input {name}"),
            };
            Tensor::<2, Int>::from_ints(TensorData::new(v, [1, n]), &device)
        };
        let start = Instant::now();
        let out = m.forward(input(order[0]), input(order[1]), input(order[2]));
        let data = out.into_data();
        eprintln!("{k}: {:?} in {:.2?}", data.shape(), start.elapsed());
        let bytes: Vec<u8> = data
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        std::fs::write(format!("{}/{k}.f32", a[2]), bytes).unwrap();
    }
}
