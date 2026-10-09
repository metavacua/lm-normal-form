// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later

// An example for burn-onnx's own crate, which the workflow copies to
// crates/burn-onnx/examples/gen_bytes.rs of its clone: the chain of `onnx2burn`, with
// LoadStrategy::Bytes, which that binary does not offer (it offers File and Embedded).
// Usage: gen_bytes INPUT.onnx OUTDIR
use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    ModelGen::new()
        .input(&a[1])
        .out_dir(&a[2])
        .development(false)
        .load_strategy(LoadStrategy::Bytes)
        .run_from_cli();
}
