// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later

// A WebAssembly module for wasm32v1-none (core and alloc only) around the Rust that burn-onnx
// generated from a graph whose three inputs are int64 [batch, seq] tensors and whose output is
// float logits [batch, seq, vocab], with LoadStrategy::Bytes (src/model.rs).
// The host: weights_alloc(len) returns a buffer in linear memory, the host writes the weights
// file into it, model_load() reads it; ids_alloc(n) returns a buffer of n i32, the host writes the
// token ids; forward(k0, k1, k2) runs the model and returns the number of f32 written; out_ptr()
// is where. The generated `forward` takes its inputs by position; k0..k2 say which input each
// position is: 0 input_ids, 1 attention_mask (all ones), 2 position_ids (0..n), as
// batches/0010/runner does.
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use burn::prelude::*;
use burn::tensor::Bytes;
use core::cell::UnsafeCell;

// The generated file is included as it is. It calls f64::ceil (the ONNX Range operator), which is in std
// and not in core; num_traits::float::Float, through libm, provides it, and the trait is imported here, in
// the module that holds the generated items, because a method is looked up where it is called.
mod model {
    use num_traits::float::Float;
    include!("model.rs");
}

#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

// The panic message is kept in a buffer the host reads after the trap (panic_ptr, panic_len).
struct PanicBuf;
impl core::fmt::Write for PanicBuf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let (len, buf) = (PANIC_LEN.get(), PANIC.get());
        let n = s.len().min(buf.len() - *len);
        buf[*len..*len + n].copy_from_slice(&s.as_bytes()[..n]);
        *len += n;
        Ok(())
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    use core::fmt::Write;
    *PANIC_LEN.get() = 0;
    let _ = write!(PanicBuf, "{info}");
    #[allow(unused_unsafe)]
    unsafe {
        core::arch::wasm32::unreachable()
    }
}

// The module runs on one thread: no atomics, no locks.
struct Single<T>(UnsafeCell<T>);
unsafe impl<T> Sync for Single<T> {}
impl<T> Single<T> {
    const fn new(v: T) -> Self {
        Self(UnsafeCell::new(v))
    }
    #[allow(clippy::mut_from_ref)]
    fn get(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }
}

static PANIC: Single<[u8; 2048]> = Single::new([0; 2048]);
static PANIC_LEN: Single<usize> = Single::new(0);
static WEIGHTS: Single<Vec<u8>> = Single::new(Vec::new());
static IDS: Single<Vec<i32>> = Single::new(Vec::new());
static OUT: Single<Vec<f32>> = Single::new(Vec::new());
static MODEL: Single<Option<model::Model>> = Single::new(None);

#[cfg(not(target_has_atomic = "ptr"))]
mod one_thread {
    struct OneThread;
    critical_section::set_impl!(OneThread);
    unsafe impl critical_section::Impl for OneThread {
        unsafe fn acquire() -> critical_section::RawRestoreState {}
        unsafe fn release(_: critical_section::RawRestoreState) {}
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn weights_alloc(len: usize) -> *mut u8 {
    let w = WEIGHTS.get();
    *w = alloc::vec![0u8; len];
    w.as_mut_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn model_load() -> i32 {
    let bytes = Bytes::from_bytes_vec(core::mem::take(WEIGHTS.get()));
    let device = Device::flex();
    *MODEL.get() = Some(model::Model::from_bytes(bytes, &device));
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn ids_alloc(n: usize) -> *mut i32 {
    let ids = IDS.get();
    *ids = alloc::vec![0i32; n];
    ids.as_mut_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn forward(k0: i32, k1: i32, k2: i32) -> usize {
    let device = Device::flex();
    let n = IDS.get().len();
    let input = |k: i32| {
        let v: Vec<i32> = match k {
            0 => IDS.get().clone(),
            1 => alloc::vec![1; n],
            _ => (0..n as i32).collect(),
        };
        Tensor::<2, Int>::from_ints(TensorData::new(v, [1, n]), &device)
    };
    let out = MODEL
        .get()
        .as_ref()
        .unwrap()
        .forward(input(k0), input(k1), input(k2));
    let data = out.into_data();
    *OUT.get() = data.as_slice::<f32>().unwrap().to_vec();
    OUT.get().len()
}

#[unsafe(no_mangle)]
pub extern "C" fn out_ptr() -> *const f32 {
    OUT.get().as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn panic_ptr() -> *const u8 {
    PANIC.get().as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn panic_len() -> usize {
    *PANIC_LEN.get()
}
