// SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batch 0027: a module with no imports for `wasm32v1-none` (WebAssembly 1.0 plus mutable globals, `core` only).
//!
//! Three groups of exports, all over one 64 KiB scratch buffer in the module's own linear memory:
//!  * `probe_*`: one instruction or one resource each (a load or store at an address, `memory.grow`, `unreachable`, a
//!    recursion, a loop, an indirect call), so that the host can check that each bound which `closure.dl` derives from the
//!    module's structure is the bound that wasmtime enforces, and what the failure is (a trap or `-1`).
//!  * `tern_i8`, `tern_f32`: a row of a ternary matrix (two bits per trit, the code of ggml's TQ2_0 in the byte layout of this
//!    module) times a vector, with int8 activations accumulated in i32 (exact: the sum of at most 65536 terms of absolute
//!    value 128 is below 2^31) and with f32 activations accumulated in f32 (not exact: the sum depends on the order of the
//!    terms).
//!  * `nf_units`: the canonical form of a block of gated units over {-1, 0, 1} under the sign gauge (u, w) -> (-u, -w) and the
//!    permutations of the units: the leading non-zero trit of u (of w when u = 0) is made +1, then the units are sorted.
#![no_std]

use core::arch::wasm32;
use core::hint::black_box;
use core::ptr::addr_of_mut;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    wasm32::unreachable()
}

const BUF_LEN: usize = 1 << 16;
static mut BUF: [u8; BUF_LEN] = [0; BUF_LEN];

/// The scratch buffer. The module has one thread and no reference to the buffer outlives the call that took it.
fn buf() -> &'static mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(addr_of_mut!(BUF).cast::<u8>(), BUF_LEN) }
}

/// Taking the address of a `static mut` is safe since Rust 1.82; the block keeps older compilers working.
#[allow(unused_unsafe)]
#[no_mangle]
pub extern "C" fn buf_ptr() -> usize {
    unsafe { addr_of_mut!(BUF) as usize }
}

#[no_mangle]
pub extern "C" fn buf_len() -> usize {
    BUF_LEN
}

// ---------------------------------------------------------------------------------------------- probes

#[no_mangle]
pub extern "C" fn probe_load8(addr: usize) -> u32 {
    unsafe { core::ptr::read_volatile(addr as *const u8) as u32 }
}

#[no_mangle]
pub extern "C" fn probe_store8(addr: usize, v: u32) {
    unsafe { core::ptr::write_volatile(addr as *mut u8, v as u8) }
}

/// The size of the linear memory in pages (`memory.size`).
#[no_mangle]
pub extern "C" fn probe_mem_size() -> usize {
    wasm32::memory_size::<0>()
}

/// `memory.grow`: the previous size in pages, or `usize::MAX` (that is, -1) when the memory cannot grow.
#[no_mangle]
pub extern "C" fn probe_mem_grow(pages: usize) -> usize {
    wasm32::memory_grow::<0>(pages)
}

#[no_mangle]
pub extern "C" fn probe_trap() {
    wasm32::unreachable()
}

#[inline(never)]
fn recurse(n: u32) -> u32 {
    let mut local = [0u8; 64];
    local[(n % 64) as usize] = n as u8;
    let local = black_box(local);
    if n == 0 {
        return local[0] as u32;
    }
    let me: fn(u32) -> u32 = recurse;
    let next = black_box(me);
    next(n - 1).wrapping_add(local[(n % 64) as usize] as u32)
}

/// A recursion that cannot be turned into a loop and that uses the shadow stack: the sum of k mod 256 for k = 1..n.
#[no_mangle]
pub extern "C" fn probe_recurse(n: u32) -> u32 {
    recurse(n)
}

/// A loop of n iterations: the sum of i for i < n, modulo 2^32. The host bounds it with fuel.
#[no_mangle]
pub extern "C" fn probe_loop(n: u32) -> u32 {
    let mut acc = 0u32;
    let mut i = 0u32;
    while i < n {
        acc = black_box(acc.wrapping_add(i));
        i += 1;
    }
    acc
}

fn f0(x: i32) -> i32 {
    x.wrapping_add(1)
}

fn f1(x: i32) -> i32 {
    x.wrapping_mul(2)
}

fn f2(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

/// The table index of f0, f1 and f2 (a function pointer is a table index on this target).
#[no_mangle]
pub extern "C" fn probe_addr0() -> usize {
    black_box(f0 as fn(i32) -> i32) as usize
}

#[no_mangle]
pub extern "C" fn probe_addr1() -> usize {
    black_box(f1 as fn(i32) -> i32) as usize
}

#[no_mangle]
pub extern "C" fn probe_addr2() -> usize {
    black_box(f2 as fn(i32, i32) -> i32) as usize
}

/// `call_indirect` with the type (i32) -> i32 on the table index `idx`: an index outside the table, or the index of a
/// function of another type, is a trap.
#[no_mangle]
pub extern "C" fn probe_call(idx: usize, x: i32) -> i32 {
    let f: fn(i32) -> i32 = unsafe { core::mem::transmute::<usize, fn(i32) -> i32>(idx) };
    f(x)
}

// ---------------------------------------------------------------------------------------------- the ternary kernels

/// The two-bit code of ggml's TQ2_0, q = trit + 1, with four consecutive trits to a byte (the byte layout of this module, not ggml's block layout);
/// the code 3, which ggml would read as +2, is read as 0 here.
fn trit(code: u8) -> i32 {
    match code & 3 {
        0 => -1,
        2 => 1,
        _ => 0,
    }
}

/// y[r] = sum_c trit(r, c) * x[c] for r < rows. The trits are at `w_off`, four to a byte, the first in the low bits, row by
/// row; x is `cols` signed bytes at `x_off`; y is `rows` little-endian i32 at `y_off`. Every sum is exact, so it does not
/// depend on the order of the columns.
#[no_mangle]
pub extern "C" fn tern_i8(rows: usize, cols: usize, w_off: usize, x_off: usize, y_off: usize) {
    let b = buf();
    for r in 0..rows {
        let mut acc: i32 = 0;
        for c in 0..cols {
            let k = r * cols + c;
            let w = trit(b[w_off + k / 4] >> (2 * (k % 4)));
            let x = b[x_off + c] as i8 as i32;
            acc = acc.wrapping_add(w * x);
        }
        let o = y_off + 4 * r;
        b[o..o + 4].copy_from_slice(&acc.to_le_bytes());
    }
}

/// The same with `cols` little-endian f32 at `x_off` and f32 sums at `y_off`, accumulated in the order of the columns.
#[no_mangle]
pub extern "C" fn tern_f32(rows: usize, cols: usize, w_off: usize, x_off: usize, y_off: usize) {
    let b = buf();
    for r in 0..rows {
        let mut acc: f32 = 0.0;
        for c in 0..cols {
            let k = r * cols + c;
            let w = trit(b[w_off + k / 4] >> (2 * (k % 4))) as f32;
            let xo = x_off + 4 * c;
            let x = f32::from_le_bytes([b[xo], b[xo + 1], b[xo + 2], b[xo + 3]]);
            acc += w * x;
        }
        let o = y_off + 4 * r;
        b[o..o + 4].copy_from_slice(&acc.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------------------------- the canonical form

fn unit_less(b: &[u8], a: usize, c: usize, l: usize) -> bool {
    for k in 0..l {
        let x = b[a + k] as i8;
        let y = b[c + k] as i8;
        if x != y {
            return x < y;
        }
    }
    false
}

/// `n` units of `2d + e` trits (one signed byte each: g, then u, then w) at `off`, in place. The sign of a unit is the
/// leading non-zero trit of u, or of w when u is zero, or +1; when it is -1, u and w are negated. Then the units are sorted
/// in the lexicographic order of their trits (insertion sort: n is small).
#[no_mangle]
pub extern "C" fn nf_units(n: usize, d: usize, e: usize, off: usize) {
    let b = buf();
    let l = 2 * d + e;
    for i in 0..n {
        let base = off + i * l;
        let mut c: i8 = 0;
        for k in d..2 * d {
            let t = b[base + k] as i8;
            if t != 0 {
                c = t;
                break;
            }
        }
        if c == 0 {
            for k in 2 * d..l {
                let t = b[base + k] as i8;
                if t != 0 {
                    c = t;
                    break;
                }
            }
        }
        if c < 0 {
            for k in d..l {
                b[base + k] = (-(b[base + k] as i8)) as u8;
            }
        }
    }
    for i in 1..n {
        let mut j = i;
        while j > 0 && unit_less(b, off + j * l, off + (j - 1) * l, l) {
            for k in 0..l {
                b.swap(off + j * l + k, off + (j - 1) * l + k);
            }
            j -= 1;
        }
    }
}
