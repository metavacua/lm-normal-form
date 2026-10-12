;; SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
;; SPDX-License-Identifier: AGPL-3.0-or-later
;; A hand-written module with the probe_* exports of batches/0027/wasm/src/lib.rs (not the ternary kernels or the canonical form):
;; the positive control of wasm_host.py and of closure.dl, which can be run without compiling any Rust. Three pages, at most 64, a
;; stack pointer at 64 KiB with the scratch buffer above it, and a table of three functions at the indices 1, 2 and 3.
(module
  (type $t1 (func (param i32) (result i32)))
  (type $t2 (func (param i32 i32) (result i32)))
  (memory (export "memory") 3 64)
  (global $sp (mut i32) (i32.const 65536))
  (table 4 funcref)
  (elem (i32.const 1) $f0 $f1 $f2)
  (func $f0 (type $t1) (i32.add (local.get 0) (i32.const 1)))
  (func $f1 (type $t1) (i32.mul (local.get 0) (i32.const 2)))
  (func $f2 (type $t2) (i32.sub (local.get 0) (local.get 1)))
  (func (export "buf_ptr") (result i32) (i32.const 65536))
  (func (export "buf_len") (result i32) (i32.const 65536))
  (func (export "probe_load8") (param i32) (result i32) (i32.load8_u (local.get 0)))
  (func (export "probe_store8") (param i32 i32) (i32.store8 (local.get 0) (local.get 1)))
  (func (export "probe_mem_size") (result i32) (memory.size))
  (func (export "probe_mem_grow") (param i32) (result i32) (memory.grow (local.get 0)))
  (func (export "probe_trap") (unreachable))
  (func (export "probe_loop") (param i32) (result i32)
    (local i32 i32)
    (block $done
      (loop $l
        (br_if $done (i32.ge_u (local.get 2) (local.get 0)))
        (local.set 1 (i32.add (local.get 1) (local.get 2)))
        (local.set 2 (i32.add (local.get 2) (i32.const 1)))
        (br $l)))
    (local.get 1))
  (func (export "probe_addr0") (result i32) (i32.const 1))
  (func (export "probe_addr1") (result i32) (i32.const 2))
  (func (export "probe_addr2") (result i32) (i32.const 3))
  (func (export "probe_call") (param i32 i32) (result i32)
    (call_indirect (type $t1) (local.get 1) (local.get 0)))
  (func $rec (export "probe_recurse") (param i32) (result i32)
    (if (result i32) (i32.eqz (local.get 0))
      (then (i32.const 0))
      (else (i32.add (call $rec (i32.sub (local.get 0) (i32.const 1))) (i32.and (local.get 0) (i32.const 255)))))))
