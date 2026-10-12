;; SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
;; SPDX-License-Identifier: AGPL-3.0-or-later
;; A control of the engine's configuration: a module with a SIMD value and instruction, a proposal after WebAssembly 1.0. The validator limited to
;; WebAssembly 1.0 must refuse it, the default engine must compile it, and the engine configured for WebAssembly 1.0 must refuse it.
(module
  (func (export "add") (param v128 v128) (result v128)
    (i32x4.add (local.get 0) (local.get 1))))
