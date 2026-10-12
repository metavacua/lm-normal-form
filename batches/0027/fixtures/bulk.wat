;; SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
;; SPDX-License-Identifier: AGPL-3.0-or-later
;; A control of the engine's configuration: a module with a bulk-memory instruction, a proposal after WebAssembly 1.0. The validator limited to
;; WebAssembly 1.0 must refuse it, the default engine must compile it, and the engine configured for WebAssembly 1.0 must refuse it.
(module
  (memory 1)
  (func (export "copy") (param i32 i32 i32)
    (memory.copy (local.get 0) (local.get 1) (local.get 2))))
