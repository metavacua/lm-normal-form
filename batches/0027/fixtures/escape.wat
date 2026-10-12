;; SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
;; SPDX-License-Identifier: AGPL-3.0-or-later
;; The negative control of closure.dl: a module with an import and a start function. closure.dl must derive two escape facts and closed = 0.
(module
  (import "env" "host_log" (func $log (param i32)))
  (memory (export "memory") 1 2)
  (func $s (call $log (i32.const 1)))
  (start $s)
  (func (export "run") (param i32) (result i32) (local.get 0)))
