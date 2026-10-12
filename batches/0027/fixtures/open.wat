;; SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
;; SPDX-License-Identifier: AGPL-3.0-or-later
;; The second control of closure.dl: a closed module whose memory declares no maximum and which can grow it. closure.dl must derive that its
;; memory is bounded by the host's limit alone (mem_declared_max_kib = -1, mem_bound_kib = the host's limit), and that "grow" may grow it.
(module
  (memory (export "memory") 1)
  (func (export "grow") (param i32) (result i32) (memory.grow (local.get 0)))
  (func (export "size") (result i32) (memory.size)))
