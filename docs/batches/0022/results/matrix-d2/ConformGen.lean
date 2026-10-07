-- SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Written by batches/0022/gen22.py; do not edit.
import LmnfProofs.Runtime
open Lmnf Lmnf.Runtime Matrix

def E : Env Fp 2 (Fin 4) (Fin 2) (Fin 1) := mkEnv 2 4 2 1 2 (by intro a; have := a.isLt; omega) [(2 : Fp)]
def toks : Fin 2 → Fin 3 := ![2, 0]
def M0_emb : List (List Fp) := [[(885440 : Fp), (403958 : Fp)], [(794772 : Fp), (933488 : Fp)], [(441001 : Fp), (42450 : Fp)]]
def M0_head : List (List Fp) := [[(509532 : Fp), (424604 : Fp)], [(962838 : Fp), (821872 : Fp)], [(870163 : Fp), (318046 : Fp)]]
def M0_gf : List Fp := [(271494 : Fp), (536111 : Fp)]
def M0_0_wq : List (List Fp) := [[(611720 : Fp), (934973 : Fp)], [(952225 : Fp), (229053 : Fp)], [(529202 : Fp), (146039 : Fp)], [(295528 : Fp), (146534 : Fp)], [(792518 : Fp), (99437 : Fp)], [(648406 : Fp), (838234 : Fp)], [(262674 : Fp), (953938 : Fp)], [(558433 : Fp), (739426 : Fp)]]
def M0_0_wk : List (List Fp) := [[(849574 : Fp), (631140 : Fp)], [(945989 : Fp), (154100 : Fp)], [(325213 : Fp), (103560 : Fp)], [(765284 : Fp), (77324 : Fp)]]
def M0_0_wv : List (List Fp) := [[(942500 : Fp), (891786 : Fp)], [(717209 : Fp), (346236 : Fp)], [(495077 : Fp), (587007 : Fp)], [(105592 : Fp), (370977 : Fp)]]
def M0_0_wo : List (List Fp) := [[(455262 : Fp), (331556 : Fp), (640561 : Fp), (671532 : Fp), (957361 : Fp), (214410 : Fp), (579363 : Fp), (500181 : Fp)], [(464197 : Fp), (907343 : Fp), (546678 : Fp), (273145 : Fp), (65304 : Fp), (844132 : Fp), (963080 : Fp), (575352 : Fp)]]
def M0_0_wg : List (List Fp) := [[(97802 : Fp), (754665 : Fp)], [(880899 : Fp), (418196 : Fp)]]
def M0_0_wu : List (List Fp) := [[(744754 : Fp), (864912 : Fp)], [(823182 : Fp), (700609 : Fp)]]
def M0_0_wd : List (List Fp) := [[(655638 : Fp), (1198 : Fp)], [(641620 : Fp), (517553 : Fp)]]
def M0_0_an : List (List Fp) := [[(499749 : Fp), (375442 : Fp)]]
def M0_0_mn : List (List Fp) := [[(960490 : Fp), (14724 : Fp)]]
def M0_0 : Flat := { wq := tab M0_0_wq, wk := tab M0_0_wk, wv := tab M0_0_wv, wo := tab M0_0_wo, wg := tab M0_0_wg, wu := tab M0_0_wu, wd := tab M0_0_wd, an := tab1 (M0_0_an.getD 0 []), mn := tab1 (M0_0_mn.getD 0 []) }
def M0_0L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 M0_0
def M0_1_wq : List (List Fp) := [[(349317 : Fp), (255759 : Fp)], [(765752 : Fp), (341001 : Fp)], [(737822 : Fp), (912755 : Fp)], [(66043 : Fp), (200348 : Fp)], [(961564 : Fp), (595078 : Fp)], [(232473 : Fp), (250206 : Fp)], [(842368 : Fp), (149416 : Fp)], [(842194 : Fp), (569366 : Fp)]]
def M0_1_wk : List (List Fp) := [[(469730 : Fp), (95646 : Fp)], [(84353 : Fp), (335601 : Fp)], [(917595 : Fp), (532614 : Fp)], [(978147 : Fp), (513054 : Fp)]]
def M0_1_wv : List (List Fp) := [[(114355 : Fp), (316089 : Fp)], [(578045 : Fp), (305230 : Fp)], [(740883 : Fp), (130873 : Fp)], [(574033 : Fp), (348914 : Fp)]]
def M0_1_wo : List (List Fp) := [[(854030 : Fp), (967048 : Fp), (566528 : Fp), (213072 : Fp), (838260 : Fp), (632485 : Fp), (573812 : Fp), (616161 : Fp)], [(301630 : Fp), (466604 : Fp), (96083 : Fp), (625252 : Fp), (836695 : Fp), (403598 : Fp), (332447 : Fp), (603613 : Fp)]]
def M0_1_wg : List (List Fp) := [[(192800 : Fp), (198591 : Fp)], [(861370 : Fp), (195800 : Fp)]]
def M0_1_wu : List (List Fp) := [[(34574 : Fp), (642539 : Fp)], [(688557 : Fp), (272688 : Fp)]]
def M0_1_wd : List (List Fp) := [[(499678 : Fp), (72441 : Fp)], [(94187 : Fp), (711693 : Fp)]]
def M0_1_an : List (List Fp) := [[(868288 : Fp), (909748 : Fp)]]
def M0_1_mn : List (List Fp) := [[(253868 : Fp), (304433 : Fp)]]
def M0_1 : Flat := { wq := tab M0_1_wq, wk := tab M0_1_wk, wv := tab M0_1_wv, wo := tab M0_1_wo, wg := tab M0_1_wg, wu := tab M0_1_wu, wd := tab M0_1_wd, an := tab1 (M0_1_an.getD 0 []), mn := tab1 (M0_1_mn.getD 0 []) }
def M0_1L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 M0_1
def M0 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L, M0_1L]
def Mf_emb : List (List Fp) := [[(885440 : Fp), (403958 : Fp)], [(794772 : Fp), (933488 : Fp)], [(441001 : Fp), (42450 : Fp)]]
def Mf_head : List (List Fp) := [[(465806 : Fp), (192142 : Fp)], [(955763 : Fp), (297953 : Fp)], [(324793 : Fp), (447585 : Fp)]]
def Mf_gf : List Fp := [(1 : Fp), (1 : Fp)]
def Mf_0_wq : List (List Fp) := [[(541165 : Fp), (79985 : Fp)], [(63909 : Fp), (858441 : Fp)], [(376897 : Fp), (9751 : Fp)], [(379405 : Fp), (852986 : Fp)], [(889808 : Fp), (714158 : Fp)], [(277977 : Fp), (305307 : Fp)], [(675016 : Fp), (316155 : Fp)], [(496092 : Fp), (743462 : Fp)]]
def Mf_0_wk : List (List Fp) := [[(483210 : Fp), (753015 : Fp)], [(638496 : Fp), (438635 : Fp)], [(383965 : Fp), (656880 : Fp)], [(766372 : Fp), (590118 : Fp)]]
def Mf_0_wv : List (List Fp) := [[(19464 : Fp), (914976 : Fp)], [(405272 : Fp), (146339 : Fp)], [(493434 : Fp), (420936 : Fp)], [(338101 : Fp), (928997 : Fp)]]
def Mf_0_wo : List (List Fp) := [[(455262 : Fp), (331556 : Fp), (640561 : Fp), (671532 : Fp), (957361 : Fp), (214410 : Fp), (579363 : Fp), (500181 : Fp)], [(464197 : Fp), (907343 : Fp), (546678 : Fp), (273145 : Fp), (65304 : Fp), (844132 : Fp), (963080 : Fp), (575352 : Fp)]]
def Mf_0_wg : List (List Fp) := [[(561169 : Fp), (654127 : Fp)], [(142234 : Fp), (499433 : Fp)]]
def Mf_0_wu : List (List Fp) := [[(623482 : Fp), (926086 : Fp)], [(707215 : Fp), (735971 : Fp)]]
def Mf_0_wd : List (List Fp) := [[(655638 : Fp), (1198 : Fp)], [(641620 : Fp), (517553 : Fp)]]
def Mf_0_an : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_0_mn : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_0 : Flat := { wq := tab Mf_0_wq, wk := tab Mf_0_wk, wv := tab Mf_0_wv, wo := tab Mf_0_wo, wg := tab Mf_0_wg, wu := tab Mf_0_wu, wd := tab Mf_0_wd, an := tab1 (Mf_0_an.getD 0 []), mn := tab1 (Mf_0_mn.getD 0 []) }
def Mf_0L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 Mf_0
def Mf_1_wq : List (List Fp) := [[(849378 : Fp), (540707 : Fp)], [(277903 : Fp), (47076 : Fp)], [(66816 : Fp), (544618 : Fp)], [(172352 : Fp), (645509 : Fp)], [(977699 : Fp), (396237 : Fp)], [(910668 : Fp), (725219 : Fp)], [(831739 : Fp), (499378 : Fp)], [(750080 : Fp), (25834 : Fp)]]
def Mf_1_wk : List (List Fp) := [[(698663 : Fp), (496169 : Fp)], [(477938 : Fp), (422615 : Fp)], [(337158 : Fp), (67643 : Fp)], [(754406 : Fp), (450148 : Fp)]]
def Mf_1_wv : List (List Fp) := [[(776364 : Fp), (472892 : Fp)], [(31236 : Fp), (548997 : Fp)], [(888413 : Fp), (92821 : Fp)], [(470232 : Fp), (861406 : Fp)]]
def Mf_1_wo : List (List Fp) := [[(854030 : Fp), (967048 : Fp), (566528 : Fp), (213072 : Fp), (838260 : Fp), (632485 : Fp), (573812 : Fp), (616161 : Fp)], [(301630 : Fp), (466604 : Fp), (96083 : Fp), (625252 : Fp), (836695 : Fp), (403598 : Fp), (332447 : Fp), (603613 : Fp)]]
def Mf_1_wg : List (List Fp) := [[(603565 : Fp), (472532 : Fp)], [(623141 : Fp), (802579 : Fp)]]
def Mf_1_wu : List (List Fp) := [[(205901 : Fp), (488560 : Fp)], [(64070 : Fp), (976862 : Fp)]]
def Mf_1_wd : List (List Fp) := [[(499678 : Fp), (72441 : Fp)], [(94187 : Fp), (711693 : Fp)]]
def Mf_1_an : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_1_mn : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_1 : Flat := { wq := tab Mf_1_wq, wk := tab Mf_1_wk, wv := tab Mf_1_wv, wo := tab Mf_1_wo, wg := tab Mf_1_wg, wu := tab Mf_1_wu, wd := tab Mf_1_wd, an := tab1 (Mf_1_an.getD 0 []), mn := tab1 (Mf_1_mn.getD 0 []) }
def Mf_1L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 Mf_1
def Mf : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab Mf_emb) (tab Mf_head) (tab1 Mf_gf) [Mf_0L, Mf_1L]
def M0one : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L]
def sg1 : Equiv.Perm (Fin 2) := (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2))
def sgn1 : Fin 2 → Fp := fun i => tab1 [(1000002 : Fp), (1 : Fp)] i.val
def Mh1 := M0.streamGauge (sp sg1 sgn1) (sp sg1.symm (fun k => sgn1 (sg1.symm k))) (fun γ i => γ (sg1 i))
def tQ : List (List Fp) := [[(540227 : Fp), (209033 : Fp)], [(790970 : Fp), (540227 : Fp)]]
def Qpy : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab tQ i.val j.val
def Mh4 := Mf.streamGauge Qpyᵀ Qpy id
def tA0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(995900 : Fp), (439797 : Fp)], [(607854 : Fp), (288579 : Fp)]] i.val j.val
def tA0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(472449 : Fp), (516586 : Fp)], [(692317 : Fp), (672343 : Fp)]] i.val j.val
def tB0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(49627 : Fp), (287566 : Fp)], [(525359 : Fp), (411318 : Fp)]] i.val j.val
def tB0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(213883 : Fp), (164020 : Fp)], [(509302 : Fp), (228898 : Fp)]] i.val j.val
def tA1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(734239 : Fp), (961482 : Fp)], [(831861 : Fp), (374727 : Fp)]] i.val j.val
def tA1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(86374 : Fp), (340079 : Fp)], [(642549 : Fp), (120952 : Fp)]] i.val j.val
def tB1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(929833 : Fp), (432324 : Fp)], [(388013 : Fp), (569618 : Fp)]] i.val j.val
def tB1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(654150 : Fp), (97295 : Fp)], [(648853 : Fp), (671288 : Fp)]] i.val j.val
def M_ov : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [ovGauge E M0_0L ![tA0_0, tA0_1] ![tB0_0, tB0_1], ovGauge E M0_1L ![tA1_0, tA1_1] ![tB1_0, tB1_1]] }
def M_qk : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [qkGauge E M0_0L (fun g p => tab [[(510073 : Fp)], [(615592 : Fp)]] g.val p.val) (fun g p => tab [[(660757 : Fp)], [(351556 : Fp)]] g.val p.val), qkGauge E M0_1L (fun g p => tab [[(886128 : Fp)], [(199626 : Fp)]] g.val p.val) (fun g p => tab [[(254841 : Fp)], [(16996 : Fp)]] g.val p.val)] }
def M_hp : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [headPerm M0_0L (Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4)) (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2)), headPerm M0_1L (Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4)) (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2))] }
def M_m1 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [unitPerm (mlpScale M0_0L (fun u => tab1 [(635792 : Fp), (870409 : Fp)] u.val)) (1 : Equiv.Perm (Fin 2)), unitPerm (mlpScale M0_1L (fun u => tab1 [(603931 : Fp), (125510 : Fp)] u.val)) (1 : Equiv.Perm (Fin 2))] }
def M_n1 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [mlpNorm (attnNorm M0_0L (fun j => tab1 [(410213 : Fp), (95979 : Fp)] j.val)) (fun j => tab1 [(388120 : Fp), (874350 : Fp)] j.val), mlpNorm (attnNorm M0_1L (fun j => tab1 [(121684 : Fp), (38160 : Fp)] j.val)) (fun j => tab1 [(634918 : Fp), (22688 : Fp)] j.val)] }
def main (args : List String) : IO UInt32 := do
  let want : String → Bool := fun s => args.isEmpty || args.contains s
  let t0 ← IO.monoMsNow
  if want "perm" then
    emitNat "sigma" [(List.finRange 2).map fun i => (sg1 i).val]
    emitNat "sigma_heads" [(List.finRange 4).map fun i => (((Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4))) i).val, (List.finRange 4).map fun i => (((Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4))) i).val]
  if want "orig" then
    emitWeights "orig" M0
    emitRun "orig" E M0 toks
    stamp "orig" t0
  if want "fold" then
    emitWeights "fold" Mf
    emitRun "fold" E Mf toks
    stamp "fold" t0
  if want "h1" then
    emitWeights "h1" Mh1
    emitRun "h1" E Mh1 toks
    stamp "h1" t0
  if want "h4" then
    emitWeights "h4" Mh4
    emitRun "h4" E Mh4 toks
    stamp "h4" t0
  if want "ov" then
    emitWeights "ov" M_ov
    emitRun "ov" E M_ov toks
    stamp "ov" t0
  if want "qk" then
    emitWeights "qk" M_qk
    emitRun "qk" E M_qk toks
    stamp "qk" t0
  if want "hp" then
    emitWeights "hp" M_hp
    emitRun "hp" E M_hp toks
    stamp "hp" t0
  if want "m1" then
    emitWeights "m1" M_m1
    emitRun "m1" E M_m1 toks
    stamp "m1" t0
  if want "n1" then
    emitWeights "n1" M_n1
    emitRun "n1" E M_n1 toks
    stamp "n1" t0
  if want "orig1" then
    emitRun "orig1" E M0one toks
    stamp "orig1 tables" t0
    emitLogitsDef "orig1" E M0one toks
    stamp "orig1 definition" t0
  return 0
