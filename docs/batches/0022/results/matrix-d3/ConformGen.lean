-- SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Written by batches/0022/gen22.py; do not edit.
import LmnfProofs.Runtime
open Lmnf Lmnf.Runtime Matrix

def E : Env Fp 2 (Fin 4) (Fin 2) (Fin 1) := mkEnv 2 4 2 1 2 (by intro a; have := a.isLt; omega) [(2 : Fp)]
def toks : Fin 2 → Fin 3 := ![2, 0]
def M0_emb : List (List Fp) := [[(885440 : Fp), (403958 : Fp), (794772 : Fp)], [(933488 : Fp), (441001 : Fp), (42450 : Fp)], [(271493 : Fp), (536110 : Fp), (509532 : Fp)]]
def M0_head : List (List Fp) := [[(870163 : Fp), (318046 : Fp), (499748 : Fp)], [(375441 : Fp), (611720 : Fp), (934973 : Fp)], [(952225 : Fp), (229053 : Fp), (529202 : Fp)]]
def M0_gf : List Fp := [(424605 : Fp), (962839 : Fp), (821873 : Fp)]
def M0_0_wq : List (List Fp) := [[(792518 : Fp), (99437 : Fp), (648406 : Fp)], [(838234 : Fp), (262674 : Fp), (953938 : Fp)], [(558433 : Fp), (739426 : Fp), (849574 : Fp)], [(631140 : Fp), (945989 : Fp), (154100 : Fp)], [(325213 : Fp), (103560 : Fp), (765284 : Fp)], [(77324 : Fp), (942500 : Fp), (891786 : Fp)], [(717209 : Fp), (346236 : Fp), (495077 : Fp)], [(587007 : Fp), (105592 : Fp), (370977 : Fp)]]
def M0_0_wk : List (List Fp) := [[(455262 : Fp), (331556 : Fp), (640561 : Fp)], [(671532 : Fp), (957361 : Fp), (214410 : Fp)], [(579363 : Fp), (500181 : Fp), (464197 : Fp)], [(907343 : Fp), (546678 : Fp), (273145 : Fp)]]
def M0_0_wv : List (List Fp) := [[(65304 : Fp), (844132 : Fp), (963080 : Fp)], [(575352 : Fp), (960489 : Fp), (14723 : Fp)], [(97802 : Fp), (754665 : Fp), (880899 : Fp)], [(418196 : Fp), (744754 : Fp), (864912 : Fp)]]
def M0_0_wo : List (List Fp) := [[(823182 : Fp), (700609 : Fp), (655638 : Fp), (1198 : Fp), (641620 : Fp), (517553 : Fp), (868287 : Fp), (909747 : Fp)], [(349317 : Fp), (255759 : Fp), (765752 : Fp), (341001 : Fp), (737822 : Fp), (912755 : Fp), (66043 : Fp), (200348 : Fp)], [(961564 : Fp), (595078 : Fp), (232473 : Fp), (250206 : Fp), (842368 : Fp), (149416 : Fp), (842194 : Fp), (569366 : Fp)]]
def M0_0_wg : List (List Fp) := [[(335601 : Fp), (917595 : Fp), (532614 : Fp)], [(978147 : Fp), (513054 : Fp), (114355 : Fp)], [(316089 : Fp), (578045 : Fp), (305230 : Fp)]]
def M0_0_wu : List (List Fp) := [[(740883 : Fp), (130873 : Fp), (574033 : Fp)], [(348914 : Fp), (854030 : Fp), (967048 : Fp)], [(566528 : Fp), (213072 : Fp), (838260 : Fp)]]
def M0_0_wd : List (List Fp) := [[(632485 : Fp), (573812 : Fp), (616161 : Fp)], [(301630 : Fp), (466604 : Fp), (96083 : Fp)], [(625252 : Fp), (836695 : Fp), (403598 : Fp)]]
def M0_0_an : List (List Fp) := [[(146040 : Fp), (295529 : Fp), (146535 : Fp)]]
def M0_0_mn : List (List Fp) := [[(469731 : Fp), (95647 : Fp), (84354 : Fp)]]
def M0_0 : Flat := { wq := tab M0_0_wq, wk := tab M0_0_wk, wv := tab M0_0_wv, wo := tab M0_0_wo, wg := tab M0_0_wg, wu := tab M0_0_wu, wd := tab M0_0_wd, an := tab1 (M0_0_an.getD 0 []), mn := tab1 (M0_0_mn.getD 0 []) }
def M0_0L : Layer Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) := toLayer 3 2 1 4 2 3 M0_0
def M0_1_wq : List (List Fp) := [[(304432 : Fp), (192800 : Fp), (198591 : Fp)], [(861370 : Fp), (195800 : Fp), (34574 : Fp)], [(642539 : Fp), (688557 : Fp), (272688 : Fp)], [(499678 : Fp), (72441 : Fp), (94187 : Fp)], [(711693 : Fp), (794405 : Fp), (136550 : Fp)], [(919360 : Fp), (156814 : Fp), (968235 : Fp)], [(40518 : Fp), (883383 : Fp), (84146 : Fp)], [(941802 : Fp), (733293 : Fp), (967922 : Fp)]]
def M0_1_wk : List (List Fp) := [[(869647 : Fp), (566860 : Fp), (716700 : Fp)], [(410303 : Fp), (878565 : Fp), (739543 : Fp)], [(550055 : Fp), (289023 : Fp), (547136 : Fp)], [(851054 : Fp), (246941 : Fp), (890750 : Fp)]]
def M0_1_wv : List (List Fp) := [[(225654 : Fp), (938516 : Fp), (712480 : Fp)], [(618451 : Fp), (865351 : Fp), (995900 : Fp)], [(439797 : Fp), (607854 : Fp), (288579 : Fp)], [(472449 : Fp), (516586 : Fp), (692317 : Fp)]]
def M0_1_wo : List (List Fp) := [[(672343 : Fp), (734239 : Fp), (961482 : Fp), (831861 : Fp), (374727 : Fp), (86374 : Fp), (340079 : Fp), (642549 : Fp)], [(120952 : Fp), (510073 : Fp), (615592 : Fp), (660757 : Fp), (351556 : Fp), (886128 : Fp), (199626 : Fp), (254841 : Fp)], [(16996 : Fp), (767022 : Fp), (284203 : Fp), (122824 : Fp), (739595 : Fp), (231169 : Fp), (390133 : Fp), (833180 : Fp)]]
def M0_1_wg : List (List Fp) := [[(855546 : Fp), (65213 : Fp), (105494 : Fp)], [(821159 : Fp), (153467 : Fp), (896870 : Fp)], [(731560 : Fp), (229400 : Fp), (47431 : Fp)]]
def M0_1_wu : List (List Fp) := [[(856812 : Fp), (601742 : Fp), (665013 : Fp)], [(954220 : Fp), (982012 : Fp), (560147 : Fp)], [(631421 : Fp), (713649 : Fp), (77591 : Fp)]]
def M0_1_wd : List (List Fp) := [[(27993 : Fp), (130488 : Fp), (665845 : Fp)], [(197678 : Fp), (635791 : Fp), (870408 : Fp)], [(603930 : Fp), (125509 : Fp), (410212 : Fp)]]
def M0_1_an : List (List Fp) := [[(332448 : Fp), (603614 : Fp), (253868 : Fp)]]
def M0_1_mn : List (List Fp) := [[(178764 : Fp), (348690 : Fp), (446831 : Fp)]]
def M0_1 : Flat := { wq := tab M0_1_wq, wk := tab M0_1_wk, wv := tab M0_1_wv, wo := tab M0_1_wo, wg := tab M0_1_wg, wu := tab M0_1_wu, wd := tab M0_1_wd, an := tab1 (M0_1_an.getD 0 []), mn := tab1 (M0_1_mn.getD 0 []) }
def M0_1L : Layer Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) := toLayer 3 2 1 4 2 3 M0_1
def M0 : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 3 2 1 4 2 3 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L, M0_1L]
def Mf_emb : List (List Fp) := [[(885440 : Fp), (403958 : Fp), (794772 : Fp)], [(933488 : Fp), (441001 : Fp), (42450 : Fp)], [(271493 : Fp), (536110 : Fp), (509532 : Fp)]]
def Mf_head : List (List Fp) := [[(452193 : Fp), (173916 : Fp), (155820 : Fp)], [(647566 : Fp), (106122 : Fp), (759151 : Fp)], [(283171 : Fp), (499847 : Fp), (530541 : Fp)]]
def Mf_gf : List Fp := [(1 : Fp), (1 : Fp), (1 : Fp)]
def Mf_0_wq : List (List Fp) := [[(981506 : Fp), (429015 : Fp), (888171 : Fp)], [(326115 : Fp), (551665 : Fp), (885478 : Fp)], [(310661 : Fp), (170791 : Fp), (952617 : Fp)], [(409087 : Fp), (344483 : Fp), (975760 : Fp)], [(964041 : Fp), (891428 : Fp), (554520 : Fp)], [(363084 : Fp), (246895 : Fp), (469479 : Fp)], [(888140 : Fp), (471878 : Fp), (890560 : Fp)], [(245102 : Fp), (404553 : Fp), (951615 : Fp)]]
def Mf_0_wk : List (List Fp) := [[(263022 : Fp), (119172 : Fp), (324543 : Fp)], [(239070 : Fp), (90188 : Fp), (475096 : Fp)], [(918693 : Fp), (547298 : Fp), (903335 : Fp)], [(974199 : Fp), (717988 : Fp), (182500 : Fp)]]
def Mf_0_wv : List (List Fp) := [[(967552 : Fp), (737436 : Fp), (504428 : Fp)], [(154008 : Fp), (502128 : Fp), (428334 : Fp)], [(961234 : Fp), (723713 : Fp), (147719 : Fp)], [(160621 : Fp), (744581 : Fp), (499703 : Fp)]]
def Mf_0_wo : List (List Fp) := [[(823182 : Fp), (700609 : Fp), (655638 : Fp), (1198 : Fp), (641620 : Fp), (517553 : Fp), (868287 : Fp), (909747 : Fp)], [(349317 : Fp), (255759 : Fp), (765752 : Fp), (341001 : Fp), (737822 : Fp), (912755 : Fp), (66043 : Fp), (200348 : Fp)], [(961564 : Fp), (595078 : Fp), (232473 : Fp), (250206 : Fp), (842368 : Fp), (149416 : Fp), (842194 : Fp), (569366 : Fp)]]
def Mf_0_wg : List (List Fp) := [[(720408 : Fp), (945673 : Fp), (986575 : Fp)], [(590065 : Fp), (928725 : Fp), (272732 : Fp)], [(356631 : Fp), (104251 : Fp), (294179 : Fp)]]
def Mf_0_wu : List (List Fp) := [[(668431 : Fp), (572280 : Fp), (834419 : Fp)], [(230449 : Fp), (162355 : Fp), (122270 : Fp)], [(965626 : Fp), (636447 : Fp), (371910 : Fp)]]
def Mf_0_wd : List (List Fp) := [[(632485 : Fp), (573812 : Fp), (616161 : Fp)], [(301630 : Fp), (466604 : Fp), (96083 : Fp)], [(625252 : Fp), (836695 : Fp), (403598 : Fp)]]
def Mf_0_an : List (List Fp) := [[(1 : Fp), (1 : Fp), (1 : Fp)]]
def Mf_0_mn : List (List Fp) := [[(1 : Fp), (1 : Fp), (1 : Fp)]]
def Mf_0 : Flat := { wq := tab Mf_0_wq, wk := tab Mf_0_wk, wv := tab Mf_0_wv, wo := tab Mf_0_wo, wg := tab Mf_0_wg, wu := tab Mf_0_wu, wd := tab Mf_0_wd, an := tab1 (Mf_0_an.getD 0 []), mn := tab1 (Mf_0_mn.getD 0 []) }
def Mf_0L : Layer Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) := toLayer 3 2 1 4 2 3 Mf_0
def Mf_1_wq : List (List Fp) := [[(505915 : Fp), (430072 : Fp), (748743 : Fp)], [(874683 : Fp), (266639 : Fp), (205901 : Fp)], [(164642 : Fp), (398135 : Fp), (549506 : Fp)], [(453396 : Fp), (270596 : Fp), (993586 : Fp)], [(204664 : Fp), (541134 : Fp), (571405 : Fp)], [(476366 : Fp), (841834 : Fp), (145571 : Fp)], [(87654 : Fp), (746502 : Fp), (912645 : Fp)], [(251999 : Fp), (593030 : Fp), (685127 : Fp)]]
def Mf_1_wk : List (List Fp) := [[(538523 : Fp), (605551 : Fp), (649762 : Fp)], [(2532 : Fp), (542974 : Fp), (739089 : Fp)], [(136048 : Fp), (805751 : Fp), (905351 : Fp)], [(351402 : Fp), (597606 : Fp), (242604 : Fp)]]
def Mf_1_wv : List (List Fp) := [[(995941 : Fp), (697327 : Fp), (330015 : Fp)], [(181242 : Fp), (411506 : Fp), (382722 : Fp)], [(194429 : Fp), (83632 : Fp), (753792 : Fp)], [(253960 : Fp), (606353 : Fp), (604888 : Fp)]]
def Mf_1_wo : List (List Fp) := [[(672343 : Fp), (734239 : Fp), (961482 : Fp), (831861 : Fp), (374727 : Fp), (86374 : Fp), (340079 : Fp), (642549 : Fp)], [(120952 : Fp), (510073 : Fp), (615592 : Fp), (660757 : Fp), (351556 : Fp), (886128 : Fp), (199626 : Fp), (254841 : Fp)], [(16996 : Fp), (767022 : Fp), (284203 : Fp), (122824 : Fp), (739595 : Fp), (231169 : Fp), (390133 : Fp), (833180 : Fp)]]
def Mf_1_wg : List (List Fp) := [[(366324 : Fp), (52753 : Fp), (848103 : Fp)], [(227097 : Fp), (247694 : Fp), (116726 : Fp)], [(199512 : Fp), (246033 : Fp), (577582 : Fp)]]
def Mf_1_wu : List (List Fp) := [[(680870 : Fp), (788520 : Fp), (532362 : Fp)], [(672343 : Fp), (737032 : Fp), (293287 : Fp)], [(5019 : Fp), (523287 : Fp), (960114 : Fp)]]
def Mf_1_wd : List (List Fp) := [[(27993 : Fp), (130488 : Fp), (665845 : Fp)], [(197678 : Fp), (635791 : Fp), (870408 : Fp)], [(603930 : Fp), (125509 : Fp), (410212 : Fp)]]
def Mf_1_an : List (List Fp) := [[(1 : Fp), (1 : Fp), (1 : Fp)]]
def Mf_1_mn : List (List Fp) := [[(1 : Fp), (1 : Fp), (1 : Fp)]]
def Mf_1 : Flat := { wq := tab Mf_1_wq, wk := tab Mf_1_wk, wv := tab Mf_1_wv, wo := tab Mf_1_wo, wg := tab Mf_1_wg, wu := tab Mf_1_wu, wd := tab Mf_1_wd, an := tab1 (Mf_1_an.getD 0 []), mn := tab1 (Mf_1_mn.getD 0 []) }
def Mf_1L : Layer Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) := toLayer 3 2 1 4 2 3 Mf_1
def Mf : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 3 2 1 4 2 3 3 (tab Mf_emb) (tab Mf_head) (tab1 Mf_gf) [Mf_0L, Mf_1L]
def M0one : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 3 2 1 4 2 3 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L]
def sg1 : Equiv.Perm (Fin 3) := (Equiv.swap (1 : Fin 3) 2 : Equiv.Perm (Fin 3))
def sgn1 : Fin 3 → Fp := fun i => tab1 [(1 : Fp), (1 : Fp), (1000002 : Fp)] i.val
def Mh1 := M0.streamGauge (sp sg1 sgn1) (sp sg1.symm (fun k => sgn1 (sg1.symm k))) (fun γ i => γ (sg1 i))
def tQ : List (List Fp) := [[(205379 : Fp), (773386 : Fp), (804607 : Fp)], [(831180 : Fp), (317892 : Fp), (204646 : Fp)], [(515436 : Fp), (375203 : Fp), (98187 : Fp)]]
def Qpy : Matrix (Fin 3) (Fin 3) Fp := Matrix.of fun i j => tab tQ i.val j.val
def Mh4 := Mf.streamGauge Qpyᵀ Qpy id
def tA0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(23889 : Fp), (570672 : Fp)], [(446293 : Fp), (650746 : Fp)]] i.val j.val
def tA0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(106430 : Fp), (876507 : Fp)], [(272545 : Fp), (73404 : Fp)]] i.val j.val
def tB0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(537141 : Fp), (439610 : Fp)], [(723073 : Fp), (577749 : Fp)]] i.val j.val
def tB0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(366670 : Fp), (977799 : Fp)], [(379929 : Fp), (353995 : Fp)]] i.val j.val
def tA1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(231556 : Fp), (75467 : Fp)], [(678350 : Fp), (315685 : Fp)]] i.val j.val
def tA1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(367309 : Fp), (457251 : Fp)], [(189077 : Fp), (64007 : Fp)]] i.val j.val
def tB1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(553664 : Fp), (399142 : Fp)], [(89419 : Fp), (431906 : Fp)]] i.val j.val
def tB1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(517072 : Fp), (859612 : Fp)], [(371146 : Fp), (743462 : Fp)]] i.val j.val
def M_ov : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [ovGauge E M0_0L ![tA0_0, tA0_1] ![tB0_0, tB0_1], ovGauge E M0_1L ![tA1_0, tA1_1] ![tB1_0, tB1_1]] }
def M_qk : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [qkGauge E M0_0L (fun g p => tab [[(528101 : Fp)], [(489822 : Fp)]] g.val p.val) (fun g p => tab [[(41291 : Fp)], [(625459 : Fp)]] g.val p.val), qkGauge E M0_1L (fun g p => tab [[(105823 : Fp)], [(733293 : Fp)]] g.val p.val) (fun g p => tab [[(410282 : Fp)], [(209039 : Fp)]] g.val p.val)] }
def M_hp : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [headPerm M0_0L (Equiv.swap (2 : Fin 4) 3 * Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4)) (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2)), headPerm M0_1L (Equiv.swap (0 : Fin 4) 1 : Equiv.Perm (Fin 4)) (1 : Equiv.Perm (Fin 2))] }
def M_m1 : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [unitPerm (mlpScale M0_0L (fun u => tab1 [(680457 : Fp), (374501 : Fp), (407521 : Fp)] u.val)) (Equiv.swap (1 : Fin 3) 2 * Equiv.swap (0 : Fin 3) 1 : Equiv.Perm (Fin 3)), unitPerm (mlpScale M0_1L (fun u => tab1 [(878352 : Fp), (689561 : Fp), (263122 : Fp)] u.val)) (Equiv.swap (1 : Fin 3) 2 * Equiv.swap (0 : Fin 3) 2 : Equiv.Perm (Fin 3))] }
def M_n1 : Model Fp 3 2 3 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [mlpNorm (attnNorm M0_0L (fun j => tab1 [(160865 : Fp), (587832 : Fp), (724381 : Fp)] j.val)) (fun j => tab1 [(13041 : Fp), (480200 : Fp), (777598 : Fp)] j.val), mlpNorm (attnNorm M0_1L (fun j => tab1 [(82915 : Fp), (352235 : Fp), (774915 : Fp)] j.val)) (fun j => tab1 [(47917 : Fp), (570761 : Fp), (294528 : Fp)] j.val)] }
def main (args : List String) : IO UInt32 := do
  let want : String → Bool := fun s => args.isEmpty || args.contains s
  let t0 ← IO.monoMsNow
  if want "perm" then
    emitNat "sigma" [(List.finRange 3).map fun i => (sg1 i).val]
    emitNat "sigma_heads" [(List.finRange 4).map fun i => (((Equiv.swap (2 : Fin 4) 3 * Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4))) i).val, (List.finRange 4).map fun i => (((Equiv.swap (0 : Fin 4) 1 : Equiv.Perm (Fin 4))) i).val]
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
