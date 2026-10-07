-- SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Written by batches/0022/gen22.py; do not edit.
import LmnfProofs.Runtime
open Lmnf Lmnf.Runtime Matrix

def E : Env Fp 2 (Fin 4) (Fin 2) (Fin 1) := mkEnv 2 4 2 1 2 (by intro a; have := a.isLt; omega) [(2 : Fp)]
def toks : Fin 2 → Fin 3 := ![2, 0]
def M0_emb : List (List Fp) := [[(152745 : Fp), (481850 : Fp)], [(477025 : Fp), (997948 : Fp)], [(808225 : Fp), (183236 : Fp)]]
def M0_head : List (List Fp) := [[(767514 : Fp), (366725 : Fp)], [(454572 : Fp), (531486 : Fp)], [(838882 : Fp), (115311 : Fp)]]
def M0_gf : List Fp := [(739785 : Fp), (412126 : Fp)]
def M0_0_wq : List (List Fp) := [[(84047 : Fp), (772497 : Fp)], [(478093 : Fp), (276212 : Fp)], [(50213 : Fp), (690818 : Fp)], [(978281 : Fp), (678953 : Fp)], [(993083 : Fp), (214418 : Fp)], [(351640 : Fp), (240863 : Fp)], [(324435 : Fp), (879655 : Fp)], [(805325 : Fp), (213772 : Fp)]]
def M0_0_wk : List (List Fp) := [[(187129 : Fp), (147672 : Fp)], [(197756 : Fp), (936225 : Fp)], [(363860 : Fp), (388038 : Fp)], [(656745 : Fp), (429594 : Fp)]]
def M0_0_wv : List (List Fp) := [[(875423 : Fp), (221075 : Fp)], [(422646 : Fp), (483507 : Fp)], [(582378 : Fp), (289175 : Fp)], [(944926 : Fp), (830814 : Fp)]]
def M0_0_wo : List (List Fp) := [[(857754 : Fp), (393319 : Fp), (168214 : Fp), (898087 : Fp), (680046 : Fp), (669469 : Fp), (130121 : Fp), (189422 : Fp)], [(5919 : Fp), (632041 : Fp), (414728 : Fp), (154774 : Fp), (811074 : Fp), (892860 : Fp), (593659 : Fp), (170435 : Fp)]]
def M0_0_wg : List (List Fp) := [[(882035 : Fp), (27404 : Fp)], [(958164 : Fp), (698043 : Fp)]]
def M0_0_wu : List (List Fp) := [[(248748 : Fp), (469963 : Fp)], [(838234 : Fp), (668781 : Fp)]]
def M0_0_wd : List (List Fp) := [[(967118 : Fp), (946175 : Fp)], [(406491 : Fp), (132765 : Fp)]]
def M0_0_an : List (List Fp) := [[(558788 : Fp), (127205 : Fp)]]
def M0_0_mn : List (List Fp) := [[(201999 : Fp), (175128 : Fp)]]
def M0_0 : Flat := { wq := tab M0_0_wq, wk := tab M0_0_wk, wv := tab M0_0_wv, wo := tab M0_0_wo, wg := tab M0_0_wg, wu := tab M0_0_wu, wd := tab M0_0_wd, an := tab1 (M0_0_an.getD 0 []), mn := tab1 (M0_0_mn.getD 0 []) }
def M0_0L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 M0_0
def M0_1_wq : List (List Fp) := [[(793626 : Fp), (55005 : Fp)], [(638687 : Fp), (259093 : Fp)], [(752147 : Fp), (648895 : Fp)], [(812599 : Fp), (973571 : Fp)], [(531746 : Fp), (748623 : Fp)], [(303935 : Fp), (723266 : Fp)], [(619052 : Fp), (355168 : Fp)], [(553836 : Fp), (714932 : Fp)]]
def M0_1_wk : List (List Fp) := [[(359942 : Fp), (568831 : Fp)], [(895663 : Fp), (989174 : Fp)], [(380631 : Fp), (507053 : Fp)], [(421250 : Fp), (798492 : Fp)]]
def M0_1_wv : List (List Fp) := [[(64952 : Fp), (853350 : Fp)], [(106244 : Fp), (933711 : Fp)], [(314467 : Fp), (683154 : Fp)], [(455562 : Fp), (264799 : Fp)]]
def M0_1_wo : List (List Fp) := [[(251445 : Fp), (888534 : Fp), (864019 : Fp), (953466 : Fp), (741236 : Fp), (749463 : Fp), (458678 : Fp), (804213 : Fp)], [(931032 : Fp), (538526 : Fp), (840023 : Fp), (517453 : Fp), (945479 : Fp), (610988 : Fp), (266558 : Fp), (154338 : Fp)]]
def M0_1_wg : List (List Fp) := [[(613786 : Fp), (903181 : Fp)], [(342630 : Fp), (172705 : Fp)]]
def M0_1_wu : List (List Fp) := [[(57237 : Fp), (790497 : Fp)], [(288516 : Fp), (740485 : Fp)]]
def M0_1_wd : List (List Fp) := [[(729787 : Fp), (126881 : Fp)], [(752671 : Fp), (53184 : Fp)]]
def M0_1_an : List (List Fp) := [[(654178 : Fp), (580861 : Fp)]]
def M0_1_mn : List (List Fp) := [[(61048 : Fp), (708812 : Fp)]]
def M0_1 : Flat := { wq := tab M0_1_wq, wk := tab M0_1_wk, wv := tab M0_1_wv, wo := tab M0_1_wo, wg := tab M0_1_wg, wu := tab M0_1_wu, wd := tab M0_1_wd, an := tab1 (M0_1_an.getD 0 []), mn := tab1 (M0_1_mn.getD 0 []) }
def M0_1L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 M0_1
def M0 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L, M0_1L]
def Mf_emb : List (List Fp) := [[(152745 : Fp), (481850 : Fp)], [(477025 : Fp), (997948 : Fp)], [(808225 : Fp), (183236 : Fp)]]
def Mf_head : List (List Fp) := [[(641111 : Fp), (453942 : Fp)], [(538168 : Fp), (542122 : Fp)], [(458600 : Fp), (518620 : Fp)]]
def Mf_gf : List Fp := [(1 : Fp), (1 : Fp)]
def Mf_0_wq : List (List Fp) := [[(314144 : Fp), (186090 : Fp)], [(829831 : Fp), (442055 : Fp)], [(337670 : Fp), (240065 : Fp)], [(43478 : Fp), (957270 : Fp)], [(198641 : Fp), (959868 : Fp)], [(622847 : Fp), (886001 : Fp)], [(840913 : Fp), (178587 : Fp)], [(596088 : Fp), (785684 : Fp)]]
def Mf_0_wk : List (List Fp) := [[(125957 : Fp), (560408 : Fp)], [(348219 : Fp), (143849 : Fp)], [(991723 : Fp), (225710 : Fp)], [(124120 : Fp), (340832 : Fp)]]
def Mf_0_wv : List (List Fp) := [[(399802 : Fp), (761012 : Fp)], [(804544 : Fp), (323423 : Fp)], [(861592 : Fp), (395523 : Fp)], [(725655 : Fp), (377821 : Fp)]]
def Mf_0_wo : List (List Fp) := [[(857754 : Fp), (393319 : Fp), (168214 : Fp), (898087 : Fp), (680046 : Fp), (669469 : Fp), (130121 : Fp), (189422 : Fp)], [(5919 : Fp), (632041 : Fp), (414728 : Fp), (154774 : Fp), (811074 : Fp), (892860 : Fp), (593659 : Fp), (170435 : Fp)]]
def Mf_0_wg : List (List Fp) := [[(653458 : Fp), (193315 : Fp)], [(589195 : Fp), (507766 : Fp)]]
def Mf_0_wu : List (List Fp) := [[(696514 : Fp), (433355 : Fp)], [(921803 : Fp), (927605 : Fp)]]
def Mf_0_wd : List (List Fp) := [[(967118 : Fp), (946175 : Fp)], [(406491 : Fp), (132765 : Fp)]]
def Mf_0_an : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_0_mn : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_0 : Flat := { wq := tab Mf_0_wq, wk := tab Mf_0_wk, wv := tab Mf_0_wv, wo := tab Mf_0_wo, wg := tab Mf_0_wg, wu := tab Mf_0_wu, wd := tab Mf_0_wd, an := tab1 (Mf_0_an.getD 0 []), mn := tab1 (Mf_0_mn.getD 0 []) }
def Mf_0L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 Mf_0
def Mf_1_wq : List (List Fp) := [[(111915 : Fp), (163455 : Fp)], [(730847 : Fp), (567585 : Fp)], [(544058 : Fp), (667847 : Fp)], [(793876 : Fp), (728110 : Fp)], [(491223 : Fp), (599871 : Fp)], [(993952 : Fp), (751681 : Fp)], [(984352 : Fp), (620742 : Fp)], [(239890 : Fp), (870630 : Fp)]]
def Mf_1_wk : List (List Fp) := [[(431281 : Fp), (752261 : Fp)], [(272251 : Fp), (875104 : Fp)], [(679321 : Fp), (429055 : Fp)], [(655787 : Fp), (470179 : Fp)]]
def Mf_1_wv : List (List Fp) := [[(41986 : Fp), (247322 : Fp)], [(278926 : Fp), (678109 : Fp)], [(775978 : Fp), (325146 : Fp)], [(743985 : Fp), (950509 : Fp)]]
def Mf_1_wo : List (List Fp) := [[(251445 : Fp), (888534 : Fp), (864019 : Fp), (953466 : Fp), (741236 : Fp), (749463 : Fp), (458678 : Fp), (804213 : Fp)], [(931032 : Fp), (538526 : Fp), (840023 : Fp), (517453 : Fp), (945479 : Fp), (610988 : Fp), (266558 : Fp), (154338 : Fp)]]
def Mf_1_wg : List (List Fp) := [[(295318 : Fp), (610423 : Fp)], [(813492 : Fp), (9215 : Fp)]]
def Mf_1_wu : List (List Fp) := [[(193894 : Fp), (78628 : Fp)], [(271929 : Fp), (79231 : Fp)]]
def Mf_1_wd : List (List Fp) := [[(729787 : Fp), (126881 : Fp)], [(752671 : Fp), (53184 : Fp)]]
def Mf_1_an : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_1_mn : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_1 : Flat := { wq := tab Mf_1_wq, wk := tab Mf_1_wk, wv := tab Mf_1_wv, wo := tab Mf_1_wo, wg := tab Mf_1_wg, wu := tab Mf_1_wu, wd := tab Mf_1_wd, an := tab1 (Mf_1_an.getD 0 []), mn := tab1 (Mf_1_mn.getD 0 []) }
def Mf_1L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 Mf_1
def Mf : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab Mf_emb) (tab Mf_head) (tab1 Mf_gf) [Mf_0L, Mf_1L]
def M0one : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L]
def sg1 : Equiv.Perm (Fin 2) := (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2))
def sgn1 : Fin 2 → Fp := fun i => tab1 [(1000002 : Fp), (1 : Fp)] i.val
def Mh1 := M0.streamGauge (sp sg1 sgn1) (sp sg1.symm (fun k => sgn1 (sg1.symm k))) (fun γ i => γ (sg1 i))
def tQ : List (List Fp) := [[(239664 : Fp), (955077 : Fp)], [(44926 : Fp), (239664 : Fp)]]
def Qpy : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab tQ i.val j.val
def Mh4 := Mf.streamGauge Qpyᵀ Qpy id
def tA0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(124870 : Fp), (755086 : Fp)], [(624602 : Fp), (219305 : Fp)]] i.val j.val
def tA0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(46077 : Fp), (408782 : Fp)], [(760066 : Fp), (981904 : Fp)]] i.val j.val
def tB0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(928683 : Fp), (824756 : Fp)], [(825663 : Fp), (899295 : Fp)]] i.val j.val
def tB0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(998247 : Fp), (105320 : Fp)], [(176603 : Fp), (636662 : Fp)]] i.val j.val
def tA1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(109083 : Fp), (420069 : Fp)], [(711141 : Fp), (747853 : Fp)]] i.val j.val
def tA1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(158855 : Fp), (648527 : Fp)], [(691781 : Fp), (674687 : Fp)]] i.val j.val
def tB1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(437525 : Fp), (524551 : Fp)], [(225327 : Fp), (123929 : Fp)]] i.val j.val
def tB1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(882393 : Fp), (835550 : Fp)], [(864079 : Fp), (542357 : Fp)]] i.val j.val
def M_ov : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [ovGauge E M0_0L ![tA0_0, tA0_1] ![tB0_0, tB0_1], ovGauge E M0_1L ![tA1_0, tA1_1] ![tB1_0, tB1_1]] }
def M_qk : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [qkGauge E M0_0L (fun g p => tab [[(499867 : Fp)], [(980153 : Fp)]] g.val p.val) (fun g p => tab [[(571566 : Fp)], [(40761 : Fp)]] g.val p.val), qkGauge E M0_1L (fun g p => tab [[(734878 : Fp)], [(466861 : Fp)]] g.val p.val) (fun g p => tab [[(168953 : Fp)], [(51774 : Fp)]] g.val p.val)] }
def M_hp : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [headPerm M0_0L (Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4)) (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2)), headPerm M0_1L (1 : Equiv.Perm (Fin 4)) (1 : Equiv.Perm (Fin 2))] }
def M_m1 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [unitPerm (mlpScale M0_0L (fun u => tab1 [(497157 : Fp), (472390 : Fp)] u.val)) (1 : Equiv.Perm (Fin 2)), unitPerm (mlpScale M0_1L (fun u => tab1 [(360360 : Fp), (729248 : Fp)] u.val)) (1 : Equiv.Perm (Fin 2))] }
def M_n1 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [mlpNorm (attnNorm M0_0L (fun j => tab1 [(476283 : Fp), (549054 : Fp)] j.val)) (fun j => tab1 [(203637 : Fp), (635921 : Fp)] j.val), mlpNorm (attnNorm M0_1L (fun j => tab1 [(607899 : Fp), (898489 : Fp)] j.val)) (fun j => tab1 [(263422 : Fp), (438058 : Fp)] j.val)] }
def main (args : List String) : IO UInt32 := do
  let want : String → Bool := fun s => args.isEmpty || args.contains s
  let t0 ← IO.monoMsNow
  if want "perm" then
    emitNat "sigma" [(List.finRange 2).map fun i => (sg1 i).val]
    emitNat "sigma_heads" [(List.finRange 4).map fun i => (((Equiv.swap (1 : Fin 4) 3 * Equiv.swap (0 : Fin 4) 2 : Equiv.Perm (Fin 4))) i).val, (List.finRange 4).map fun i => (((1 : Equiv.Perm (Fin 4))) i).val]
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
