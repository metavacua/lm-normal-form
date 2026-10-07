-- SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
-- SPDX-License-Identifier: AGPL-3.0-or-later
-- Written by batches/0022/gen22.py; do not edit.
import LmnfProofs.Runtime
open Lmnf Lmnf.Runtime Matrix

def E : Env Fp 2 (Fin 4) (Fin 2) (Fin 1) := mkEnv 2 4 2 1 2 (by intro a; have := a.isLt; omega) [(2 : Fp)]
def toks : Fin 2 → Fin 3 := ![2, 0]
def M0_emb : List (List Fp) := [[(47824 : Fp), (945361 : Fp)], [(213329 : Fp), (770183 : Fp)], [(743570 : Fp), (31837 : Fp)]]
def M0_head : List (List Fp) := [[(901786 : Fp), (837506 : Fp)], [(279668 : Fp), (960876 : Fp)], [(12030 : Fp), (725451 : Fp)]]
def M0_gf : List Fp := [(149788 : Fp), (647717 : Fp)]
def M0_0_wq : List (List Fp) := [[(175332 : Fp), (753539 : Fp)], [(17014 : Fp), (459911 : Fp)], [(291647 : Fp), (460790 : Fp)], [(482135 : Fp), (243125 : Fp)], [(202224 : Fp), (679809 : Fp)], [(894251 : Fp), (126946 : Fp)], [(314598 : Fp), (738603 : Fp)], [(734870 : Fp), (406108 : Fp)]]
def M0_0_wk : List (List Fp) := [[(367997 : Fp), (680762 : Fp)], [(105845 : Fp), (485936 : Fp)], [(455697 : Fp), (241008 : Fp)], [(718961 : Fp), (992098 : Fp)]]
def M0_0_wv : List (List Fp) := [[(510664 : Fp), (308715 : Fp)], [(673391 : Fp), (750239 : Fp)], [(127652 : Fp), (771419 : Fp)], [(146696 : Fp), (904883 : Fp)]]
def M0_0_wo : List (List Fp) := [[(136454 : Fp), (505471 : Fp), (515551 : Fp), (168347 : Fp), (902805 : Fp), (198937 : Fp), (401766 : Fp), (511244 : Fp)], [(860126 : Fp), (995878 : Fp), (116604 : Fp), (594577 : Fp), (236335 : Fp), (573727 : Fp), (705991 : Fp), (681121 : Fp)]]
def M0_0_wg : List (List Fp) := [[(832202 : Fp), (129319 : Fp)], [(152744 : Fp), (961637 : Fp)]]
def M0_0_wu : List (List Fp) := [[(567801 : Fp), (682470 : Fp)], [(249029 : Fp), (657688 : Fp)]]
def M0_0_wd : List (List Fp) := [[(785762 : Fp), (48084 : Fp)], [(608276 : Fp), (721969 : Fp)]]
def M0_0_an : List (List Fp) := [[(461343 : Fp), (744348 : Fp)]]
def M0_0_mn : List (List Fp) := [[(823751 : Fp), (341581 : Fp)]]
def M0_0 : Flat := { wq := tab M0_0_wq, wk := tab M0_0_wk, wv := tab M0_0_wv, wo := tab M0_0_wo, wg := tab M0_0_wg, wu := tab M0_0_wu, wd := tab M0_0_wd, an := tab1 (M0_0_an.getD 0 []), mn := tab1 (M0_0_mn.getD 0 []) }
def M0_0L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 M0_0
def M0_1_wq : List (List Fp) := [[(177512 : Fp), (492654 : Fp)], [(488890 : Fp), (613349 : Fp)], [(719209 : Fp), (303846 : Fp)], [(295504 : Fp), (924288 : Fp)], [(226477 : Fp), (110764 : Fp)], [(207153 : Fp), (751144 : Fp)], [(332508 : Fp), (940322 : Fp)], [(264452 : Fp), (582033 : Fp)]]
def M0_1_wk : List (List Fp) := [[(840113 : Fp), (378605 : Fp)], [(69377 : Fp), (865738 : Fp)], [(853898 : Fp), (269840 : Fp)], [(749249 : Fp), (539655 : Fp)]]
def M0_1_wv : List (List Fp) := [[(361806 : Fp), (998656 : Fp)], [(540258 : Fp), (2025 : Fp)], [(147190 : Fp), (554689 : Fp)], [(213727 : Fp), (322712 : Fp)]]
def M0_1_wo : List (List Fp) := [[(744539 : Fp), (708184 : Fp), (73039 : Fp), (20598 : Fp), (469677 : Fp), (335124 : Fp), (123502 : Fp), (324654 : Fp)], [(362349 : Fp), (943964 : Fp), (14319 : Fp), (796129 : Fp), (274404 : Fp), (676229 : Fp), (904204 : Fp), (380925 : Fp)]]
def M0_1_wg : List (List Fp) := [[(255610 : Fp), (32497 : Fp)], [(547696 : Fp), (572299 : Fp)]]
def M0_1_wu : List (List Fp) := [[(930353 : Fp), (711693 : Fp)], [(771335 : Fp), (449801 : Fp)]]
def M0_1_wd : List (List Fp) := [[(222269 : Fp), (502587 : Fp)], [(502206 : Fp), (236251 : Fp)]]
def M0_1_an : List (List Fp) := [[(373938 : Fp), (821809 : Fp)]]
def M0_1_mn : List (List Fp) := [[(288860 : Fp), (103703 : Fp)]]
def M0_1 : Flat := { wq := tab M0_1_wq, wk := tab M0_1_wk, wv := tab M0_1_wv, wo := tab M0_1_wo, wg := tab M0_1_wg, wu := tab M0_1_wu, wd := tab M0_1_wd, an := tab1 (M0_1_an.getD 0 []), mn := tab1 (M0_1_mn.getD 0 []) }
def M0_1L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 M0_1
def M0 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L, M0_1L]
def Mf_emb : List (List Fp) := [[(47824 : Fp), (945361 : Fp)], [(213329 : Fp), (770183 : Fp)], [(743570 : Fp), (31837 : Fp)]]
def Mf_head : List (List Fp) := [[(316140 : Fp), (246407 : Fp)], [(784714 : Fp), (852973 : Fp)], [(944237 : Fp), (535712 : Fp)]]
def Mf_gf : List Fp := [(1 : Fp), (1 : Fp)]
def Mf_0_wq : List (List Fp) := [[(948215 : Fp), (564893 : Fp)], [(266255 : Fp), (806032 : Fp)], [(898277 : Fp), (85959 : Fp)], [(940021 : Fp), (64593 : Fp)], [(346950 : Fp), (951496 : Fp)], [(201428 : Fp), (717735 : Fp)], [(149703 : Fp), (16516 : Fp)], [(113332 : Fp), (770732 : Fp)]]
def Mf_0_wk : List (List Fp) := [[(330655 : Fp), (313010 : Fp)], [(703345 : Fp), (404616 : Fp)], [(990378 : Fp), (284605 : Fp)], [(629565 : Fp), (946715 : Fp)]]
def Mf_0_wv : List (List Fp) := [[(554982 : Fp), (703450 : Fp)], [(292124 : Fp), (223861 : Fp)], [(179963 : Fp), (467206 : Fp)], [(969700 : Fp), (830649 : Fp)]]
def Mf_0_wo : List (List Fp) := [[(136454 : Fp), (505471 : Fp), (515551 : Fp), (168347 : Fp), (902805 : Fp), (198937 : Fp), (401766 : Fp), (511244 : Fp)], [(860126 : Fp), (995878 : Fp), (116604 : Fp), (594577 : Fp), (236335 : Fp), (573727 : Fp), (705991 : Fp), (681121 : Fp)]]
def Mf_0_wg : List (List Fp) := [[(173127 : Fp), (780823 : Fp)], [(645278 : Fp), (942672 : Fp)]]
def Mf_0_wu : List (List Fp) := [[(238376 : Fp), (85716 : Fp)], [(272368 : Fp), (50769 : Fp)]]
def Mf_0_wd : List (List Fp) := [[(785762 : Fp), (48084 : Fp)], [(608276 : Fp), (721969 : Fp)]]
def Mf_0_an : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_0_mn : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_0 : Flat := { wq := tab Mf_0_wq, wk := tab Mf_0_wk, wv := tab Mf_0_wv, wo := tab Mf_0_wo, wg := tab Mf_0_wg, wu := tab Mf_0_wu, wd := tab Mf_0_wd, an := tab1 (Mf_0_an.getD 0 []), mn := tab1 (Mf_0_mn.getD 0 []) }
def Mf_0L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 Mf_0
def Mf_1_wq : List (List Fp) := [[(283122 : Fp), (276488 : Fp)], [(378 : Fp), (216179 : Fp)], [(768228 : Fp), (628308 : Fp)], [(843255 : Fp), (918237 : Fp)], [(102362 : Fp), (578998 : Fp)], [(146128 : Fp), (47611 : Fp)], [(3493 : Fp), (764212 : Fp)], [(355312 : Fp), (522743 : Fp)]]
def Mf_1_wk : List (List Fp) := [[(232547 : Fp), (63025 : Fp)], [(618800 : Fp), (145635 : Fp)], [(952415 : Fp), (275292 : Fp)], [(832049 : Fp), (5419 : Fp)]]
def Mf_1_wv : List (List Fp) := [[(606152 : Fp), (26598 : Fp)], [(389938 : Fp), (158233 : Fp)], [(769103 : Fp), (44860 : Fp)], [(407166 : Fp), (830390 : Fp)]]
def Mf_1_wo : List (List Fp) := [[(744539 : Fp), (708184 : Fp), (73039 : Fp), (20598 : Fp), (469677 : Fp), (335124 : Fp), (123502 : Fp), (324654 : Fp)], [(362349 : Fp), (943964 : Fp), (14319 : Fp), (796129 : Fp), (274404 : Fp), (676229 : Fp), (904204 : Fp), (380925 : Fp)]]
def Mf_1_wg : List (List Fp) := [[(283095 : Fp), (26281 : Fp)], [(991942 : Fp), (945153 : Fp)]]
def Mf_1_wu : List (List Fp) := [[(961360 : Fp), (477767 : Fp)], [(159679 : Fp), (573168 : Fp)]]
def Mf_1_wd : List (List Fp) := [[(222269 : Fp), (502587 : Fp)], [(502206 : Fp), (236251 : Fp)]]
def Mf_1_an : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_1_mn : List (List Fp) := [[(1 : Fp), (1 : Fp)]]
def Mf_1 : Flat := { wq := tab Mf_1_wq, wk := tab Mf_1_wk, wv := tab Mf_1_wv, wo := tab Mf_1_wo, wg := tab Mf_1_wg, wu := tab Mf_1_wu, wd := tab Mf_1_wd, an := tab1 (Mf_1_an.getD 0 []), mn := tab1 (Mf_1_mn.getD 0 []) }
def Mf_1L : Layer Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) := toLayer 2 2 1 4 2 2 Mf_1
def Mf : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab Mf_emb) (tab Mf_head) (tab1 Mf_gf) [Mf_0L, Mf_1L]
def M0one : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := mkModel 2 2 1 4 2 2 3 (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L]
def sg1 : Equiv.Perm (Fin 2) := (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2))
def sgn1 : Fin 2 → Fp := fun i => tab1 [(1000002 : Fp), (1000002 : Fp)] i.val
def Mh1 := M0.streamGauge (sp sg1 sgn1) (sp sg1.symm (fun k => sgn1 (sg1.symm k))) (fun γ i => γ (sg1 i))
def tQ : List (List Fp) := [[(301028 : Fp), (736582 : Fp)], [(263421 : Fp), (301028 : Fp)]]
def Qpy : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab tQ i.val j.val
def Mh4 := Mf.streamGauge Qpyᵀ Qpy id
def tA0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(994722 : Fp), (891619 : Fp)], [(315086 : Fp), (653020 : Fp)]] i.val j.val
def tA0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(838277 : Fp), (51993 : Fp)], [(266735 : Fp), (150593 : Fp)]] i.val j.val
def tB0_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(834588 : Fp), (63324 : Fp)], [(793990 : Fp), (576965 : Fp)]] i.val j.val
def tB0_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(592819 : Fp), (309349 : Fp)], [(234493 : Fp), (555300 : Fp)]] i.val j.val
def tA1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(829099 : Fp), (143809 : Fp)], [(468597 : Fp), (11860 : Fp)]] i.val j.val
def tA1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(965199 : Fp), (835298 : Fp)], [(745848 : Fp), (476563 : Fp)]] i.val j.val
def tB1_0 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(990336 : Fp), (295886 : Fp)], [(122843 : Fp), (837787 : Fp)]] i.val j.val
def tB1_1 : Matrix (Fin 2) (Fin 2) Fp := Matrix.of fun i j => tab [[(323393 : Fp), (795071 : Fp)], [(825452 : Fp), (874241 : Fp)]] i.val j.val
def M_ov : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [ovGauge E M0_0L ![tA0_0, tA0_1] ![tB0_0, tB0_1], ovGauge E M0_1L ![tA1_0, tA1_1] ![tB1_0, tB1_1]] }
def M_qk : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [qkGauge E M0_0L (fun g p => tab [[(195864 : Fp)], [(31078 : Fp)]] g.val p.val) (fun g p => tab [[(435146 : Fp)], [(278447 : Fp)]] g.val p.val), qkGauge E M0_1L (fun g p => tab [[(234477 : Fp)], [(481941 : Fp)]] g.val p.val) (fun g p => tab [[(600426 : Fp)], [(543795 : Fp)]] g.val p.val)] }
def M_hp : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [headPerm M0_0L (Equiv.swap (2 : Fin 4) 3 * Equiv.swap (1 : Fin 4) 2 * Equiv.swap (0 : Fin 4) 3 : Equiv.Perm (Fin 4)) (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2)), headPerm M0_1L (1 : Equiv.Perm (Fin 4)) (1 : Equiv.Perm (Fin 2))] }
def M_m1 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [unitPerm (mlpScale M0_0L (fun u => tab1 [(665983 : Fp), (233141 : Fp)] u.val)) (Equiv.swap (0 : Fin 2) 1 : Equiv.Perm (Fin 2)), unitPerm (mlpScale M0_1L (fun u => tab1 [(656489 : Fp), (524696 : Fp)] u.val)) (1 : Equiv.Perm (Fin 2))] }
def M_n1 : Model Fp 2 2 2 (Fin 4) (Fin 2) (Fin 1) (Fin 3) := { M0 with layers := [mlpNorm (attnNorm M0_0L (fun j => tab1 [(911918 : Fp), (769763 : Fp)] j.val)) (fun j => tab1 [(255198 : Fp), (747032 : Fp)] j.val), mlpNorm (attnNorm M0_1L (fun j => tab1 [(812114 : Fp), (187974 : Fp)] j.val)) (fun j => tab1 [(432428 : Fp), (89843 : Fp)] j.val)] }
def main (args : List String) : IO UInt32 := do
  let want : String → Bool := fun s => args.isEmpty || args.contains s
  let t0 ← IO.monoMsNow
  if want "perm" then
    emitNat "sigma" [(List.finRange 2).map fun i => (sg1 i).val]
    emitNat "sigma_heads" [(List.finRange 4).map fun i => (((Equiv.swap (2 : Fin 4) 3 * Equiv.swap (1 : Fin 4) 2 * Equiv.swap (0 : Fin 4) 3 : Equiv.Perm (Fin 4))) i).val, (List.finRange 4).map fun i => (((1 : Equiv.Perm (Fin 4))) i).val]
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
