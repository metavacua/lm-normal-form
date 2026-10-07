/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import LmnfProofs.Model
import LmnfProofs.Glue
import LmnfProofs.GaugeOV
import LmnfProofs.GaugeQK
import LmnfProofs.GaugeHead
import LmnfProofs.GaugeMlp
import LmnfProofs.GaugeNorm
import LmnfProofs.GaugeStream
import LmnfProofs.GaugeStreamPerm
import LmnfProofs.Whole

/-! The axioms that each theorem of batch 0021 depends on: only `propext`, `Classical.choice` and `Quot.sound` are acceptable (no `sorryAx`). -/

#print axioms Lmnf.runLayers_forall₂
#print axioms Lmnf.logits_of_forall₂
#print axioms Lmnf.submatrix_mulVec_row
#print axioms Lmnf.score_of_qk
#print axioms Lmnf.att_congr
#print axioms Lmnf.attnOut_congr
#print axioms Lmnf.attnOut_of_qkv
#print axioms Lmnf.afterAttn_congr
#print axioms Lmnf.layerFn_congr
#print axioms Lmnf.qv_ov
#print axioms Lmnf.kv_ov
#print axioms Lmnf.vv_ov
#print axioms Lmnf.attnOut_ov
#print axioms Lmnf.layerFn_ov
#print axioms Lmnf.cplx_neg_transpose
#print axioms Lmnf.rot_comm
#print axioms Lmnf.cplx_comm_smul
#print axioms Lmnf.kv_qk
#print axioms Lmnf.qv_qk
#print axioms Lmnf.vv_qk
#print axioms Lmnf.score_qk
#print axioms Lmnf.layerFn_qk
#print axioms Lmnf.qv_head
#print axioms Lmnf.kv_head
#print axioms Lmnf.vv_head
#print axioms Lmnf.score_head
#print axioms Lmnf.att_head
#print axioms Lmnf.attnOut_headPerm
#print axioms Lmnf.layerFn_headPerm
#print axioms Lmnf.mlpOut_scale
#print axioms Lmnf.layerFn_mlpScale
#print axioms Lmnf.mlpOut_unitPerm
#print axioms Lmnf.layerFn_unitPerm
#print axioms Lmnf.reader_scale
#print axioms Lmnf.nrm_scale
#print axioms Lmnf.qv_attnNorm
#print axioms Lmnf.kv_attnNorm
#print axioms Lmnf.vv_attnNorm
#print axioms Lmnf.layerFn_attnNorm
#print axioms Lmnf.mlpOut_mlpNorm
#print axioms Lmnf.layerFn_mlpNorm
#print axioms Lmnf.nrm_stream
#print axioms Lmnf.reader_stream
#print axioms Lmnf.qv_stream
#print axioms Lmnf.kv_stream
#print axioms Lmnf.vv_stream
#print axioms Lmnf.attnOut_stream
#print axioms Lmnf.afterAttn_stream
#print axioms Lmnf.mlpOut_stream
#print axioms Lmnf.layerFn_stream
#print axioms Lmnf.runLayers_stream
#print axioms Lmnf.logits_stream
#print axioms Lmnf.sp_mulVec
#print axioms Lmnf.sp_preservesDot
#print axioms Lmnf.sp_leftInv
#print axioms Lmnf.sp_compat
#print axioms Lmnf.logits_signedPerm
#print axioms Lmnf.compat_const
#print axioms Lmnf.orth_leftInv
#print axioms Lmnf.logits_orthogonal
#print axioms Lmnf.layerFn_gauged
#print axioms Lmnf.logits_gauged
#print axioms Lmnf.table_stream
#print axioms Lmnf.table_ov
#print axioms Lmnf.table_qk
#print axioms Lmnf.table_head
#print axioms Lmnf.table_mlpScale
#print axioms Lmnf.table_unitPerm
#print axioms Lmnf.table_attnNorm
#print axioms Lmnf.table_mlpNorm
#print axioms Lmnf.both_moved
