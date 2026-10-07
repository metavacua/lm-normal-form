/-
SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: AGPL-3.0-or-later
-/
import LmnfProofs.Norm
import LmnfProofs.Attention
import LmnfProofs.Mlp
import LmnfProofs.Compose

/-! The axioms that each theorem depends on: only `propext`, `Classical.choice` and `Quot.sound` are acceptable (no `sorryAx`). -/

#print axioms Lmnf.preservesDot_of_transpose_mul
#print axioms Lmnf.rms_equivariant
#print axioms Lmnf.rms_scale
#print axioms Lmnf.mulVec_weighted_sum
#print axioms Lmnf.ov_gauge
#print axioms Lmnf.bilinear_gauge
#print axioms Lmnf.cplx_comm
#print axioms Lmnf.cplx_transpose
#print axioms Lmnf.cplx_inv_mul
#print axioms Lmnf.rope_score_gauge
#print axioms Lmnf.rope_gauge_complex
#print axioms Lmnf.head_perm
#print axioms Lmnf.swiglu_scale
#print axioms Lmnf.unit_perm
#print axioms Lmnf.stack_equivariant
#print axioms Lmnf.residual_equivariant
#print axioms Lmnf.model_invariant
