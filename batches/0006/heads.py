# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Per layer: the attention heads' query and output slices, cosine between
# heads, and how many heads have another head above 0.9; key and value heads
# likewise. Usage: heads.py MODEL
import sys,torch,transformers as t,numpy as np
g=t.AutoModelForCausalLM.from_pretrained(sys.argv[1]).float();c=g.config;H=c.num_attention_heads;K=c.num_key_value_heads;d=c.hidden_size//H
def dup(M):
 r=M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-12);s=r@r.T;np.fill_diagonal(s,0);return int((s>0.9).any(1).sum()),round(float(s.max()),3)
for i,l in enumerate(g.model.layers):
 a=l.self_attn;W=lambda m:m.weight.detach().numpy()
 q=W(a.q_proj).reshape(H,-1);o=W(a.o_proj).T.reshape(H,-1);k=W(a.k_proj).reshape(K,-1);v=W(a.v_proj).reshape(K,-1)
 print(i,'q',dup(q),'o',dup(o),'qo',dup(np.concatenate([q,o],1)),'k',dup(k),'v',dup(v))
