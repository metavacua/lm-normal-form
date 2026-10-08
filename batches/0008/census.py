# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Exact structure that could be contracted without changing the function, and
# whether the architecture can express the contraction. Per model: for every
# layer, dead query/key dimensions (rows of q_proj or k_proj that are all
# zero) counted per head, and the same by rotary plane (a plane is the pair of
# dimensions j and j+d/2 that rotary embedding rotates together; it adds
# nothing to an attention score when the query pair or the key pair is zero in
# both components; a head with every plane dead has all-zero scores, which a
# head with every dimension dead in the per-dimension count need not have);
# dead value dimensions (zero rows of v_proj, or zero columns of o_proj) per head; dead FFN neurons (zero
# rows of up_proj or gate_proj, or zero columns of down_proj); exactly
# duplicate FFN neurons (equal sign patterns of the rows of up_proj and gate_proj
# together). Also the
# distinct values of each two-dimensional tensor when there are at most eight.
# Usage: census.py MODEL
import sys,transformers as t,numpy as np
g=t.AutoModelForCausalLM.from_pretrained(sys.argv[1]).float();c=g.config;H=c.num_attention_heads;K=c.num_key_value_heads;d=c.hidden_size//H
W=lambda m:m.weight.detach().numpy();zr=lambda w:(np.abs(w).sum(1)==0);zc=lambda w:(np.abs(w).sum(0)==0)
tot={'dead_qk_dims':0,'dead_qk_heads':0,'dead_qk_planes':0,'dead_qk_plane_heads':0,'dead_v_dims':0,'dead_v_heads':0,'dead_ffn':0,'dup_ffn':0}
for i,l in enumerate(g.model.layers):
 a=l.self_attn;q=zr(W(a.q_proj)).reshape(H,d);kk=zr(W(a.k_proj)).reshape(K,d);v=zr(W(a.v_proj)).reshape(K,d);o=zc(W(a.o_proj)).reshape(H,d)
 qk=q|np.repeat(kk,H//K,0);h=d//2;kr=np.repeat(kk,H//K,0);pl=(q[:,:h]&q[:,h:])|(kr[:,:h]&kr[:,h:]);vo=np.repeat(v,H//K,0)|o
 up=W(l.mlp.up_proj);ga=W(l.mlp.gate_proj);dn=W(l.mlp.down_proj);ffn=zr(up)|zr(ga)|zc(dn)
 s=np.concatenate([np.sign(up),np.sign(ga)],1);dup=up.shape[0]-np.unique(s,axis=0).shape[0]
 r={'dead_qk_dims':int(qk.sum()),'dead_qk_heads':int(qk.all(1).sum()),'dead_qk_planes':int(pl.sum()),'dead_qk_plane_heads':int(pl.all(1).sum()),'dead_v_dims':int(vo.sum()),'dead_v_heads':int(vo.all(1).sum()),'dead_ffn':int(ffn.sum()),'dup_ffn':int(dup)}
 for k2 in tot:tot[k2]+=r[k2]
 if any(r.values()):print('layer',i,r,'dead qk per head',qk.sum(1).tolist())
print('total',tot,'heads',H,'kv heads',K,'head dim',d)
for n,p in g.named_parameters():
 if p.ndim==2:
  u=np.unique(p.detach().numpy())
  if len(u)<=8:print(n,len(u),[f'{x:.6g}'for x in u])
