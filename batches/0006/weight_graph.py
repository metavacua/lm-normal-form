# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# numpy over every two-dimensional parameter of a model: distinct values (and
# the values themselves when there are at most eight), signs, rows and columns
# that are all zero (dead neurons), rows and columns that repeat exactly, rows
# with another row at cosine above 0.99, and the effective rank (entropy of
# the singular values). Usage: weight_graph.py MODEL OUT.tsv
import sys,transformers as t,numpy as np
g=t.AutoModelForCausalLM.from_pretrained(sys.argv[1]).float();f=open(sys.argv[2],'w')
f.write('tensor\trows\tcols\tdistinct\tvalues\tneg\tzero\tpos\tzero_rows\tzero_cols\tdup_rows\tdup_cols\tcos99_rows\teff_rank\n')
for n,p in g.named_parameters():
 if p.ndim==2:
  w=p.detach().numpy();s=np.sign(w);u=np.unique(w)
  r=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-12);c=r@r.T if w.shape[0]<=8192 else np.zeros((1,1));np.fill_diagonal(c,0)
  v=np.linalg.svd(w,compute_uv=False)if min(w.shape)<=8192 else np.ones(1);q=v/v.sum()
  f.write(f"{n}\t{w.shape[0]}\t{w.shape[1]}\t{len(u)}\t{' '.join(f'{x:.6g}'for x in u)if len(u)<=8 else ''}\t{int((s<0).sum())}\t{int((s==0).sum())}\t{int((s>0).sum())}\t{int((np.abs(w).sum(1)==0).sum())}\t{int((np.abs(w).sum(0)==0).sum())}\t{w.shape[0]-np.unique(w,axis=0).shape[0]}\t{w.shape[1]-np.unique(w.T,axis=0).shape[0]}\t{int((c>0.99).any(1).sum())}\t{np.exp(-(q*np.log(q+1e-12)).sum()):.1f}\n");f.flush()
