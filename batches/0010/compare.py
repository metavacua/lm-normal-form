# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Two programs' logits for the same prompts, as written by logits.py and by the Rust
# runner: float32 files N.f32 of shape [positions, VOCAB]. Per prompt: the largest and
# mean absolute difference of logits over all positions, how many positions have the same
# argmax, whether the five most probable next tokens are the same ids in the same order,
# and the largest difference of their probabilities. A difference is a result, not a
# failure. Usage: compare.py DIR_A DIR_B VOCAB
import sys,glob,os,numpy as np
A,B,V=sys.argv[1],sys.argv[2],int(sys.argv[3])
P=lambda l:(lambda e:e/e.sum())(np.exp(l.astype('float64')-l.max()))
for f in sorted(glob.glob(A+'/*.f32')):
 g=os.path.join(B,os.path.basename(f))
 if not os.path.exists(g):print(os.path.basename(f),'missing in',B);continue
 a,b=[np.fromfile(x,'float32').reshape(-1,V)for x in(f,g)]
 if a.shape!=b.shape:print(os.path.basename(f),'shapes',a.shape,b.shape);continue
 pa,pb=P(a[-1]),P(b[-1]);ja,jb=np.argsort(-pa)[:5],np.argsort(-pb)[:5];d=np.abs(a-b)
 print(os.path.basename(f),a.shape,'max',float(d.max()),'mean',float(d.mean()),'argmax equal',int((a.argmax(1)==b.argmax(1)).sum()),'of',len(a),'top5 same order',bool((ja==jb).all()),'max dp',float(np.abs(pa[ja]-pb[ja]).max()))
