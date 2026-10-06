# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# ONNX Runtime on a graph whose inputs are input_ids, attention_mask and position_ids.
# LEVEL is its graph optimization: all (the default, which fuses operators) or none
# (every node run as the graph states it). For each argument IDS (token ids, comma
# separated) the logits of every position as float32 in OUTDIR/N.f32 (N counts from 0),
# and the five most probable next tokens.
# Usage: logits.py MODEL OUTDIR LEVEL IDS [IDS ...]
import sys,numpy as np,onnxruntime as o
q=o.SessionOptions();q.graph_optimization_level=o.GraphOptimizationLevel.ORT_DISABLE_ALL if sys.argv[3]=='none' else o.GraphOptimizationLevel.ORT_ENABLE_ALL
s=o.InferenceSession(sys.argv[1],q,providers=['CPUExecutionProvider']);n={i.name for i in s.get_inputs()}
for k,a in enumerate(sys.argv[4:]):
 i=np.array([[int(x)for x in a.split(',')]]);f={'input_ids':i,'attention_mask':np.ones_like(i),'position_ids':np.arange(i.shape[1])[None]}
 L=s.run(None,{u:v for u,v in f.items()if u in n})[0][0].astype('float32');L.tofile(f'{sys.argv[2]}/{k}.f32')
 p=np.exp(L[-1].astype('float64')-L[-1].max());p/=p.sum();j=np.argsort(-p)[:5];print(k,L.shape,[(int(x),round(float(p[x]),4))for x in j])
