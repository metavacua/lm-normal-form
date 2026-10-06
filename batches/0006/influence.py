# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# For each layer, over the forward prompts (ids starting "fwd-") and all their
# positions: the mean of one minus the cosine between the layer's input and
# output hidden states. A layer that changes little is a candidate for removal.
# The library applies the final norm to the last state, which inflates the last
# layer. Usage: influence.py MODEL PROMPTS.tsv
import sys,torch,transformers as t,torch.nn.functional as F
torch.set_grad_enabled(False);m=sys.argv[1];k=t.AutoTokenizer.from_pretrained(m);g=t.AutoModelForCausalLM.from_pretrained(m).float().eval()
P=[l.split('\t')[1].rstrip('\n')for l in open(sys.argv[2])if l.startswith('fwd-')];S=0
for p in P:
 h=g(**k(p,return_tensors='pt'),output_hidden_states=True).hidden_states
 S=S+torch.stack([(1-F.cosine_similarity(h[i][0],h[i+1][0],dim=-1)).mean()for i in range(len(h)-1)])
print(len(P),'prompts');print([round(float(x),4)for x in S/len(P)])
