# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Read a model on every prompt after removing the decoder layers named (by
# index) in the third argument, or none; then save the reduced model.
# Usage: remove_layers.py MODEL PROMPTS.tsv "12,13" OUT_DIR
import sys,json,torch,transformers as t
torch.set_grad_enabled(False);m,prompts,drop,out=sys.argv[1:5];k=t.AutoTokenizer.from_pretrained(m);g=t.AutoModelForCausalLM.from_pretrained(m).float().eval()
D={int(x)for x in drop.split(',')if x};L=g.model.layers;keep=[l for i,l in enumerate(L)if i not in D];g.model.layers=torch.nn.ModuleList(keep)
for i,l in enumerate(g.model.layers):
 if hasattr(l,'self_attn')and hasattr(l.self_attn,'layer_idx'):l.self_attn.layer_idx=i
g.config.num_hidden_layers=len(keep);print('layers kept',len(keep),'removed',sorted(D),'parameters',sum(p.numel()for p in g.parameters()))
with open(out+'/answers.jsonl','w')as a:
 for l in open(prompts):
  i,p=l.rstrip('\n').split('\t');x=k(p,return_tensors='pt');v,j=g(**x,use_cache=False).logits[0,-1].softmax(-1).topk(20);a.write(json.dumps({'id':i,'probs':[{'token':k.decode([int(b)]),'logprob':float(q.log())}for b,q in zip(j,v)]})+'\n')
g.save_pretrained(out+'/model');k.save_pretrained(out+'/model')
