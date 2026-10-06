# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# torch.export of a model on one prompt: the graph's nodes by kind and by
# operator, its inputs and outputs by kind, the module paths, and the fan-out
# of nodes. Usage: export_graph.py MODEL PROMPT
import sys,torch,transformers as t,collections as c
torch.set_grad_enabled(False);m,p=sys.argv[1:3];g=t.AutoModelForCausalLM.from_pretrained(m,use_cache=False).eval();i=t.AutoTokenizer.from_pretrained(m)(p,return_tensors='pt')
e=torch.export.export(g,(),dict(i));s=e.graph_signature;G=e.graph
print(m,len(G.nodes),'nodes',c.Counter(x.kind.name for x in s.input_specs),c.Counter(x.kind.name for x in s.output_specs),len(e.module_call_graph),'module paths')
print('operators',c.Counter(n.target.name()if hasattr(n.target,'name')else n.op for n in G.nodes).most_common(40))
print('fan-out',sorted(c.Counter(len(n.users)for n in G.nodes).items()))
