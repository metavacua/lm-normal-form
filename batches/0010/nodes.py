# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# An ONNX graph as text: one row per node (index, name, domain, operator, inputs,
# outputs, attribute names) in NODES.tsv, and on stdout the opsets, the graph's inputs and
# outputs with their dimensions, the initializers' count and elements, and the operators
# by count. Weights are not read. Usage: nodes.py MODEL NODES.tsv
import sys,math,onnx,collections as c
m=onnx.load(sys.argv[1],load_external_data=False);g=m.graph
open(sys.argv[2],'w').write(''.join('\t'.join(map(str,[i,n.name,n.domain,n.op_type,','.join(n.input),','.join(n.output),','.join(a.name for a in n.attribute)]))+'\n' for i,n in enumerate(g.node)))
d=lambda v:[x.dim_param or x.dim_value for x in v.type.tensor_type.shape.dim]
print('ir',m.ir_version,'opsets',[(o.domain,o.version)for o in m.opset_import],'producer',m.producer_name,m.producer_version)
print('inputs',[(v.name,d(v))for v in g.input],'outputs',[(v.name,d(v))for v in g.output])
print(len(g.node),'nodes',len(g.initializer),'initializers',sum(math.prod(t.dims)for t in g.initializer),'elements')
print('operators',c.Counter((n.domain,n.op_type)for n in g.node).most_common())
