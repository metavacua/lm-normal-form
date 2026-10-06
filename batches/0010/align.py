# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Aligns an ONNX node table (nodes.py's NODES.tsv) with the Rust burn-onnx generated from the
# graph. The rule is onnx-ir's (phases/node_conversion.rs): nodes are visited in file order and
# named <NodeType lowercase><ordinal among nodes of that type>; a MatMul whose second input is an
# initializer becomes a Linear and is named linear<ordinal among Linears>; outputs are <name>_out<N>.
# Prints, for every node whose name matches SCOPE (a regular expression), the node's index,
# operator, generated name and generated statement, or that it has none (removed or folded).
# Usage: align.py NODES.tsv model.rs SCOPE
import sys,re,collections as c
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
made={o for r in rows for o in r[5].split(',')}
src=open(sys.argv[2]).read();stm={}
for m in re.finditer(r'^\s*let (\w+)(?::[^=]+)? = ',src,re.M):
 j=src.find(';\n',m.start());stm[m.group(1)]=' '.join(src[m.start():j+1].split())
n=c.Counter();k=0;scope=re.compile(sys.argv[3])
for r in rows:
 i,name,dom,op,ins=r[:5];n[op]+=1;b=f'{op.lower()}{n[op]}'
 if op=='MatMul' and ins.split(',')[1] not in made|{'input_ids','attention_mask','position_ids'}:k+=1;b=f'linear{k}'
 if scope.search(name):print(f'{int(i):5d} {op:16s} {b:14s}',stm.get(f'{b}_out1','(no statement: removed or folded)'))
