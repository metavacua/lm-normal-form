# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The whole of batch 0028's cells 02 to 06 on two small synthetic models, with the layout of artifacts that the workflow writes:
safetensors, ONNX (the linear weights stored transposed, the tied head, an anonymous initializer name), GGUF (the query and key
weights permuted as llama.cpp does), a burnpack of each strategy (the weights as [in, out], constants, random param_ids), generated
Rust, a WebAssembly module, and logits. Then combine.py (with Oxigraph) and grade28.py on the result.

The models are not SmolLM2, so the counts that grade28.py registers for SmolLM2 are not the counts here; this test checks that every
program runs on the output of the one before it, that the maps have the counts that the construction gives, and that the ground
check passes and fails where it should.

Usage: mini_pipeline.py [OXIGRAPH [WASM_TOOLS]]
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

import combine
import common
import gguf_tools as GT
import grade28
import ground as G
import onnxgraph
import wasm_tools as W
from common import identity

HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY_CLI = os.path.join(HERE, "..", "0011", "identity.py")
WAT = """(module
  (memory (export "memory") 1)
  (data (i32.const 0) "ab\\00\\00")
  (func $leaf (param i32) (result i32) local.get 0)
  (func $rec (export "forward") (param i32) (result i32)
    nop
    block
      local.get 0
      br_if 0
    end
    loop
      local.get 0
      br_if 0
    end
    local.get 0
    call $rec
    drop
    local.get 0
    call $leaf)
  (func (param i32) (result i32) local.get 0 call $leaf))"""
RUST = '''// Generated from ONNX "work/graph/%(tag)s/model.onnx" by burn-onnx
#[derive(Module, Debug)]
pub struct Submodule1 {
    constant1: burn::module::Param<Tensor<2>>,
    linear1: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    pub fn new(device: &Device) -> Self {
        let constant1: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
        );
        Self { constant1 }
    }
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let gather1_out1 = x.clone();
        let linear1_out1 = self.linear1.forward(gather1_out1);
        linear1_out1
    }
}
#[derive(Module, Debug)]
pub struct Model {
    submodule1: Submodule1,
}
%(tail)s
'''
FILE_TAIL = '''impl Default for Model {
    fn default() -> Self {
        Self::from_file("work/gen/%(tag)s/model.bpk", &Default::default())
    }
}'''


def make_onnx(st, path):
    from onnx import TensorProto, helper, numpy_helper
    import onnx
    inits, nodes, cur, k = [], [], "h0", 0
    emb = st["model.embed_tokens.weight"].astype(np.float32)
    inits.append(numpy_helper.from_array(emb, "model.embed_tokens.weight"))
    nodes.append(helper.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["h0"], name="/model/embed_tokens/Gather"))
    for name in sorted(st):
        if name == "model.embed_tokens.weight":
            continue
        a = st[name].astype(np.float32)
        k += 1
        if a.ndim == 1:
            inits.append(numpy_helper.from_array(a, name))
            nodes.append(helper.make_node("Mul", [cur, name], [f"h{k}"], name="/" + name.replace(".", "/") + "/Mul"))
        else:
            w = f"onnx::MatMul_{9000 + k}"
            inits.append(numpy_helper.from_array(np.ascontiguousarray(a.T), w))
            nodes.append(helper.make_node("MatMul", [cur, w], [f"h{k}"], name="/" + name[:-len(".weight")].replace(".", "/") + "/MatMul"))
        cur = f"h{k}"
    inits.append(numpy_helper.from_array(np.ascontiguousarray(emb.T), "onnx::MatMul_9999"))   # the tied head, transposed
    nodes.append(helper.make_node("Constant", [], ["eps"], name="/eps/Constant",
                                  value=numpy_helper.from_array(np.array(1e-5, dtype=np.float32))))
    nodes.append(helper.make_node("Constant", [], ["neg1"], name="/neg1/Constant",
                                  value=numpy_helper.from_array(np.array([-1], dtype=np.int64))))
    nodes.append(helper.make_node("MatMul", [cur, "onnx::MatMul_9999"], ["logits"], name="/lm_head/MatMul"))
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 3])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3, 20])], inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 14)]), path)


def make_bpk(st, path, seed):
    """A burnpack: the linear weights as [in, out] under generated names, the embedding as a constant, constants, random ids."""
    import cbor2
    rng = np.random.default_rng(seed)
    tensors, order = {}, []
    k = 0
    for name in sorted(st):
        a = st[name].astype(np.float32)
        if name == "model.embed_tokens.weight":
            order.append(("submodule1.constant1", a))
        elif a.ndim == 1:
            k += 1
            order.append((f"submodule1.constant{k + 1}", a))
        else:
            k += 1
            order.append((f"submodule1.linear{k}.weight", np.ascontiguousarray(a.T)))
    order.append(("submodule1.linear999.weight", np.ascontiguousarray(st["model.embed_tokens.weight"].astype(np.float32).T)))
    order.append(("submodule2.constant1", np.array([1e-5], dtype=np.float32)))
    order.append(("submodule2.constant2", np.array([-1], dtype=np.int64)))
    off, chunks = 0, []
    for name, a in order:
        raw = a.tobytes()
        pad = -off % 256
        chunks.append(b"\0" * pad + raw)
        off += pad
        tensors[name] = {"data_offsets": [off, off + len(raw)], "dtype": "F32" if a.dtype == np.float32 else "I64",
                         "param_id": int(rng.integers(1, 2 ** 62)), "shape": list(a.shape)}
        off += len(raw)
    meta = cbor2.dumps({"tensors": tensors, "metadata": {"producer": "burn-onnx"}})
    head = struct.pack("<IHI", 0x4255524E, 1, len(meta)) + meta
    head += b"\0" * (-len(head) % 256)
    with open(path, "wb") as f:
        f.write(head + b"".join(chunks))


def build(root, ox, tool):
    art, work = os.path.join(root, "artifacts"), os.path.join(root, "work")
    models = {"base": GT.synthetic_model(11), "instruct": GT.synthetic_model(22)}
    rng = np.random.default_rng(5)
    for tag, (st, cfg) in models.items():
        d = os.path.join(work, tag)
        os.makedirs(d)
        GT.write_safetensors(d + "/model.safetensors", st)
        json.dump(cfg, open(d + "/config.json", "w"))
        make_onnx(st, d + "/model.onnx")
        extra = {"general.name": tag, "tokenizer.chat_template": "x"} if tag == "instruct" else {"general.name": tag}
        GT.write_gguf(d + "/f32.gguf", st, cfg, "f32", extra_kv=extra)
        GT.write_gguf(d + "/bf16.gguf", st, cfg, "bf16", extra_kv=extra)
        make_bpk(st, d + "/file.bpk", seed=1)
        make_bpk(st, d + "/bytes.bpk", seed=2)
        for label in ("file", "bytes"):
            tail = FILE_TAIL % {"tag": tag} if label == "file" else ""
            open(f"{d}/{label}.rs", "w").write(RUST % {"tag": tag, "tail": tail})
        # inputs-TAG
        o = os.path.join(art, f"inputs-{tag}", tag)
        os.makedirs(o)
        run = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True)  # noqa: E731
        run(IDENTITY_CLI, "st", d + "/model.safetensors", o + "/st-record.json", o + "/st-census.json")
        run(IDENTITY_CLI, "onnx", d + "/model.onnx", o + "/onnx-record.json", o + "/onnx-census0011.json")
        onnxgraph.run(d + "/model.onnx", tag, "onnx", o)
        open(o + "/vocab.txt", "w").write("20\n")
        open(o + "/revision.txt", "w").write("abc\n")
        open(o + "/model.sha256", "w").write(identity.sha256_file(d + "/model.onnx") + "  work/onnx/model.onnx\n")
        json.dump({"export": {"outcome": "success"}}, open(o + "/outcomes.json", "w"))
        # gguf-TAG
        g = os.path.join(art, f"gguf-{tag}", tag)
        GT.analyze(tag, d + "/f32.gguf", d + "/bf16.gguf", d + "/model.safetensors", d + "/config.json", g)
        # burn-TAG
        b = os.path.join(art, f"burn-{tag}", tag)
        os.makedirs(b)
        import burn_tools
        burn_tools.run(tag, d + "/model.onnx", d + "/file.bpk", d + "/file.rs", d + "/bytes.bpk", d + "/bytes.rs", b)
        shutil.copy(d + "/file.rs", b + "/model.rs")
        shutil.copy(d + "/bytes.rs", b + "/bytes-model.rs")
        open(d + "/m.wat", "w").write(WAT)
        subprocess.run([tool, "parse", d + "/m.wat", "-o", b + "/patched.wasm"], check=True)
        W.run(tag, "wasm", b + "/patched.wasm", b, tool)
        W.search(b + "/patched.wasm", d + "/bytes.bpk", b + "/weights-search.json")
        open(b + "/host.log", "w").write("imports: []\n")
        json.dump({"build": {"outcome": "success"}}, open(b + "/outcomes.json", "w"))
    # logits: the same prompts for both; the module agrees with ONNX Runtime within 1e-5, the models differ
    for tag in models:
        for sub, noise in (("ort-all", 0.0), ("ort-none", 1e-6)):
            os.makedirs(os.path.join(art, f"inputs-{tag}", tag, sub))
        os.makedirs(os.path.join(art, f"burn-{tag}", tag, "wasm-logits"))
    for n, pos in enumerate((2, 1, 3)):
        for tag in ("base", "instruct"):
            x = rng.standard_normal((pos, 20)).astype("float32")
            x.tofile(os.path.join(art, f"inputs-{tag}", tag, "ort-all", f"{n}.f32"))
            (x + 1e-6).astype("float32").tofile(os.path.join(art, f"inputs-{tag}", tag, "ort-none", f"{n}.f32"))
            (x + 1e-5).astype("float32").tofile(os.path.join(art, f"burn-{tag}", tag, "wasm-logits", f"{n}.f32"))
    # the closed module
    c = os.path.join(art, "wasm27-closed", "closed")
    os.makedirs(c)
    shutil.copy(os.path.join(work, "base", "m.wat"), os.path.join(root, "c.wat"))
    subprocess.run([tool, "parse", os.path.join(root, "c.wat"), "-o", os.path.join(root, "c.wasm")], check=True)
    W.run("closed", "closed", os.path.join(root, "c.wasm"), c, tool)
    # weights.json beside the results
    import weights_compare
    out = os.path.join(root, "results")
    os.makedirs(out)
    weights_compare.run(work + "/base/model.safetensors", work + "/instruct/model.safetensors", out + "/weights.json")
    return art, out


def main(ox, tool):
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    with tempfile.TemporaryDirectory() as root:
        art, out = build(root, ox, tool)
        obs = combine.run(art, out, ["base", "instruct"], ["base", "instruct"], ox)
        check("observations.json, the tables and the dataset were written", all(os.path.exists(os.path.join(out, f)) for f in (
            "observations.json", "census-table.md", "tensor-map-base.tsv", "tensor-map-instruct.tsv", "dataset.nq.gz")))
        b = obs["base"]
        n_lin = 2 * 7 + 0
        check("ST to ONNX: the matrices are transposed, the vectors and the table identical, the embedding twice",
              b["st_onnx"]["relations"] == {"transposed": n_lin + 1, "identical": 1 + 2 * 2 + 1}
              and b["st_onnx"]["ref_tensors_matched_by_more_than_one"] == {"model.embed_tokens.weight": 2})
        check("ST to GGUF: the query and key weights are permuted, the rest identical",
              b["gguf"]["relations"] == {"identical": 20 - 4, "permuted": 4})
        check("ONNX to burnpack: every tensor has a counterpart, the scalar is reshaped",
              "none" not in b["onnx_bpk"]["large"] and "none" not in b["onnx_bpk"]["small"] and b["onnx_bpk"]["small"].get("reshaped", 0) >= 1)
        check("burnpack File against Bytes: every param_id differs, the data is equal", b["bpk_file_vs_bytes"]["data_region_sha256_equal"]
              and b["bpk_file_vs_bytes"]["param_ids_differing"] == b["bpk_census"]["file"]["tensors"])
        c = obs["cross"]
        check("base against Instruct: no weight key is shared; the burnpack constants are", all(
            c["weights_shared_keys"][k] == 0 for k in ("st", "onnx", "gguf-f32", "bpk")) and c["bpk_small"]["digests_equal"])
        check("the programs are one program: skeleton, structure triples, generated Rust apart from the path, module bytes",
              c["skeleton"]["equal"] and c["structure_triples"]["equal"] and c["rust"]["bytes-model.rs"]["differing_lines"] == 1
              and c["rust"]["model.rs"]["differing_lines"] == 2 and c["wasm"]["equal"])
        check("the key-value tables differ in general.name and the chat template only", set(c["kv"]["differ"] + c["kv"]["only_instruct"])
              == {"general.name", "tokenizer.chat_template"})
        g = obs["ground"]
        ox_res = g.get("oxigraph", {})
        check("ground: no blank node by Python or by Oxigraph, and the control is caught by both",
              g["python_blank_statements"] == 0 and ox_res.get("ground-terms") == "false" and ox_res.get("ground-graph-names") == "false"
              and ox_res.get("control_ground_terms") == "true" and g["control_python_blank_statements"] == 1)
        check("ground: every graph is written the same twice and nothing collapsed", all(ok for _, ok in g["determinism"])
              and g["collapsed_triples"] == 0)
        q = obs["queries"]
        check("SPARQL: ST tensors have counterparts, one to one up to the tied embedding",
              q["st_counterparts"]["base/gguf-f32"] == {"1": 20} and q["st_counterparts"]["base/onnx"] == {"1": 19, "2": 1})
        check("SPARQL: across the models only the burnpack constants are related",
              set(q["cross_shared"]) == {"bpk"} and q["cross_shared"]["bpk"] == 2)
        check("logits: the module agrees with ONNX Runtime, and not with the other model", b["logits"]["ort_all"]["P7"] and b["logits"][
            "ort_all"]["P8"] and b["logits"]["control_other_model"]["max_abs"] > 0.1)
        check("no query failed", not q["errors"])
        rows, ctl = grade28.grade(obs)
        by = {r["id"]: r["status"] for r in rows}
        print("grade28 on the synthetic models (the registered SmolLM2 counts do not apply):", by)
        free = ("P9", "P10", "P11", "P12", "P14", "P15", "P16", "P18", "P20")
        check("the checks that do not depend on the size of SmolLM2 are supported", all(by[p] == "supported" for p in free))
        print({r["id"]: r["detail"] for r in rows if r["id"] in free and r["status"] != "supported"})
        check("the controls hold", all(v is not False for v in ctl.values()))
        # a run of one model, as the push of the workflow makes with `tiny`: only what applies to it is graded
        one = os.path.join(root, "results-one")
        os.makedirs(one)
        obs1 = combine.run(art, one, ["base"], ["base"], ox)
        rows1, ctl1 = grade28.grade(obs1)
        by1 = {r["id"]: r["status"] for r in rows1}
        check("a run of one model grades P15 to P18 and leaves the predictions about two models not applicable",
              all(by1[p] == "supported" for p in ("P15", "P16", "P18")) and by1["P17"] in ("supported", "refuted")
              and by1["P7"] == "not applicable" and by1["P1"] == "not applicable")
        if os.environ.get("MINI_KEEP"):
            shutil.copytree(out, os.environ["MINI_KEEP"], dirs_exist_ok=True)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else shutil.which("oxigraph"), sys.argv[2] if len(sys.argv) > 2 else "wasm-tools") else 0)
