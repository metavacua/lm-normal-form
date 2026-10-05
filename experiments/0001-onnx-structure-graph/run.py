#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment 0001: structure of a published ONNX language model, as RDF, checked in SPARQL.

What is registered, and what each verdict means, is in
docs/experiments/0001-onnx-structure-graph.md. This script only carries it out.

Exit status: 0 when every registered hypothesis was evaluated (whatever the
verdicts), 2 when the run is inconclusive. Anything else is a crash.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

REPOSITORY = "HuggingFaceTB/SmolLM2-135M-Instruct"
ONNX_FILE = "onnx/model.onnx"
PROMPT = "The capital of France is"
PROBABILITY_TOLERANCE = 1e-3
INCONCLUSIVE = 2

H1 = "H1 no control flow, no cycle"
H2 = "H2 every parameter is stored in the graph"
H5 = "H5 the graph computes what the PyTorch weights compute"

PACKAGES = ("onnx", "pyoxigraph", "numpy", "huggingface_hub", "onnxruntime", "torch", "transformers", "tokenizers")
ORT_TYPES = {
    "tensor(int64)": "INT64",
    "tensor(int32)": "INT32",
    "tensor(float)": "FLOAT",
    "tensor(float16)": "FLOAT16",
    "tensor(double)": "DOUBLE",
    "tensor(bool)": "BOOL",
}


def versions():
    found = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def top(logits, tokenizer, count=5):
    import numpy as np

    logits = logits.astype(np.float64)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    order = np.argsort(-probabilities)[:count]
    return probabilities, [
        {"id": int(i), "token": tokenizer.decode([int(i)]), "probability": float(probabilities[i])} for i in order
    ]


def reference_forward(onnx_path, revision):
    """The same prompt through the ONNX graph and through the repository's PyTorch weights."""
    import numpy as np
    import onnxruntime
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lmnf import feeds

    tokenizer = AutoTokenizer.from_pretrained(REPOSITORY, revision=revision)
    default = list(tokenizer(PROMPT)["input_ids"])
    encodings = {"tokenizer default": default}
    bos = tokenizer.bos_token_id
    if bos is not None and default[:1] != [bos]:
        encodings["BOS prepended"] = [bos] + default

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    declared = [
        {"name": i.name, "type": ORT_TYPES.get(i.type, i.type), "dims": list(i.shape)} for i in session.get_inputs()
    ]
    outputs = [o.name for o in session.get_outputs()]
    if "logits" not in outputs:
        raise feeds.UnknownInput(f"the graph has no output named logits (outputs: {outputs[:6]})")
    model = AutoModelForCausalLM.from_pretrained(REPOSITORY, revision=revision).float()
    model.eval()

    result = {"prompt": PROMPT, "bos_token_id": bos, "encodings": {}}
    for label, ids in encodings.items():
        from_onnx = session.run(["logits"], feeds.first_step(declared, ids))[0][0, -1]
        with torch.no_grad():
            from_torch = model(torch.tensor([ids])).logits[0, -1].numpy()
        p_onnx, top_onnx = top(from_onnx, tokenizer)
        p_torch, top_torch = top(from_torch, tokenizer)
        result["encodings"][label] = {
            "token_ids": ids,
            "tokens": [tokenizer.decode([i]) for i in ids],
            "onnx_top5": top_onnx,
            "pytorch_top5": top_torch,
            "same_top5_ids": [t["id"] for t in top_onnx] == [t["id"] for t in top_torch],
            "max_abs_logit_difference": float(np.abs(from_onnx.astype(np.float64) - from_torch).max()),
            "max_abs_probability_difference": float(np.abs(p_onnx - p_torch).max()),
        }
    return result


def summary(report):
    """The report as Markdown. Lists, not tables: it is also read in a terminal."""
    subject = report["subject"]
    lines = [
        "# Experiment 0001: ONNX structure graph",
        "",
        f"Subject: `{subject['repository']}` at revision `{subject['revision']}`, file `{subject['file']}` "
        f"({subject['bytes']:,} bytes, SHA-256 `{subject['sha256']}`).",
        "",
        "## Verdicts",
        "",
    ]
    lines += [f"- {name}: {verdict}" for name, verdict in report["verdicts"].items()]
    if report.get("inconclusive"):
        lines += ["", f"**Inconclusive:** {report['inconclusive']}"]

    direct = report.get("direct")
    if direct:
        stored = direct["initializers"]
        lines += [
            "",
            "## Structure",
            "",
            f"- Nodes: {direct['nodes']:,}",
            f"- Control-flow nodes: {len(direct['control_flow'])}",
            f"- Values consumed at or before their producer: {direct['order_violations']}",
            f"- Undefined inputs: {direct['undefined_inputs']}",
            f"- Model-local functions: {direct['functions']}",
            f"- Initializers: {stored['count']:,} ({stored['elements']:,} elements, {stored['bytes']:,} bytes)",
            f"- Initializer element types: {direct['initializer_types']}",
            f"- RDF quads: {report['rdf']['stored_quads']:,}",
            f"- Opsets: {direct['opsets']}; IR version {direct['ir_version']}; producer `{direct['producer']}`",
            "",
            "Operators:",
            "",
        ]
        ordered = sorted(direct["op_histogram"].items(), key=lambda item: (-item[1], item[0]))
        lines += [f"- `{name.removeprefix('::')}`: {count}" for name, count in ordered]
        lines += ["", "Graph inputs and outputs (the first eight of each):", ""]
        for direction, key in (("input", "inputs"), ("output", "outputs")):
            lines += [f"- {direction} `{v['name']}`: {v['type']} {v['dims']}" for v in direct[key][:8]]
            if len(direct[key]) > 8:
                lines.append(f"- and {len(direct[key]) - 8} more {direction}s")

    parameters = report.get("parameters")
    if parameters:
        lines += [
            "",
            "## Parameters",
            "",
            f"{parameters['reference_tensors']} reference tensors, {parameters['matched']} matched in the graph, "
            f"{len(parameters['missing_from_onnx'])} missing, {len(parameters['only_in_onnx'])} stored tensors "
            f"beyond the reference. Elements: reference {parameters['elements_reference']:,}, "
            f"graph {parameters['elements_onnx']:,}.",
        ]
        if parameters["missing_from_onnx"]:
            lines += ["", f"Missing shapes: {parameters['missing_from_onnx'][:20]}"]

    forward = report.get("forward")
    if forward:
        lines += ["", "## Forward pass", "", f"Prompt: `{forward['prompt']}`"]
        for label, encoding in forward["encodings"].items():
            lines += [
                "",
                f"With {label} ({len(encoding['token_ids'])} tokens, ids `{encoding['token_ids']}`): "
                f"same top five {encoding['same_top5_ids']}, largest probability difference "
                f"{encoding['max_abs_probability_difference']:.2e}, largest logit difference "
                f"{encoding['max_abs_logit_difference']:.2e}.",
                "",
            ]
            for rank, (a, b) in enumerate(zip(encoding["onnx_top5"], encoding["pytorch_top5"]), 1):
                lines.append(
                    f"{rank}. ONNX `{a['token']!r}` {a['probability']:.4f}; "
                    f"PyTorch `{b['token']!r}` {b['probability']:.4f}"
                )

    if "exploratory" in report:
        lines += [
            "",
            "## Exploratory, no verdict registered",
            "",
            f"- Stored matrices matching the gate-by-wiring query: {report['exploratory']['silu_gate_matches']}",
        ]
    if report.get("versions"):
        lines += ["", "Versions: " + ", ".join(f"{name} {version}" for name, version in report["versions"].items()) + "."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

    from lmnf import feeds, graphstore, inventory
    from lmnf.__main__ import check

    # The model is taken at whatever revision is published now; that revision is recorded.
    info = HfApi().model_info(REPOSITORY)
    revision = info.sha
    # A graph may keep its tensors in companion files next to it; fetch those too.
    companions = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(ONNX_FILE + "_"))
    for name in companions:
        hf_hub_download(REPOSITORY, name, revision=revision)
    path = Path(hf_hub_download(REPOSITORY, ONNX_FILE, revision=revision))
    subject = {
        "repository": REPOSITORY,
        "revision": revision,
        "file": ONNX_FILE,
        "companions": companions,
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
    }
    iri = f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{ONNX_FILE}"

    status = check(path, iri, out)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    report["subject"] = subject
    report["versions"] = versions()
    verdicts = {H1: report.get("verdicts", {}).get("H1", "NOT EVALUATED"), H2: "NOT EVALUATED", H5: "NOT EVALUATED"}
    unevaluated = []

    if status == 0:
        store = graphstore.load(out / "model.nq")
        metadata = get_safetensors_metadata(REPOSITORY, revision=revision)
        reference = [tuple(t.shape) for f in metadata.files_metadata.values() for t in f.tensors.values()]
        comparison = inventory.compare(graphstore.stored_float_shapes(store), reference)
        report["parameters"] = dict(comparison, reference_tensors=len(reference))
        if reference:
            verdicts[H2] = "HOLDS" if not comparison["missing_from_onnx"] else "FAILS"
        else:
            unevaluated.append("H2: the reference inventory is empty")
        report["exploratory"] = {"silu_gate_matches": len(graphstore.rows(store, "silu_gate"))}
        del store
        gc.collect()

        try:
            forward = reference_forward(path, revision)
            report["forward"] = forward
            default = forward["encodings"]["tokenizer default"]
            agrees = default["same_top5_ids"] and default["max_abs_probability_difference"] <= PROBABILITY_TOLERANCE
            verdicts[H5] = "HOLDS" if agrees else "FAILS"
        except feeds.UnknownInput as error:
            unevaluated.append(f"H5: {error}")
    else:
        unevaluated.append(report.get("inconclusive", "the structure check did not complete"))

    report["verdicts"] = verdicts
    if unevaluated:
        report["inconclusive"] = "; ".join(unevaluated)
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text(summary(report), encoding="utf-8")
    for name, verdict in verdicts.items():
        print(f"{name}: {verdict}")
    if unevaluated:
        print(f"INCONCLUSIVE: {report['inconclusive']}")
        return INCONCLUSIVE
    return 0


if __name__ == "__main__":
    sys.exit(main())
