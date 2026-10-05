# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The structure of a published ONNX language model, as RDF, checked in SPARQL.

A cell names an ONNX file in a model repository, the repository that holds the
PyTorch weights it was exported from, and a prompt. The run:

1. fetches the file at the revision published now, and records that revision;
2. renders the graph as RDF and checks the rendering against the file (H1);
3. compares the tensors the graph stores with the safetensors header (H2);
4. runs the prompt through the graph and through the PyTorch weights (H5).

What each hypothesis means, and when it holds, is registered in the batch's
document under docs/batches/. This module only carries it out.
"""

import gc
import hashlib
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

from .. import feeds, graphstore, inventory
from ..check import check

INCONCLUSIVE = 2
NOT_EVALUATED = "NOT EVALUATED"
PROBABILITY_TOLERANCE = 1e-3

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


def reference_forward(onnx_path, repository, revision, prompt):
    """One prompt through the ONNX graph and through the PyTorch weights it was exported from."""
    import numpy as np
    import onnxruntime
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repository, revision=revision)
    default = list(tokenizer(prompt)["input_ids"])
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
    model = AutoModelForCausalLM.from_pretrained(repository, revision=revision).float()
    model.eval()

    result = {"prompt": prompt, "bos_token_id": bos, "encodings": {}}
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


def headline(report):
    """One sentence for the batch summary."""
    subject = report["subject"]
    direct = report.get("direct")
    if not direct or "nodes" not in direct:
        return f"`{subject['repository']}` `{subject['file']}`: the structure check did not complete."
    stored = direct["initializers"]
    return (
        f"`{subject['repository']}` `{subject['file']}` ({subject['bytes']:,} bytes): {direct['nodes']:,} nodes, "
        f"{len(direct['op_histogram'])} kinds of operator, {stored['count']:,} initializers "
        f"({stored['elements']:,} elements), {report['rdf']['stored_quads']:,} quads."
    )


def summary(report):
    """The report as Markdown. Lists, not tables: it is also read in a terminal."""
    subject = report["subject"]
    lines = [
        f"# {report['cell']}",
        "",
        "Structure of a published ONNX file.",
        "",
        f"- File: `{subject['repository']}` at revision `{subject['revision']}`, `{subject['file']}` "
        f"({subject['bytes']:,} bytes, SHA-256 `{subject['sha256']}`)",
        f"- Reference weights: `{subject['reference_repository']}` at revision `{subject['reference_revision']}`",
    ]
    if subject.get("companions"):
        lines.append(f"- Companion files: {', '.join(f'`{name}`' for name in subject['companions'])}")
    lines += ["", "## Verdicts", ""]
    lines += [f"- {name}: {verdict}" for name, verdict in report["verdicts"].items()]
    if report.get("inconclusive"):
        lines += ["", f"**Inconclusive:** {report['inconclusive']}"]

    direct = report.get("direct")
    if direct and "nodes" in direct:
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
            f"{len(parameters['missing_from_onnx'])} missing, {len(parameters['only_in_onnx'])} stored "
            f"floating-point tensors beyond the reference. Elements: reference {parameters['elements_reference']:,}, "
            f"graph {parameters['elements_onnx']:,}.",
        ]
        if parameters["missing_from_onnx"]:
            lines += ["", f"Missing shapes (the first twenty): {parameters['missing_from_onnx'][:20]}"]

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

    exploratory = report.get("exploratory")
    if exploratory:
        lines += [
            "",
            "## Exploratory, no verdict registered",
            "",
            f"- Stored matrices matching the gate-by-wiring query: {exploratory['silu_gate_matches']}",
            "- Reference tensors matched when stored tensors of every element type are counted: "
            f"{exploratory['parameters_matched_counting_every_element_type']}",
        ]
    lines += ["", "Versions: " + ", ".join(f"{name} {version}" for name, version in report["versions"].items()) + "."]
    return "\n".join(lines) + "\n"


def evaluate(cell, subject, path, iri, reference_shapes, forward, out):
    """Everything after the downloads.

    `reference_shapes` is the parameter inventory from an independent source.
    `forward` is called with no arguments and returns the forward-pass
    comparison; it raises `feeds.UnknownInput` when the graph cannot be fed.
    """
    out = Path(out)
    status = check(path, iri, out)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    report.update(cell=cell["id"], experiment="structure", subject=subject, versions=versions())
    verdicts = {H1: report.get("verdicts", {}).get("H1", NOT_EVALUATED), H2: NOT_EVALUATED, H5: NOT_EVALUATED}
    unevaluated = []

    if status == 0:
        store = graphstore.load(out / "model.nq")
        stored = graphstore.stored_tensors(store)
        floats = [shape for kind, shape in stored if kind in graphstore.FLOAT_TYPES]
        comparison = inventory.compare(floats, reference_shapes)
        report["parameters"] = dict(comparison, reference_tensors=len(reference_shapes))
        report["exploratory"] = {
            "silu_gate_matches": len(graphstore.rows(store, "silu_gate")),
            "parameters_matched_counting_every_element_type": inventory.compare(
                [shape for _, shape in stored], reference_shapes
            )["matched"],
        }
        if reference_shapes:
            verdicts[H2] = "HOLDS" if not comparison["missing_from_onnx"] else "FAILS"
        else:
            unevaluated.append("H2: the reference inventory is empty")
        del store
        gc.collect()

        try:
            result = forward()
        except feeds.UnknownInput as error:
            unevaluated.append(f"H5: {error}")
        else:
            report["forward"] = result
            default = result["encodings"]["tokenizer default"]
            agrees = default["same_top5_ids"] and default["max_abs_probability_difference"] <= PROBABILITY_TOLERANCE
            verdicts[H5] = "HOLDS" if agrees else "FAILS"
    else:
        unevaluated.append(report.get("inconclusive", "the structure check did not complete"))

    report["verdicts"] = verdicts
    report["headline"] = headline(report)
    report.pop("inconclusive", None)
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


def run(cell, out):
    """Fetch what the cell names, then evaluate it."""
    onnx_repository = cell["onnx_repository"]
    onnx_file = cell["onnx_file"]
    reference_repository = cell["reference_repository"]
    prompt = cell["prompt"]

    from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    # Models are taken at whatever revision is published now; the revisions are recorded.
    info = api.model_info(onnx_repository)
    revision = info.sha
    reference_revision = revision
    if reference_repository != onnx_repository:
        reference_revision = api.model_info(reference_repository).sha
    # A graph may keep its tensors in companion files next to it; they are part of the model.
    companions = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(onnx_file + "_"))

    with tempfile.TemporaryDirectory(prefix="lmnf-model-") as work:
        for name in companions:
            hf_hub_download(onnx_repository, name, revision=revision, local_dir=work)
        path = Path(hf_hub_download(onnx_repository, onnx_file, revision=revision, local_dir=work))
        subject = {
            "repository": onnx_repository,
            "revision": revision,
            "file": onnx_file,
            "companions": companions,
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
            "reference_repository": reference_repository,
            "reference_revision": reference_revision,
        }
        metadata = get_safetensors_metadata(reference_repository, revision=reference_revision)
        reference_shapes = [tuple(t.shape) for f in metadata.files_metadata.values() for t in f.tensors.values()]
        iri = f"https://huggingface.co/{onnx_repository}/resolve/{revision}/{onnx_file}"
        return evaluate(
            cell,
            subject,
            path,
            iri,
            reference_shapes,
            lambda: reference_forward(path, reference_repository, reference_revision, prompt),
            out,
        )
