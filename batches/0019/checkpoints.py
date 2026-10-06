# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The architecture of a real checkpoint, read from its published metadata and not from its weights: config.json and the header of its safetensors file (names, shapes and counts of the
# tensors, which Hugging Face's API returns by range requests), reified (ir.py), and put to the Datalog program (states.dl). For each model:
#   the dimension of the gauge group that the program derives from the reified configuration;
#   the conformance of the reified architecture with the file: the names of the parameter tensors that the program has, and their numbers of elements, against the header's.
# Usage: checkpoints.py OUT.json
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
import derive
import ir
import hf_adapter as H

MODELS = {"smol-instruct": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
          "smol-base": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
          "floatlm-99m": ("SpectraSuite/FloatLM_99M", "0516fbe2979fcad7c26805885273562c88c3de90"),
          "trilm-99m": ("SpectraSuite/TriLM_99M_Unpacked", "61cd2c766000fc544595d6570ad1990017d0cb43")}


def main(out):
    from huggingface_hub import get_safetensors_metadata, hf_hub_download
    res = {}
    for key, (repo, rev) in MODELS.items():
        cfg = json.load(open(hf_hub_download(repo, "config.json", revision=rev)))
        a = H.arch_of(cfg)
        eps0 = float(cfg["rms_norm_eps"]) == 0.0
        r, p = derive.derive(a, eps_zero=eps0)
        meta = get_safetensors_metadata(repo, revision=rev)
        header = {n: t.parameter_count for f in meta.files_metadata.values() for n, t in f.tensors.items()}
        mine = ir.param_counts(p)
        mismatched = sorted(n for n in set(header) | set(mine) if header.get(n) != mine.get(n))
        res[key] = {"repo": repo, "revision": rev, "arch": {k: getattr(a, k) for k in ("d", "n_heads", "n_kv", "hd", "d_ff", "n_layers", "vocab", "tied")}, "rms_norm_eps": cfg["rms_norm_eps"],
                    "derived_dimension": r["total"], "groups": {k: v for k, v in sorted({(s.rstrip("0123456789"), g) for s, g in r["groups"].items()})},
                    "library_checks_empty": not (r["no_glb"] or r["many_glb"] or r["ungrouped"] or r["two_groups"]),
                    "tensors_in_header": len(header), "tensors_in_program": len(mine), "parameters_in_header": sum(header.values()), "parameters_in_program": sum(mine.values()), "mismatched": mismatched[:10]}
        print(f"{key:14s} dimension of the gauge group {r['total']:>9,}  parameters: header {sum(header.values()):>12,}, program {sum(mine.values()):>12,}, tensors {len(header)} / {len(mine)}, mismatched {len(mismatched)}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
