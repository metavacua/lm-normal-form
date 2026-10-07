# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Markdown tables for docs/batches/0023.md from the raw result files of a run (RESULTS dir with the json files of the jobs).
# Usage: tables23.py RESULTS > tables.md
import json, os, sys

R = sys.argv[1]


def load(name):
    p = os.path.join(R, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def g(x, f="{:.2e}"):
    return "n/a" if x is None else f.format(x)


def lines_for(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


out = []
# ------------------------------------------------------------------ the test "is a language model"
rows = []
for key, label in [("systems-smol-instruct", "SmolLM2-135M-Instruct"), ("systems-smol-base", "SmolLM2-135M"), ("systems-floatlm-99m", "FloatLM 99M"), ("systems-trilm-99m", "TriLM 99M (ternary)"),
                   ("systems-delphi-100k", "delphi-suite/v0-llama2-100k"), ("outside-qwen3-0.6b", "Qwen3-0.6B"), ("outside-gpt2", "GPT-2 (124M)"), ("outside-pythia-160m", "Pythia-160M")]:
    d = load(key)
    if d is None or not d.get("membership"):
        rows.append([label, "not run", "", "", ""])
        continue
    m = d["membership"]
    rows.append([label, m["tokens"], f"{m['mean_loss_nats']:.3f}", f"{m['ln_output_vocabulary']:.3f}", f"{m['fraction_of_ln_vocabulary']:.3f}"])
b = load("bridge-torch")
if b:
    m = b["membership_control"]
    rows.append(["the finite instance (control; uniformly random sequences of its 5 symbols)", "", f"{m['mean_loss_nats']:.3f}", f"{m['ln_output_vocabulary']:.3f}", f"{m['fraction_of_ln_vocabulary']:.3f}"])
out += lines_for("The registered test of 'is a language model' (mean loss on `membership.txt` over ln of the vocabulary; at most 0.9 passes)", ["system", "tokens", "mean loss (nats)", "ln V", "fraction"], rows)

# ------------------------------------------------------------------ kinematics, Llama family
rows = []
for key, label in [("systems-smol-instruct", "SmolLM2-135M-Instruct"), ("systems-smol-base", "SmolLM2-135M"), ("systems-floatlm-99m", "FloatLM 99M"), ("systems-trilm-99m", "TriLM 99M"), ("systems-delphi-100k", "delphi 100k")]:
    d = load(key)
    if d is None or "translation" not in d:
        rows.append([label] + ["not run"] * 7)
        continue
    tr = max(v["largest"] for v in d["translation"].values())
    dil = min(v["smallest"] for v in d["dilation"].values())
    ph = d["phase"]
    rows.append([label, g(tr), g(ph["logits"]["largest"]), g(ph["keys_of_the_gauged_model_against_the_original_at_the_shifted_positions"]["largest"]), g(dil), g(d["rotation"]["logits"]["largest"]),
                 g(d["rotation_without_the_fold_control"]["smallest"]), g(ph["keys_moved_by_the_shift_control"]["smallest_over_sentences_of_the_mean_over_layers"], "{:.3f}")])
out += lines_for("Kinematics of the Llama family, float64 (relative differences of the scores; ≤ 1e-6 is exact in the sense of the claims)", ["system", "translation of the positions, largest", "phase element on the weights, largest", "its keys against the original's at the shifted positions, largest", "dilation of the angles, smallest", "rotation (fold, untie), largest", "rotation without the fold (control), smallest", "shift moves the keys (control), smallest"], rows)

# ------------------------------------------------------------------ outside the proved class
q = load("outside-qwen3-0.6b")
if q:
    t = q["qk_norm_transformations"]
    rows = [[k, g(v["smallest"]), g(v["largest"])] for k, v in t.items()]
    rows += [["translation of the positions (1, 16, 128)", g(min(v["smallest"] for v in q["translation"].values())), g(max(v["largest"] for v in q["translation"].values()))],
             ["dilation (0.5, 2)", g(min(v["smallest"] for v in q["dilation"].values())), g(max(v["largest"] for v in q["dilation"].values()))],
             ["phase element on the weights (shift of 16)", g(q["phase"]["logits"]["smallest"]), g(q["phase"]["logits"]["largest"])],
             ["rotation with the fold", g(q["rotation"]["logits"]["smallest"]), g(q["rotation"]["logits"]["largest"])],
             ["rotation without the fold (control)", g(q["rotation_without_the_fold_control"]["smallest"]), g(q["rotation_without_the_fold_control"]["largest"])]]
    out += lines_for("Qwen3-0.6B (a norm of the queries and the keys between the projection and the rotary embedding), float64: relative differences of the scores", ["transformation", "smallest", "largest"], rows)
gp = load("outside-gpt2")
if gp:
    t = gp["layernorm_transformations"]
    rows = [["L1: the all-ones vector added to the outputs of the two output projections of every layer", g(t["L1_all_ones_added_to_the_outputs_of_the_two_output_projections"]["smallest"]), g(t["L1_all_ones_added_to_the_outputs_of_the_two_output_projections"]["largest"])],
            ["L2: the dual shift of the readers of the two LayerNorms", g(t["L2_dual_shift_of_the_readers_of_the_two_layernorms"]["smallest"]), g(t["L2_dual_shift_of_the_readers_of_the_two_layernorms"]["largest"])],
            ["L3: the embedding rows of five tokens shifted; the columns of the scores that do not change", "", g(t["L3_embedding_rows_of_five_tokens_shifted"]["unshifted_columns_largest"])],
            ["L3: the columns that change, against the prediction", "", g(t["L3_embedding_rows_of_five_tokens_shifted"]["shifted_columns_against_prediction_largest"])]]
    rows += [[f"learned absolute positions: the positions shifted by {k} (must fail)", g(v["smallest"]), g(v["largest"])] for k, v in gp["absolute_positions"].items()]
    out += lines_for("GPT-2 (LayerNorm, learned absolute positions, tied head), float64: relative differences of the scores", ["transformation", "smallest", "largest"], rows)
py = load("outside-pythia-160m")
if py:
    t = py["layernorm_transformations"]
    rows = [["translation of the positions (1, 16, 128)", g(min(v["smallest"] for v in py["translation"].values())), g(max(v["largest"] for v in py["translation"].values()))],
            ["dilation of the angles (0.5, 2)", g(min(v["smallest"] for v in py["dilation"].values())), g(max(v["largest"] for v in py["dilation"].values()))]]
    rows += [[k, g(v["smallest"]), g(v["largest"])] for k, v in t.items()]
    out += lines_for("Pythia-160M (LayerNorm, parallel residual, rotary on a quarter of each head), float64: relative differences of the scores", ["transformation", "smallest", "largest"], rows)

# ------------------------------------------------------------------ conversation and tool call
rows = []
for key, label in [("chat-qwen3-0.6b", "Qwen3-0.6B"), ("chat-smol-instruct", "SmolLM2-135M-Instruct")]:
    c = load(key)
    if c is None:
        continue
    for w, d in c["prompts"].items():
        v = d["variants"]
        same = sum(1 for n, r in v.items() if n != "broken_control" and r["identical_tokens"])
        tot = sum(1 for n in v if n != "broken_control")
        txt = d["original_text"].replace("\n", "\\n").replace("|", "\\|")
        rows.append([label, w, d["prompt_tokens"], f"`{txt[:110]}`", f"{same} of {tot}", "different" if not v["broken_control"]["identical_tokens"] else "SAME"])
out += lines_for("Conversation and tool call, float64, greedy, 64 new tokens: the original's answer and how many of the transformed copies give the same tokens", ["system", "prompt", "prompt tokens", "the original's answer", "variants with identical tokens", "control (not applied to the embedding)"], rows)

# ------------------------------------------------------------------ fine-tuning
rows = []
for key, label in [("finetune-smol-instruct", "SmolLM2-135M-Instruct"), ("finetune-delphi-100k", "delphi 100k")]:
    f = load(key)
    if f is None:
        rows.append([label, "not run", "", "", "", "", ""])
        continue
    for opt, d in f["optimizers"].items():
        v = d["variants"]
        rows.append([label, opt, f"{d['loss_first_step']:.3f} → {d['loss_last_step']:.3f}", f"{d['function_change_of_the_reference_run_relative']:.3g}"] +
                    [g(v[n]["difference_after_training_over_the_change_of_the_reference_run"], "{:.2g}") for n in ("signed_permutation_of_the_hidden_coordinates", "rotation_of_the_stream", "scaling_of_the_feed_forward_units", "invertible_matrix_on_the_value_dimensions")])
out += lines_for("Ten steps of fine-tuning on one fixed text from the original and from transformed copies: the difference of the functions after training, over the change the reference run made", ["system", "optimizer", "loss, first → last step", "change of the function by the reference run (relative)", "signed permutation", "rotation of the stream", "scaling of the units", "invertible matrix on the value dimensions"], rows)

# ------------------------------------------------------------------ the covariant optimizer
o = load("opt-field")
if o:
    short = {"unit permutation": "unit permutation", "signed permutation of the hidden coordinates": "signed permutation", "head swap": "head swap", "rotation of the stream": "rotation of the stream", "unit scaling by +-2^k": "unit scaling ±2^k",
             "invertible matrix on the value dimensions": "invertible matrix on the value dimensions", "complex scalar on the rotary plane": "complex scalar on a rotary plane"}
    gauges = list(next(iter(o["rules"].values())).keys())
    names = {"sgd": "gradient descent", "adam": "Adam (first step)", "ngd-pinv": "natural gradient, pseudo-inverse", "ngd-euclid": "natural gradient, damped by the identity of the constants (F + 1e-9 I)",
             "ngd-covdamp:1e-6": "natural gradient, damped by the metric of the scores (F + λ JᵀJ, λ = 1e-6 times the mean nonzero eigenvalue of F)", "ngd-covdamp:1e-2": "natural gradient, damped by the metric of the scores (F + λ JᵀJ, λ = 1e-2 times the mean nonzero eigenvalue of F)"}
    rows = [[names.get(r, r)] + [g(row[gg]["largest"]) for gg in gauges] + [f"{o['median_step_norm_over_the_step_norm_of_sgd'][r]:.3g}"] for r, row in o["rules"].items()]
    out += lines_for("The velocity of the predictive distributions under each training rule, at ten random points: the largest relative difference between the velocity at θ and at gθ, for seven equivalence transformations g", ["rule"] + [short.get(x, x) for x in gauges] + ["median norm of the step in the constants, over that of gradient descent"], rows)
rows = []
for key, label in (("opt-sgd", "gradient descent, lr 1.0"), ("opt-adam", "Adam, lr 0.01")):
    t = load(key)
    if t is None:
        rows.append([label, "not run"])
        continue
    for gg, v in t["gauges"].items():
        rows.append([label, short.get(gg, gg) if o else gg, g(v["max_difference"])])
if rows:
    out += lines_for("Trajectories of 200 steps in the constants from θ and from gθ: the largest difference of the predictive distributions along the trajectory", ["rule", "transformation g", "largest difference"], rows)

# ------------------------------------------------------------------ the group at a point of training
fb = load("fibre")
if fb:
    runs = fb.get("runs", [])
    rows = []
    for r in runs:
        i, t = r["initial"], r["trained"]
        rows.append([r["seed"], f"{r['loss_first']:.3f} → {r['loss_last']:.3f}", r["expected"], i["deficiency"], f"{i['sv_at_cut'] / i['sv_after_cut']:.2g}", t["deficiency"], f"{t['sv_at_cut'] / t['sv_after_cut']:.2g}"])
    out += lines_for(f"The group at a point that training has reached: the rank deficiency of the Jacobian at initialisation and after {runs[0]['steps'] if runs else '?'} steps of Adam (learning rate {fb.get('learning_rate')}, target Dirichlet({fb.get('alpha_of_the_target')}))",
                     ["seed", "loss, first → last step", "deficiency expected", "deficiency at initialisation", "singular-value gap there", "deficiency after training", "singular-value gap there"], rows)

# ------------------------------------------------------------------ the finite instance as a checkpoint
bt, bc = load("bridge-torch"), load("bridge-candle")
if bt:
    rows = [["Transformers, float64", g(bt["float64"]["relative_difference"]), bt["float64"]["argmax_agreement"], bt["float64"]["contexts"]],
            ["Transformers, float32", g(bt["float32"]["relative_difference"]), bt["float32"]["argmax_agreement"], bt["float32"]["contexts"]]]
    if bc:
        rows.append(["candle, float32", g(bc["candle_float32"]["relative_difference"]), bc["candle_float32"]["argmax_agreement"], bc["candle_float32"]["contexts"]])
    out += lines_for("The finite instance as a checkpoint, on every input (625 sequences of four symbols, 780 contexts), against the numpy float64 reference", ["runtime", "largest relative difference of the scores", "agreement of the largest score", "contexts"], rows)

# ------------------------------------------------------------------ discrete symmetries
dd = load("discrete")
if dd:
    rows = []
    for fam, rs in dd["families"].items():
        rows.append([fam, rs[0]["candidates"], ", ".join(str(r["invariant"]) for r in rs), ", ".join(str(r["violated"]) for r in rs), ", ".join(str(r["unclear"]) for r in rs)])
    out += lines_for("Discrete candidates, counted (one entry per random draw)", ["family", "candidates", "invariant", "violated", "unclear"], rows)

# ------------------------------------------------------------------ mutants
mm = load("mutate")
if mm:
    rows = [[k, v["file"].split("/")[-1], v["why"][:100], v["predicted"], "detected" if v["detected"] else "survives", f"{sum(1 for c in v['per_configuration'].values() if c['status'] != 'equal')} of {len(v['per_configuration'])}"] for k, v in mm["mutants"].items()]
    out += lines_for("Mutants of the Python definition, the surrogates and the generator, run through the conformance check against the stored Lean output", ["mutant", "file", "what it changes", "predicted", "result", "configurations that differ"], rows)

print("\n".join(out))
