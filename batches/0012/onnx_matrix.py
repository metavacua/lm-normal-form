# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The version-bearing fields of an ONNX file, one change at a time. For each row of predictions.tsv the graph is
# changed as the row says, written, given to onnx.checker, and given to ONNX Runtime in a process of its own (batches/0010/
# logits.py); what each did is recorded as one of: accept or refuse (the checker); same, near, different or refuse
# (ONNX Runtime: the logits byte for byte as the unchanged graph's, within 1e-4 with the same argmax, beyond that,
# or an error). Then the observation is set against the prediction of the row.
# A change is a list of steps separated by ';':
#   ir=N | ir=clear           the ModelProto's ir_version
#   imports=D|V,D|V,...       the opset_import entries, replaced (empty: none); D is a domain, V a version
#   addimport=D|V             one more entry
#   meta=dup | meta=inert     two metadata_props with one key | producer, domain, model_version, doc_string, graph name
#   unknown=field             a field the schema has not got, after the message
#   nodedomain=D              the domain of every node
#   dupinit                   the first initializer a second time
#   convert=N                 onnx.version_converter to opset N (a failure of the converter is the observation)
# Usage: onnx_matrix.py MODEL.onnx WORKDIR PREDICTIONS.tsv OUT_PREFIX IDS_FILE VOCAB [ROW_ID ...]
#        onnx_matrix.py selftest
import csv, os, subprocess, sys, tempfile, time
import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
LOGITS = os.path.join(HERE, "..", "0010", "logits.py")


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def change(m, spec):
    """The model changed as spec says (a new message), or an exception that is the converter's failure."""
    for step in (s for s in spec.split(";") if s):
        k, _, v = step.partition("=")
        if k == "ir":
            m.ClearField("ir_version") if v == "clear" else setattr(m, "ir_version", int(v))
        elif k == "imports":
            del m.opset_import[:]
            for e in (x for x in v.split(",") if x):
                d, _, n = e.partition("|")
                o = m.opset_import.add()
                o.domain, o.version = d, int(n)
        elif k == "addimport":
            d, _, n = v.partition("|")
            o = m.opset_import.add()
            o.domain, o.version = d, int(n)
        elif k == "meta" and v == "dup":
            for x in ("1", "2"):
                p = m.metadata_props.add()
                p.key, p.value = "k", x
        elif k == "meta" and v == "inert":
            m.producer_name, m.producer_version, m.domain, m.model_version, m.doc_string = "other", "99", "x.y", 7, "doc"
            m.graph.name = "g2"
        elif k == "unknown" and v == "field":
            r = onnx.ModelProto()
            r.ParseFromString(m.SerializeToString() + varint(9999 << 3) + varint(1))
            m = r
        elif k == "nodedomain":
            for n in m.graph.node:
                n.domain = v
        elif k == "dupinit":
            m.graph.initializer.add().CopyFrom(m.graph.initializer[0])
        elif k == "convert":
            m = onnx.version_converter.convert_version(m, int(v))
        else:
            raise ValueError(f"unknown step {step}")
    return m


def first_line(text, n=300):
    lines = [x for x in str(text).strip().splitlines() if x.strip()]
    return (lines[-1] if lines else "").replace("\t", " ")[:n]


def run_ort(path, outdir, level, ids, env):
    os.makedirs(outdir, exist_ok=True)
    e = dict(os.environ)
    if env and env != "-":
        e[env.split("=")[0]] = env.split("=", 1)[1]
    t = time.time()
    r = subprocess.run([sys.executable, LOGITS, path, outdir, level] + ids, capture_output=True, text=True, env=e, timeout=900)
    return r.returncode, first_line(r.stderr), time.time() - t


def compare(base, other, vocab, n):
    """same, near or different, and the largest difference, of the logits of n prompts."""
    worst, same, argmax = 0.0, True, True
    for k in range(n):
        a = np.fromfile(f"{base}/{k}.f32", np.float32).reshape(-1, vocab)
        b = np.fromfile(f"{other}/{k}.f32", np.float32).reshape(-1, vocab)
        if a.shape != b.shape:
            return "different", float("inf")
        same = same and bool(np.array_equal(a, b))
        worst = max(worst, float(np.abs(a - b).max()))
        argmax = argmax and bool((a.argmax(1) == b.argmax(1)).all())
    return ("same" if same else "near" if worst <= 1e-4 and argmax else "different"), worst


def hit(predicted, observed, text, wanted):
    """Does the observation match: the class is among the predicted ones, and, for a refusal, one wanted text is in the error."""
    ok = observed in predicted.split("/")
    if ok and observed == "refuse" and wanted not in ("", "-"):
        ok = any(w.lower() in text.lower() for w in wanted.split("||"))
    return ok


def main(model, work, predictions, prefix, ids_file, vocab, only):
    ids = open(ids_file).read().split()
    vocab = int(open(vocab).read()) if os.path.exists(vocab) else int(vocab)
    os.makedirs(work, exist_ok=True)
    for level in ("all", "none"):
        code, err, _ = run_ort(model, f"{work}/base-{level}", level, ids, "-")
        if code:
            sys.exit(f"the unchanged graph does not run ({level}): {err}")
    base = onnx.load(model)
    rows = list(csv.DictReader(open(predictions, encoding="utf-8"), delimiter="\t"))
    out, graded = [], []
    for row in rows:
        if only and row["id"] not in only:
            continue
        m = onnx.ModelProto()
        m.CopyFrom(base)
        obs = {"id": row["id"], "mutation": row["mutation"], "level": row["level"], "env": row["env"], "checker": "", "checker_text": "",
               "ort": "", "ort_text": "", "max_abs_diff": "", "converted": ""}
        try:
            m = change(m, row["mutation"])
            obs["converted"] = "yes" if "convert=" in row["mutation"] else ""
        except Exception as e:
            obs.update(checker="convert-fail", checker_text=first_line(f"{type(e).__name__}: {e}"), ort="na")
        if obs["checker"] == "":
            path = f"{work}/row.onnx"
            onnx.save(m, path)
            try:
                onnx.checker.check_model(path)
                obs["checker"] = "accept"
            except Exception as e:
                obs["checker"], obs["checker_text"] = "refuse", first_line(f"{type(e).__name__}: {e}")
            code, err, _ = run_ort(path, f"{work}/row", row["level"], ids, row["env"])
            if code:
                obs["ort"], obs["ort_text"] = "refuse", err
            else:
                obs["ort"], obs["max_abs_diff"] = compare(f"{work}/base-{row['level']}", f"{work}/row", vocab, len(ids))
                obs["max_abs_diff"] = f"{obs['max_abs_diff']:.3g}"
            os.remove(path)
        del m
        out.append(obs)
        g = {"id": row["id"], "mutation": row["mutation"], "risk": row["risk"], "checker_predicted": row["checker"], "checker_observed": obs["checker"],
             "checker_hit": hit(row["checker"], obs["checker"], obs["checker_text"], row["checker_text"]),
             "ort_predicted": row["ort"], "ort_observed": obs["ort"],
             "ort_hit": row["ort"] == "na" or hit(row["ort"], obs["ort"], obs["ort_text"], row["ort_text"])}
        graded.append(g)
        print(f"{g['id']:5s} {row['mutation'][:44]:44s} checker {g['checker_predicted']}->{g['checker_observed']} {'ok' if g['checker_hit'] else 'MISS'}"
              f" | ort {g['ort_predicted']}->{g['ort_observed']} {'ok' if g['ort_hit'] else 'MISS'}", flush=True)
    for name, rs in (("matrix", out), ("graded", graded)):
        with open(f"{prefix}-{name}.tsv", "w", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rs)
    miss = [g for g in graded if not (g["checker_hit"] and g["ort_hit"])]
    print(f"{len(graded)} rows; {len(graded) - len(miss)} as predicted; missed: {[g['id'] for g in miss]}")


def selftest():
    """The plumbing, on a toy graph and on changes that are not rows of the table: an inert one, a truncated file."""
    from onnx import TensorProto, helper
    with tempfile.TemporaryDirectory() as d:
        g = helper.make_graph(
            [helper.make_node("Cast", ["input_ids"], ["f"], to=TensorProto.FLOAT), helper.make_node("Unsqueeze", ["f", "ax"], ["u"]),
             helper.make_node("Mul", ["u", "w"], ["logits"])], "toy",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "n"])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "n", 4])],
            [numpy_helper.from_array(np.array([2], np.int64), "ax"), numpy_helper.from_array(np.array([[[1, 2, 3, 4]]], np.float32), "w")])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
        m.ir_version = 8
        path = f"{d}/toy.onnx"
        onnx.save(m, path)
        pred = f"{d}/p.tsv"
        cols = "id family mutation level env checker checker_text ort ort_text risk note".split()
        rows = [["T1", "toy", "ir=8", "all", "-", "accept", "-", "same", "-", "low", "no change"],
                ["T2", "toy", "graphname=x", "all", "-", "accept", "-", "same", "-", "low", "a step the grammar lacks"]]
        open(pred, "w").write("\t".join(cols) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
        open(f"{d}/ids.txt", "w").write("1,2,3\n4,5\n")
        # T1 runs; T2 must be reported as a failure of the change, not crash the harness
        try:
            main(path, f"{d}/w", pred, f"{d}/out", f"{d}/ids.txt", "4", ["T1"])
        except SystemExit as e:
            print("selftest: the harness stopped:", e)
            return False
        print(open(f"{d}/out-graded.tsv").read())
        try:
            change(m, "graphname=x")
            print("selftest: FAILED, an unknown step was accepted")
            return False
        except ValueError:
            print("selftest: an unknown step is refused")
    return True


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        sys.exit(0 if selftest() else 1)
    main(*sys.argv[1:7], set(sys.argv[7:]))
