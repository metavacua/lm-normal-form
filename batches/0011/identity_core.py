# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The identity record of docs/interfaces/identity.md, version 1, with nothing outside the standard
# library: the digest of a tensor, the roots of a bag of tensors, the header rules and their verdicts.
# The readers of containers (safetensors, ONNX, burnpack) are in identity.py.
# Usage: identity_core.py conformance DIR    run the fixture in DIR (conformance/identity/1)
#        identity_core.py verify FILE [read|write]
import hashlib, json, struct, sys

FORMAT = "lmnf-identity-record"
SCHEME = ("lmnf-tensor", 1, "sha256")     # canonicalization id, its version, hash algorithm
SUPPORTED = {"read": 1, "write": 1}       # the highest min_reader_version and min_writer_version understood
FEATURES = frozenset()                    # the feature names understood
BYTES = {"f64": 8, "f32": 4, "i64": 8, "i32": 4, "i16": 2, "i8": 1, "u64": 8, "u32": 4, "u16": 2, "u8": 1, "bool": 1}
STORED = {"bf16": 2, "f16": 2, **BYTES}   # dtypes a tensor may be stored in; bf16 and f16 are widened to f32


def widen(dtype, raw):
    """bf16 and f16 to f32, which is exact; every other dtype is returned unchanged."""
    if dtype == "bf16":
        out = bytearray(2 * len(raw))
        out[2::4], out[3::4] = raw[0::2], raw[1::2]
        return "f32", bytes(out)
    if dtype == "f16":
        n = len(raw) // 2
        return "f32", struct.pack(f"<{n}f", *struct.unpack(f"<{n}e", raw))
    return dtype, raw


def tensor_digest(dtype, shape, raw, version=1):
    """SHA-256 of: scheme tags, canonical dtype, rank, dims, then the little-endian C-order bytes."""
    if dtype not in STORED:
        raise ValueError(f"unsupported dtype {dtype}")
    n = 1
    for d in shape:
        n *= d
    if len(raw) != n * STORED[dtype]:
        raise ValueError(f"{len(raw)} bytes for shape {list(shape)} of {dtype}")
    dtype, raw = widen(dtype, raw)
    h = hashlib.sha256(b"lmnf-tensor\0" + str(version).encode() + b"\0tensor\0" + dtype.encode() + b"\0")
    h.update(struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape))
    h.update(raw)
    return h.hexdigest()


def transpose2(shape, size, raw):
    """The bytes of a rank-2 tensor transposed, element by element (for small tensors)."""
    r, c = shape
    return b"".join(raw[(i * c + j) * size:(i * c + j + 1) * size] for j in range(c) for i in range(r))


def key(entry):
    """The orientation class of a tensor: the smaller of its digest and its transpose's."""
    return min(entry["digest"], entry.get("digest_t") or entry["digest"])


def bag_root(entries):
    """The distinct keys, sorted: a tensor stored twice (a tied embedding) is one member."""
    return hashlib.sha256(b"lmnf-bag\0" + b"1\0" + b"".join(sorted({bytes.fromhex(key(e)) for e in entries}))).hexdigest()


def labelled_root(entries, container):
    """For a container whose names are its only wiring (safetensors): the labels and digests together."""
    h = hashlib.sha256(b"lmnf-labelled\0" + b"1\0" + container.encode() + b"\0")
    for label, digest in sorted((e["labels"][container], e["digest"]) for e in entries):
        h.update(label.encode() + b"\0" + bytes.fromhex(digest))
    return h.hexdigest()


def envelope(subject, producer, environment=None):
    rec = {"format": FORMAT, "format_version": 1, "min_reader_version": 1, "min_writer_version": 1, "features": [],
           "scheme": {"canonicalization": {"id": SCHEME[0], "version": SCHEME[1]}, "hash": {"alg": SCHEME[2]}},
           "producer": producer, "subject": subject}
    if environment is not None:
        rec["environment"] = environment
    return rec


def verify(rec, mode="read"):
    """One verdict, the first rule that applies: the header before the scheme, the scheme before the digests."""
    if not isinstance(rec, dict) or rec.get("format") != FORMAT:
        return "NOT-THIS-FORMAT"
    nums = [rec.get(k) for k in ("format_version", "min_reader_version", "min_writer_version")]
    if not all(type(x) is int and x >= 1 for x in nums) or not nums[1] <= nums[2] <= nums[0]:
        return "MALFORMED"
    feats = rec.get("features", [])
    if not isinstance(feats, list) or any(
            not isinstance(f, dict) or not isinstance(f.get("name"), str) or f.get("needed_by") not in ("read", "write", "none")
            for f in feats):
        return "MALFORMED"
    if nums[1] > SUPPORTED["read"]:
        return "UNSUPPORTED-READ-VERSION"
    unknown = [f["needed_by"] for f in feats if f["name"] not in FEATURES]
    if "read" in unknown:
        return "UNKNOWN-REQUIRED-FEATURE"
    if mode == "write" and (nums[2] > SUPPORTED["write"] or "write" in unknown):
        return "READ-ONLY"
    s = rec.get("scheme")
    if not (isinstance(s, dict) and isinstance(s.get("canonicalization"), dict) and isinstance(s.get("hash"), dict)):
        return "MALFORMED"
    if (s["canonicalization"].get("id"), s["canonicalization"].get("version"), s["hash"].get("alg")) != SCHEME:
        return "UNVERIFIABLE-SCHEME"
    sub = rec.get("subject")
    if isinstance(sub, dict) and "tensors" in sub and "bag_root" in sub:
        try:
            return "OK" if bag_root(sub["tensors"]) == sub["bag_root"] else "MISMATCH"
        except (KeyError, TypeError, ValueError):
            return "MALFORMED"
    return "OK"


def table(path):
    rows = [line.rstrip("\n").split("\t") for line in open(path, encoding="utf-8")]
    return rows[1:]


def conformance(d):
    """Run the fixture: every digest, every relation between digests, the effect of the version token, every verdict."""
    checks = bad = 0

    def check(what, expected, got):
        nonlocal checks, bad
        checks += 1
        if expected != got:
            bad += 1
            print(f"DIFFERS {what}: expected {expected}, got {got}")

    digests = {}
    for name, dt, shape, raw, want, want_t in table(f"{d}/digests.tsv"):
        shape = tuple(int(x) for x in shape.split(",")) if shape else ()
        raw = bytes.fromhex(raw)
        digests[name] = tensor_digest(dt, shape, raw)
        check(f"digest {name}", want, digests[name])
        check(f"digest of {name} under another version", True, tensor_digest(dt, shape, raw, version=2) != want)
        if want_t != "-":
            check(f"digest_t {name}", want_t, tensor_digest(dt, shape[::-1], transpose2(shape, STORED[dt], raw)))
    for rel, a, b in table(f"{d}/relations.tsv"):
        check(f"{rel} {a} {b}", rel == "equal", digests[a] == digests[b])
    for name, mode, want in table(f"{d}/expected.tsv"):
        check(f"{name} ({mode})", want, verify(json.load(open(f"{d}/records/{name}.json", encoding="utf-8")), mode))
    print(f"identity fixture: {checks} checks, {bad} differ")
    return bad == 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "conformance":
        sys.exit(0 if conformance(sys.argv[2]) else 1)
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        print(verify(json.load(open(sys.argv[2], encoding="utf-8")), sys.argv[3] if len(sys.argv) > 3 else "read"))
        sys.exit(0)
    sys.exit(__doc__)
