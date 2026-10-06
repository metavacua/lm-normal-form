# Interface: a model identity record, version 1

A record says which tensors a model's weights are made of, without saying what they are called or which file holds
them, and keeps what is called and where beside it. Two files that hold the same tensors give records with the same
`bag_root`; a file that differs in one bit of one tensor does not. This document is its public interface. A change
to anything below changes the version; a change to anything else does not.

## Identity

- Name: `lmnf-identity-record`. Version: **1**. Canonicalization `lmnf-tensor` 1, hash `sha256`.
- Implemented by `batches/0011/identity_core.py` (the digest, the roots, the verdicts: standard library only) and
  `batches/0011/identity.py` (readers of safetensors, ONNX and burnpack files, and the comparison of two records),
  at the commit that declares version 1.

## The header

A record is a JSON object. These seven fields are read before anything else, in this order, and their shape does not
change in any later version.

| field | content | a reader that does not understand it |
|---|---|---|
| `format` | `"lmnf-identity-record"` | `NOT-THIS-FORMAT`; it does not guess |
| `format_version` | integer, the highest revision whose features the writer used; informative | no verdict of its own; it must be at least `min_writer_version` |
| `min_reader_version` | integer, at most `min_writer_version`: the smallest revision a reader must implement | `UNSUPPORTED-READ-VERSION`; no partial result |
| `min_writer_version` | integer, at most `format_version`: the smallest revision a tool must implement to change or re-emit the record | the record is read and verified, any edit is refused (`READ-ONLY` in write mode) |
| `features` | list of `{name, needed_by}`, `needed_by` one of `read`, `write`, `none` | an unknown `read` name: `UNKNOWN-REQUIRED-FEATURE`; an unknown `write` name: `READ-ONLY` in write mode; an unknown `none` name: ignored |
| `scheme` | `{canonicalization: {id, version}, hash: {alg}}` | any difference from `lmnf-tensor` 1 and `sha256`: `UNVERIFIABLE-SCHEME`, never `MISMATCH`, and no default is substituted |
| `producer` | `{name, version}` | provenance only: it never changes a verdict |

Unknown top-level keys, and `environment`, are ignored. A record that breaks `1 <= min_reader_version <=
min_writer_version <= format_version`, or whose `features` are not a list of `{name: string, needed_by: read|write|none}`,
is `MALFORMED`.

## The verdict

One word, the first rule that applies, in this order: `NOT-THIS-FORMAT`, `MALFORMED`, `UNSUPPORTED-READ-VERSION`,
`UNKNOWN-REQUIRED-FEATURE`, `READ-ONLY` (write mode only), `MALFORMED` (a `scheme` that is not an object),
`UNVERIFIABLE-SCHEME`, `MISMATCH` (the `bag_root` of the listed tensors is not the recorded one), `OK`. The version is
read before the digests: a record with a wrong root and a future `min_reader_version` is `UNSUPPORTED-READ-VERSION`.

## Semantics of `lmnf-tensor` 1

- **A tensor's digest.** SHA-256 of these bytes, in order: `lmnf-tensor`, 0x00, the version `1`, 0x00, `tensor`,
  0x00, the dtype name, 0x00, the rank as u32 little-endian, each dimension as u64 little-endian, then the elements in
  C order, little-endian. The dtype names are `f64 f32 i64 i32 i16 i8 u64 u32 u16 u8 bool`; a stored `bf16` or `f16`
  is widened to `f32` first, which is exact, and is recorded as `stored_dtype`. Names, container, file order, padding
  and the orientation of a matrix are not in the digest. Dtype class, shape and every bit of every element are.
- **`digest_t`.** For a rank-2 tensor, the digest of its transpose; otherwise `null`. A matrix is stored `[out, in]`
  in a safetensors file and `[in, out]` in ONNX and Burn; the record does not decide which is right.
- **The key** of a tensor is the smaller of `digest` and `digest_t` (as lowercase hexadecimal strings); a matrix and its
  transpose have one key. Two tensors match when one's `digest` is the other's `digest` (identical) or `digest_t`
  (transposed).
- **`bag_root`.** SHA-256 of `lmnf-bag`, 0x00, `1`, 0x00, then the distinct keys as 32 raw bytes each, sorted. A tensor
  stored twice (a tied embedding) is one member.
- **`labelled_root`** (safetensors records only). SHA-256 of `lmnf-labelled`, 0x00, `1`, 0x00, the container name, 0x00,
  then, in order of label, the label as UTF-8, 0x00 and the digest as 32 raw bytes. It is the identity of a container
  whose names are its only wiring.
- **Labels** (`labels`: container to name) and any key beginning `x-` on a tensor entry are provenance. Changing them
  changes no digest and no root.

## The subject

`subject` has `container` (`safetensors`, `onnx`, `burnpack`, `fixture`), `tensors` (each `{digest, digest_t, dtype,
stored_dtype, shape, labels}`), `bag_root`, and optionally `labelled_root` and, for ONNX, `constants` (the values of
`Constant` nodes, beside the initializers, outside the bag). `environment` (Python and package versions, processor)
is never part of an identity. A claim about a model (an engine's output, a difference between two engines) is
kept under `(bag_root, environment)`, because the same weights run by another engine or on another processor may
give other bits.

## Compatibility

- Compatible: a new key in a record or in a tensor entry; a new feature with `needed_by` `none`; a new container name; a
  new `x-` key.
- Compatible with a raised `min_writer_version`: a feature a reader may ignore but a tool that rewrites the record
  must not lose (a signature over the roots).
- Breaking: any change to the preimage of a digest or a root (this is `lmnf-tensor` 2, and every digest changes,
  as the fixture checks); a new dtype name in the digest; any change to the header fields, the order of the verdicts
  or the meaning of a feature. Breaking changes raise `format_version` and the canonicalization version, and the batch
  documents name the version they used.
- While `format_version` is 1 a reader implements version 1 only: every `min_reader_version` above 1 is refused.

## Conformance

A fixture under `conformance/identity/1/`: `digests.tsv` (eleven tensors with the digest of each, and of the transpose
of the matrices), `relations.tsv` (which digests must be equal and which must differ: a bf16 and an f16 copy of an f32
tensor; the same bytes under another dtype; the same bytes under another shape), `records/` (sixteen records and one
more with a tensor listed twice) and `expected.tsv` (the verdict of each, in read mode and, where it differs, in write
mode). Every expected value was derived by `conformance/identity/1/derive.sh` with `printf`, `xxd`, `sha256sum` and
`jq`, not by the code it tests. The batch workflow runs `identity_core.py conformance` before any cell and stops on
any difference: 53 checks, including that every digest changes when the version in the preimage does.

## Known limits

- The bag does not see wiring. Exchanging the contents of two tensors of one shape, in a safetensors file, leaves the
  `bag_root` as it was; the `labelled_root` sees it, because there the names are the structure. In an ONNX file the
  structure is the graph, which the bag does not hash.
- A matrix and its transpose are one key, so a square matrix stored transposed by mistake is not seen by the record;
  only a function test sees it.
- Weights that differ by a symmetry of the model (a permutation of the units of the feed-forward layers, a sign) have
  different digests. The record does not quotient them out.
- `-0.0` and `0.0` differ, being different bits. NaNs are not covered: the fixture has none, and the two routes
  that widen `bf16` and `f16` (numpy, and the standard-library helper in `identity_core.py`) are checked against each
  other on finite values only.
- Only the dtype names above are defined; a tensor of another dtype is an error, not a digest.
