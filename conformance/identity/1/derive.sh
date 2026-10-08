#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Derives the expected values of the identity fixture (docs/interfaces/identity.md, version 1) with
# printf, xxd, sha256sum and jq only, not with batches/0011/identity_core.py: the digests and the
# roots are computed independently of the code they test; the verdict rows (expected.tsv) are
# written here by hand, as examples of the rules, and are not computed. Writes digests.tsv, relations.tsv, records/*.json and
# expected.tsv next to this script. The files are frozen: run this again only to change the version.
set -euo pipefail
cd "$(dirname "$0")"
hx() { printf '%s' "$1" | xxd -p | tr -d '\n'; }                              # ASCII string -> hex
le32() { printf '%08x' "$1" | sed 's/\(..\)\(..\)\(..\)\(..\)/\4\3\2\1/'; }    # u32, little-endian, hex
le64() { printf '%016x' "$1" | sed 's/\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)/\8\7\6\5\4\3\2\1/'; }
sha() { sha256sum | cut -d' ' -f1; }
# digest DTYPE SHAPE(comma separated, empty for a scalar) RAWHEX, the dtype and raw bytes already canonical (f32 for floats)
digest() {
  local dt=$1 shape=$2 raw=$3 dims="" d; local -a D=()
  [ -n "$shape" ] && IFS=, read -ra D <<< "$shape"
  for d in ${D[@]+"${D[@]}"}; do dims+=$(le64 "$d"); done
  { hx 'lmnf-tensor'; printf 00; hx 1; printf 00; hx tensor; printf 00; hx "$dt"; printf 00; le32 "${#D[@]}"; printf '%s%s' "$dims" "$raw"; } | xxd -r -p | sha
}
F1=0000803f F2=00000040 F3=00004040 F4=00008040 F5=0000a040 F6=0000c040      # 1.0 .. 6.0 as float32, little-endian
{
  printf 'name\tdtype\tshape\traw\tdigest\tdigest_t\n'
  row() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "${6:--}"; }
  row f32-pair   f32  2   "$F1$F2" "$(digest f32 2 "$F1$F2")"
  row bf16-pair  bf16 2   "803f0040" "$(digest f32 2 "$F1$F2")"             # bf16 1.0 and 2.0, widened by hand to the f32 above
  row f16-pair   f16  2   "003c0040" "$(digest f32 2 "$F1$F2")"             # f16 1.0 and 2.0
  row f32-2x3    f32  2,3 "$F1$F2$F3$F4$F5$F6" "$(digest f32 2,3 "$F1$F2$F3$F4$F5$F6")" "$(digest f32 3,2 "$F1$F4$F2$F5$F3$F6")"
  row i64-minus1 i64  1   "ffffffffffffffff" "$(digest i64 1 ffffffffffffffff)"
  row f32-scalar f32  ""  "$F1" "$(digest f32 '' "$F1")"
  row f32-one    f32  1   "$F1" "$(digest f32 1 "$F1")"
  row i32-one    i32  1   "$F1" "$(digest i32 1 "$F1")"                    # the same four bytes, another dtype
  row f32-4      f32  4   "$F1$F2$F3$F4" "$(digest f32 4 "$F1$F2$F3$F4")"
  row f32-2x2    f32  2,2 "$F1$F2$F3$F4" "$(digest f32 2,2 "$F1$F2$F3$F4")" "$(digest f32 2,2 "$F1$F3$F2$F4")"  # the same bytes, another shape
  row bool-pair  bool 2   "0001" "$(digest bool 2 0001)"
} > digests.tsv
col() { awk -F'\t' -v n="$1" -v c="$2" '$1==n{print $c}' digests.tsv; }
{
  printf 'relation\ta\tb\n'
  printf 'equal\tf32-pair\tbf16-pair\nequal\tf32-pair\tf16-pair\n'
  printf 'differ\tf32-one\ti32-one\ndiffer\tf32-4\tf32-2x2\n'
} > relations.tsv
# the root of a bag: the distinct 32-byte keys, each the smaller of the digest and its transpose's, sorted, after the scheme tags
A=$(col f32-pair 5) B=$(col f32-2x3 5) BT=$(col f32-2x3 6) C=$(col i64-minus1 5)
KB=$(printf '%s\n%s\n' "$B" "$BT" | sort | head -n 1)
BAG=$({ hx 'lmnf-bag'; printf 00; hx 1; printf 00; printf '%s\n' "$A" "$KB" "$C" | sort -u | tr -d '\n'; } | xxd -r -p | sha)
base=$(jq -n --arg a "$A" --arg b "$B" --arg bt "$BT" --arg c "$C" --arg bag "$BAG" '{
  format: "lmnf-identity-record", format_version: 1, min_reader_version: 1, min_writer_version: 1, features: [],
  scheme: {canonicalization: {id: "lmnf-tensor", version: 1}, hash: {alg: "sha256"}},
  producer: {name: "conformance/identity/1/derive.sh", version: "1"},
  subject: {container: "fixture", tensors: [
    {digest: $a, digest_t: null, dtype: "f32", stored_dtype: "f32", shape: [2], labels: {fixture: "a"}},
    {digest: $b, digest_t: $bt, dtype: "f32", stored_dtype: "f32", shape: [2,3], labels: {fixture: "b"}},
    {digest: $c, digest_t: null, dtype: "i64", stored_dtype: "i64", shape: [1], labels: {fixture: "c"}}], bag_root: $bag}}')
mk() { printf '%s\n' "$base" | jq -S "$2" > "records/$1.json"; }       # name, jq filter applied to the valid record
mk 01-valid                '.'
mk 02-other-format         '{format: "something-else", format_version: 1}'
mk 03-min-reader-99        '.format_version = 99 | .min_writer_version = 99 | .min_reader_version = 99'
mk 04-min-reader-2         '.format_version = 2 | .min_writer_version = 2 | .min_reader_version = 2'
mk 05-unknown-read-feature '.features = [{name: "x-attestation", needed_by: "read"}] | .format_version = 2 | .min_writer_version = 2 | .min_reader_version = 1'
mk 06-unknown-write-feature '.features = [{name: "x-signature", needed_by: "write"}] | .format_version = 2 | .min_writer_version = 2 | .min_reader_version = 1'
mk 07-unknown-none-feature '.features = [{name: "x-note", needed_by: "none"}] | .format_version = 2'
mk 08-min-writer-2         '.format_version = 2 | .min_writer_version = 2'
mk 09-bad-root-and-version '.subject.bag_root = ("0" * 64) | .format_version = 2 | .min_writer_version = 2 | .min_reader_version = 2'
mk 10-bad-root             '.subject.bag_root = ("0" * 64)'
mk 11-unknown-scheme       '.scheme.hash.alg = "sha256-v99" | .subject.bag_root = ("0" * 64)'
mk 12-other-producer       '.producer = {name: "another tool", version: "99.0.0"}'
mk 13-order-violated       '.format_version = 2 | .min_writer_version = 1 | .min_reader_version = 2'
mk 14-unknown-key          '. + {"x-extra": {"anything": [1, 2, 3]}}'
mk 15-environment          '. + {environment: {cpu: "any", python: "any"}}'
mk 16-bad-feature-kind     '.features = [{name: "x-note", needed_by: "sometimes"}]'
mk 17-tensor-twice         '.subject.tensors += [.subject.tensors[0]]'
mk 18-no-subject           'del(.subject)'
mk 19-subject-no-root      'del(.subject.bag_root)'
{
  printf 'file\tmode\tverdict\n'
  for m in read write; do printf '01-valid\t%s\tOK\n' "$m"; done
  printf '02-other-format\tread\tNOT-THIS-FORMAT\n03-min-reader-99\tread\tUNSUPPORTED-READ-VERSION\n04-min-reader-2\tread\tUNSUPPORTED-READ-VERSION\n'
  printf '05-unknown-read-feature\tread\tUNKNOWN-REQUIRED-FEATURE\n05-unknown-read-feature\twrite\tUNKNOWN-REQUIRED-FEATURE\n'
  printf '06-unknown-write-feature\tread\tOK\n06-unknown-write-feature\twrite\tREAD-ONLY\n'
  printf '07-unknown-none-feature\tread\tOK\n07-unknown-none-feature\twrite\tOK\n'
  printf '08-min-writer-2\tread\tOK\n08-min-writer-2\twrite\tREAD-ONLY\n'
  printf '09-bad-root-and-version\tread\tUNSUPPORTED-READ-VERSION\n10-bad-root\tread\tMISMATCH\n11-unknown-scheme\tread\tUNVERIFIABLE-SCHEME\n'
  printf '12-other-producer\tread\tOK\n12-other-producer\twrite\tOK\n13-order-violated\tread\tMALFORMED\n'
  printf '14-unknown-key\tread\tOK\n14-unknown-key\twrite\tOK\n15-environment\tread\tOK\n15-environment\twrite\tOK\n'
  printf '16-bad-feature-kind\tread\tMALFORMED\n17-tensor-twice\tread\tOK\n'
  printf '18-no-subject\tread\tMALFORMED\n19-subject-no-root\tread\tMALFORMED\n'
} > expected.tsv
echo "bag root of the valid record: $BAG"
