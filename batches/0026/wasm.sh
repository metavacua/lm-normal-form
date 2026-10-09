#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The builds of batch 0026 for the model named TAG, whose generated Rust is in work/gen/TAG/model.rs
# (LoadStrategy::File, batches/0010/burn.sh codegen) and work/genb/TAG/model.rs (LoadStrategy::Bytes,
# batches/0026/gen_bytes.rs). Each subcommand's log is out/TAG/NAME.log; the last lines are printed and
# the peak memory and elapsed time go to $GITHUB_OUTPUT, and the step fails when the claim it checks
# is not the case.
# Usage: wasm.sh gen|p2|p1|build|patch|validate TAG [LABEL]
#   gen       gen_bytes on the graph: work/graph/TAG/model.onnx to work/genb/TAG
#   p2        P2: the Bytes source has no `std::` path and no `extern crate std`
#   p1        P1: the File source does not build for wasm32v1-none, with an error that names std and
#             that comes from the crate of this batch (an error in a dependency says nothing of the source)
#   build     P3/P4: the Bytes source builds for wasm32v1-none; the module is kept as out/TAG/LABEL.wasm
#             (LABEL bytes by default)
#   patch     an auxiliary change, made only after `build` failed in a dependency: the cubecl crates
#             of the dependency graph, cloned at the revision the graph names, with cubecl-environment
#             changed to treat target_family = "wasm" as a browser only when target_os is not "none";
#             the crate of this batch gets a [patch] section for it
#   validate  P5: the module (patched.wasm if there is one, else bytes.wasm) validates as WebAssembly 1.0
#             plus mutable globals
set -u -o pipefail
c=$1; t=$2; l=${3:-bytes}
root=$PWD; o=$root/out/$t; w=$root/batches/0026/wasm
mkdir -p "$o"
used() { awk -v p="$1_peak_kb" -v e="$1_elapsed" '/Maximum resident/{print p"="$NF} /Elapsed \(wall/{print e"="$NF}' "$o/$1.log" | tee -a "${GITHUB_OUTPUT:-/dev/null}"; }
timed() { n=$1; shift; rc=0; /usr/bin/time -v "$@" > "$o/$n.log" 2>&1 || rc=$?; tail -n "${LINES_SHOWN:-30}" "$o/$n.log"; used "$n"; return $rc; }
# The crate built for the target, with the model.rs named by $1 (a path).
cargo_wasm() {
  cp "$1" "$w/src/model.rs"
  sed -i "s/@BURN_REV@/$(cat work/burn.rev)/g" "$w/Cargo.toml"
  # The stack of the shadow stack in linear memory: wasm-ld's default is 1 MiB.
  cd "$w" && RUSTFLAGS="-C link-arg=-zstack-size=16777216" cargo build --release --target wasm32v1-none
}
case $c in
  gen)
    mkdir -p "work/genb/$t"
    timed gen "$root/work/burn-onnx/target/release/examples/gen_bytes" "work/graph/$t/model.onnx" "work/genb/$t"; rc=$?
    ls -l "work/genb/$t" | tee "$o/generated-bytes-files.txt"
    cp "work/genb/$t/model.rs" "$o/bytes-model.rs" 2>/dev/null || true
    sha256sum "work/genb/$t/model.rs" "work/gen/$t/model.rs" | tee "$o/model-rs.sha256"
    exit $rc ;;
  p2)
    if grep -n -E 'std::|extern crate std' "work/genb/$t/model.rs" | tee "$o/p2.log"; then echo "P2 refuted: the Bytes source mentions std"; exit 1; fi
    echo "P2: no match in work/genb/$t/model.rs"; exit 0 ;;
  p1)
    timed p1 bash -c "$(declare -f cargo_wasm); w='$w'; cargo_wasm '$root/work/gen/$t/model.rs'"; rc=$?
    if [ $rc -eq 0 ]; then echo "P1 refuted: the File source builds for wasm32v1-none"; exit 1; fi
    if ! grep -q 'could not compile `wasmlm`' "$o/p1.log"; then echo "P1 cannot be checked: the build failed in a dependency, not in the source"; grep -E '^error: could not compile' "$o/p1.log"; exit 1; fi
    if grep -E "^error" "$o/p1.log" | grep -q -E "\bstd\b"; then echo "P1: the build of the source failed with an error that names std"; exit 0; fi
    echo "P1 refuted: the build of the source failed, but no error line names std"; exit 1 ;;
  build)
    n=$l
    timed "$n" bash -c "$(declare -f cargo_wasm); w='$w'; cargo_wasm '$root/work/genb/$t/model.rs'"; rc=$?
    if [ $rc -eq 0 ]; then
      cp "$w/target/wasm32v1-none/release/wasmlm.wasm" "$o/$n.wasm"
      ls -l "$o/$n.wasm" | tee "$o/$n.wasm.size"
      sha256sum "$o/$n.wasm" | tee "$o/$n.wasm.sha256"
    fi
    grep -E "^error" "$o/$n.log" | sort | uniq -c | sort -rn | head -20 | tee "$o/$n.errors.txt"
    exit $rc ;;
  patch)
    mkdir -p "$o"
    sed -i "s/@BURN_REV@/$(cat work/burn.rev)/g" "$w/Cargo.toml"
    (cd "$w" && cargo metadata --format-version 1 --filter-platform wasm32v1-none > "$o/metadata.json") || exit 1
    python3 - "$o/metadata.json" > "$o/cubecl-packages.txt" <<'PY' || exit 1
import json, sys
m = json.load(open(sys.argv[1]))
for p in m["packages"]:
    src = p.get("source") or ""
    if src.startswith("git+https://github.com/tracel-ai/cubecl"):
        print(p["name"], src.split("#")[-1])
PY
    cat "$o/cubecl-packages.txt"
    rev=$(awk '$1=="cubecl-environment"{print $2}' "$o/cubecl-packages.txt")
    [ -n "$rev" ] || { echo "cubecl-environment is not in the dependency graph"; exit 1; }
    git clone --quiet https://github.com/tracel-ai/cubecl work/cubecl && git -C work/cubecl checkout --quiet "$rev"
    grep -rl 'target_family = "wasm"' work/cubecl/crates/cubecl-environment --include=Cargo.toml --include=build.rs --include='*.rs' --exclude-dir=tests | tee "$o/patched-files.txt"
    xargs -a "$o/patched-files.txt" sed -i 's/target_family = "wasm"/all(target_family = "wasm", not(target_os = "none"))/g'
    git -C work/cubecl diff | tee "$o/cubecl-environment.patch"
    printf '\n[patch."https://github.com/tracel-ai/cubecl"]\ncubecl-environment = { path = "%s/work/cubecl/crates/cubecl-environment" }\n' "$root" >> "$w/Cargo.toml"
    exit 0 ;;
  validate)
    m=$o/patched.wasm; [ -f "$m" ] || m=$o/bytes.wasm
    [ -f "$m" ] || { echo "no module was built"; exit 1; }
    echo "module: $m"
    wasm-tools validate --features=wasm1,mutable-global "$m" 2>&1 | tee "$o/validate.log"; rc=${PIPESTATUS[0]}
    wasm-tools print "$m" 2>/dev/null | grep -E '^\s*\((import|export) ' | tee "$o/imports-exports.txt" | head -40
    exit $rc ;;
  *) echo "usage: wasm.sh gen|p2|p1|build|patch|validate TAG [LABEL]" >&2; exit 2 ;;
esac
