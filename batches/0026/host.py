# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runs the module of batches/0026/wasm in wasmtime, the host of its exports (see src/lib.rs): the
# weights file is read into the buffer weights_alloc returns, model_load is called, and for each
# IDS argument (token ids, comma separated) the logits of every position are written as
# little-endian float32 to OUTDIR/N.f32, as batches/0010/logits.py does. The module's linear memory
# is limited to LIMIT_MIB; the module declares no limit of its own. The imports of the module, its
# memory after loading and after each prompt, and the time of each call are printed.
# INPUT_ORDER names the graph's inputs in order, as in batches/0010.
# Usage: INPUT_ORDER=input_ids,attention_mask,position_ids host.py MODULE.wasm WEIGHTS.bpk OUTDIR LIMIT_MIB IDS [IDS ...]
import ctypes, os, sys, threading, time
import wasmtime as w

KIND = {"input_ids": 0, "attention_mask": 1, "position_ids": 2}
PAGE = 65536


def main():
    wasm, bpk, outdir, limit_mib = sys.argv[1:5]
    prompts = sys.argv[5:]
    order = [KIND[x] for x in os.environ["INPUT_ORDER"].split(",")]
    assert len(order) == 3, "the generated forward is called with three inputs"
    cfg = w.Config()
    try:
        cfg.max_wasm_stack = 2 << 20  # at most the async stack size, which this binding cannot raise
    except Exception as e:
        print("max_wasm_stack not set:", e)
    engine = w.Engine(cfg)
    module = w.Module.from_file(engine, wasm)
    print("imports:", [(i.module, i.name) for i in module.imports])
    print("exports:", [e.name for e in module.exports])
    store = w.Store(engine)
    store.set_limits(memory_size=int(limit_mib) << 20)
    inst = w.Linker(engine).instantiate(store, module)
    ex = inst.exports(store)
    mem = ex["memory"]
    base = lambda: ctypes.addressof(mem.data_ptr(store).contents)
    print(f"instantiated, memory {mem.size(store) * PAGE >> 20} MiB")

    size = os.path.getsize(bpk)
    t = time.perf_counter()
    ptr = ex["weights_alloc"](store, size)
    print(f"weights_alloc({size}) -> {ptr}, memory {mem.size(store) * PAGE >> 20} MiB, {time.perf_counter() - t:.2f}s")
    t = time.perf_counter()
    with open(bpk, "rb") as f:
        off = 0
        while off < size:
            n = min(64 << 20, size - off)
            buf = (ctypes.c_char * n).from_address(base() + ptr + off)
            assert f.readinto(buf) == n
            off += n
    print(f"weights written, {time.perf_counter() - t:.2f}s")
    t = time.perf_counter()
    rc = ex["model_load"](store)
    print(f"model_load -> {rc}, memory {mem.size(store) * PAGE >> 20} MiB, {time.perf_counter() - t:.2f}s")

    for k, ids in enumerate(prompts):
        ids = [int(x) for x in ids.split(",")]
        p = ex["ids_alloc"](store, len(ids))
        ctypes.memmove(base() + p, (ctypes.c_int32 * len(ids))(*ids), 4 * len(ids))
        t = time.perf_counter()
        count = ex["forward"](store, *order)
        dt = time.perf_counter() - t
        q = ex["out_ptr"](store)
        data = ctypes.string_at(base() + q, 4 * count)
        open(f"{outdir}/{k}.f32", "wb").write(data)
        print(f"{k}: {len(ids)} tokens, {count} floats in {dt:.2f}s, memory {mem.size(store) * PAGE >> 20} MiB")
    print(f"peak memory (it never shrinks): {mem.size(store) * PAGE >> 20} MiB of {limit_mib}")


err = []


def run():
    try:
        main()
    except BaseException as e:  # a trap is a result: print it and fail
        import traceback

        traceback.print_exc()
        err.append(e)


# wasmtime runs the module on the calling thread's stack, so the thread gets a large one.
threading.stack_size(512 << 20)
t = threading.Thread(target=run)
t.start()
t.join()
sys.exit(1 if err else 0)
