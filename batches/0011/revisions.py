# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# What the Hugging Face Hub says identifies a file, and whether it holds: the Hub's own digest of a file against
# the digest of the downloaded bytes; the files of a repository at every one of its commits, by the id the Hub
# gives (the sha256 of an LFS file, the git blob id of any other); and the repositories that name a model whose
# model.safetensors has the same sha256.
# Usage: revisions.py check REPO FILE LOCAL             the Hub's id of FILE at the head of REPO against LOCAL
#        revisions.py commits REPO OUT.tsv              one row per commit of the main branch, one column per file
#        revisions.py siblings REPO SEARCH OUT.tsv      repositories found by SEARCH, and whether their weights are REPO's
#        revisions.py content SIBLINGS.tsv REF.json WORKDIR OUT.tsv
#                                                       of those with a model.safetensors: its tensor table (names, dtypes,
#                                                       shapes) from the first bytes of the file against REF's, and, when the
#                                                       names and shapes are REF's, the tensors themselves against REF's
import hashlib, json, os, struct, sys, urllib.request
from huggingface_hub import HfApi, hf_hub_download

FILES = ["model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja", "tokenizer.model"]


def file_id(info):
    if info is None:
        return "-"
    lfs = getattr(info, "lfs", None)
    if lfs:
        return "sha256:" + (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256)
    return "git:" + info.blob_id


def ids(api, repo, paths, revision=None):
    got = {i.path: i for i in api.get_paths_info(repo, paths, expand=True, revision=revision)}
    return {p: file_id(got.get(p)) for p in paths}


def git_blob_id(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def table_of(repo):
    """{tensor name: (dtype, shape)} and the metadata, from the first bytes of the repository's model.safetensors."""
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"

    def get(a, b):
        return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
    n = struct.unpack("<Q", get(0, 7))[0]
    h = json.loads(get(8, 7 + n))
    meta = h.pop("__metadata__", None)
    return {k: (v["dtype"], v["shape"]) for k, v in h.items()}, meta


def content(siblings, ref, work, out):
    import identity
    refrec = json.load(open(ref))
    reft = {e["labels"]["safetensors"]: (e["stored_dtype"], e["shape"]) for e in refrec["subject"]["tensors"]}
    rows = [["repo", "tensors", "same_names", "same_shapes", "same_dtypes", "metadata", "keys_in_both", "bag_root_equal"]]
    done = 0
    for line in open(siblings, encoding="utf-8").read().splitlines()[1:]:
        repo, got = line.split("\t")[:2]
        if not got.startswith("sha256:"):
            continue
        try:
            t, meta = table_of(repo)
        except Exception as e:  # a repository whose first bytes cannot be read is a row
            rows.append([repo, "-", "-", "-", "-", "error:" + type(e).__name__, "-", "-"])
            continue
        names = set(t) == set(reft)
        shapes = names and all(t[k][1] == reft[k][1] for k in t)
        dtypes = shapes and all(t[k][0].lower() == reft[k][0] for k in t)
        row = [repo, str(len(t)), str(names), str(shapes), str(dtypes), json.dumps(meta, sort_keys=True), "-", "-"]
        if shapes and done < 8:
            done += 1
            path = hf_hub_download(repo, "model.safetensors", local_dir=f"{work}/{repo.replace('/', '--')}")
            rec = identity.record("safetensors", identity.read_st(path)[0], repo)
            _, summary = identity.match(refrec, rec)
            row[6], row[7] = f"{summary['keys_in_both']}/{summary['ref_distinct_keys']}", str(summary["bag_roots_equal"])
            os.remove(path)
        rows.append(row)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(rows) - 1} repositories with a model.safetensors; {done} of them read in full")


def main(argv):
    api = HfApi()
    if argv[1] == "check":
        repo, name, local = argv[2:5]
        hub = ids(api, repo, [name])[name]
        mine = ("sha256:" + hashlib.sha256(open(local, "rb").read()).hexdigest() if hub.startswith("sha256:")
                else "git:" + git_blob_id(local))
        print(f"{repo} {name}\n  the Hub says  {hub}\n  the bytes are {mine}\n  {'equal' if hub == mine else 'DIFFERENT'}")
        print("head of main:", api.model_info(repo).sha)
    elif argv[1] == "commits":
        repo, out = argv[2:4]
        rows = []
        for c in api.list_repo_commits(repo):
            rows.append([c.commit_id[:12], c.created_at.strftime("%Y-%m-%d"), c.title[:50].replace("\t", " ")]
                        + [ids(api, repo, FILES, c.commit_id)[f] for f in FILES])
        with open(out, "w", encoding="utf-8") as f:
            f.write("\t".join(["commit", "date", "title"] + FILES) + "\n")
            for r in rows:
                f.write("\t".join(x[:19] if x.startswith("sha256:") else x[:16] for x in r[:3] + r[3:]) + "\n")
        print(f"{repo}: {len(rows)} commits")
        for j, f in enumerate(FILES):
            vals = [r[3 + j] for r in rows]
            print(f"  {f}: {len(set(v for v in vals if v != '-'))} distinct ids, absent in {vals.count('-')} commits")
    elif argv[1] == "siblings":
        repo, search, out = argv[2:5]
        want = ids(api, repo, ["model.safetensors"])["model.safetensors"]
        rows = []
        for m in api.list_models(search=search, limit=40):
            if m.id == repo:
                continue
            try:
                got = ids(api, m.id, ["model.safetensors"])["model.safetensors"]
            except Exception as e:  # a repository that cannot be read is a row, not a failure
                got = "error:" + type(e).__name__
            rows.append([m.id, got, "same" if got == want else "other", ";".join(t for t in (m.tags or []) if t.startswith("base_model:"))])
        with open(out, "w", encoding="utf-8") as f:
            f.write("repo\tmodel.safetensors\tweights\tbase_model tags\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
        print(f"{len(rows)} repositories named by '{search}'; {sum(1 for r in rows if r[2] == 'same')} have the same weights file as {repo}")
    elif argv[1] == "content":
        content(*argv[2:6])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
