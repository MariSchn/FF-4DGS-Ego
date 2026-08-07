"""Upload the two irreplaceable checkpoints to a PRIVATE HF repo, then verify by hash.

WHY: /home/dmonopoli is at its quota and these two .pt files are 9.6 GB of it. More importantly
jitterrob10ep_best.pt exists ONLY there - it is the single copy backing every headline number in
the paper, and it is not on scratch. This makes it durable first and reclaims space second.

SAFETY:
  * repo is created PRIVATE. These are unpublished research checkpoints for a paper under review.
  * the token is read from STDIN, never written to disk, never passed as an argv.
  * NOTHING is deleted here. Deletion happens only after the printed hashes are compared.

VERIFICATION: for LFS-tracked files HF stores the sha256 of the content as the object oid, so
comparing our locally computed sha256 against the server's oid is a genuine end-to-end integrity
check and does not require re-downloading 9.6 GB.
"""
from __future__ import annotations

import hashlib
import os
import sys

from huggingface_hub import HfApi

REPO = "worldhand4dgs-checkpoints"
FILES = [
    "/home/dmonopoli/ckpt_backup/jitterrob10ep_best.pt",
    "/home/dmonopoli/ckpt_backup/winner10ep_best.pt",
]


def sha256(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    # HF_TOKEN env first (the SLURM path passes it via --export so it never lands on disk),
    # falling back to stdin for interactive use. Never read from or written to a file.
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and not sys.stdin.isatty():
        token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit("no token: set HF_TOKEN or pipe it on stdin")

    api = HfApi(token=token)
    me = api.whoami()
    user = me.get("name")
    print(f"authenticated as {user} ({me.get('type')})", flush=True)

    repo_id = f"{user}/{REPO}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"repo {repo_id} private={info.private}", flush=True)
    if not info.private:
        raise SystemExit("REFUSING to upload: repo is PUBLIC. These are unpublished checkpoints.")

    local = {}
    for p in FILES:
        if not os.path.exists(p):
            print(f"SKIP missing {p}", flush=True)
            continue
        gb = os.path.getsize(p) / 1024 ** 3
        print(f"hashing {os.path.basename(p)} ({gb:.2f} GB)...", flush=True)
        local[os.path.basename(p)] = sha256(p)
        print(f"  local sha256 {local[os.path.basename(p)]}", flush=True)

    for p in FILES:
        if not os.path.exists(p):
            continue
        name = os.path.basename(p)
        print(f"uploading {name} ...", flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=name, repo_id=repo_id,
                        repo_type="model")
        print(f"  uploaded {name}", flush=True)

    print("\n=================== VERIFY ===================", flush=True)
    files = api.list_repo_tree(repo_id=repo_id, repo_type="model", expand=True)
    remote = {}
    for f in files:
        oid = getattr(getattr(f, "lfs", None), "sha256", None) or getattr(getattr(f, "lfs", None), "oid", None)
        if oid:
            remote[f.path] = oid
    ok = True
    for name, lsha in local.items():
        rsha = remote.get(name)
        match = (rsha == lsha)
        ok &= match
        print(f"{name}\n  local  {lsha}\n  remote {rsha}\n  MATCH: {match}", flush=True)
    print(f"\nALL_HASHES_MATCH: {ok}", flush=True)
    print("Safe to delete the local copies ONLY if the line above says True.", flush=True)


if __name__ == "__main__":
    main()
