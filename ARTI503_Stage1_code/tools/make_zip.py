import argparse
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "arti503_stage1"))
TOP = "ARTI503_Stage1_9MA2G2"

FILES = [
    ("README.txt", False),
    ("run.py", False),
    ("dedup.py", False),
    ("sysinfo.py", False),
    ("run_windows.bat", False),
    ("run_mac_linux.sh", True),
    ("data/manifest.json", False),
    ("data/corpus.jsonl.gz", False),
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "dist", TOP + ".zip"))
    ap.add_argument("--allow-synthetic", action="store_true", help="testing only")
    args = ap.parse_args()

    for rel, _ in FILES:
        if not os.path.isfile(os.path.join(SRC, rel)):
            sys.exit("Missing {}. Run prepare_data.py first.".format(rel))

    with open(os.path.join(SRC, "data", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("synthetic") and not args.allow_synthetic:
        sys.exit("The data is SYNTHETIC test data. Run prepare_data.py without --synthetic first.")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with zipfile.ZipFile(tmp, "w") as z:
        for rel, executable in FILES:
            path = os.path.join(SRC, rel)
            with open(path, "rb") as f:
                data = f.read()
            if rel.endswith(".bat"):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            elif rel.endswith((".py", ".sh", ".txt")):
                data = data.replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(TOP + "/" + rel, date_time=(2026, 10, 16, 12, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if executable else 0o100644) << 16
            info.compress_type = zipfile.ZIP_STORED if rel.endswith(".gz") else zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    os.replace(tmp, out)

    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad:
            sys.exit("Zip check failed on " + bad)
        names = set(z.namelist())
        expected = {TOP + "/" + rel for rel, _ in FILES}
        if names != expected:
            sys.exit("Zip contents are wrong: {}".format(sorted(names ^ expected)))
        corpus_hash = sha256_bytes(z.read(TOP + "/data/corpus.jsonl.gz"))
        if corpus_hash != manifest["corpus_sha256"]:
            sys.exit("Corpus inside the zip does not match the manifest checksum.")

    print("Created {} ({:.0f} MB)".format(out, os.path.getsize(out) / 1024 ** 2))
    print("Checked: all files present, data checksum OK.")


if __name__ == "__main__":
    main()
