import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import csv
import datetime
import io
import json
import multiprocessing as mp
import shutil
import statistics
import subprocess
import sys
import time

try:
    import numpy as np
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dedup
import par
import sysinfo

OUT = os.path.join(HERE, "results")
EXE = os.path.join(HERE, "omp.exe" if os.name == "nt" else "omp")
CHECK_KEYS = ("total", "collisions", "hist_sum", "sig_sum")


def say(text=""):
    print(text, flush=True)


def find_data(given):
    for path in ([given] if given else []) + [
            os.path.join(HERE, "..", "arti503_stage1", "data", "corpus.jsonl.gz"),
            os.path.join(HERE, "data", "corpus.jsonl.gz")]:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    sys.exit("Cannot find corpus.jsonl.gz. Put this folder next to arti503_stage1 or pass --data PATH")


def find_gpp():
    found = shutil.which("g++")
    if found or os.name != "nt":
        return found
    import glob
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
                           "BrechtSanders.WinLibs*", "mingw64", "bin", "g++.exe")
    hits = glob.glob(pattern)
    return hits[0] if hits else None


def build_exe():
    src = os.path.join(HERE, "omp.cpp")
    if os.path.isfile(EXE) and os.path.getmtime(EXE) >= os.path.getmtime(src):
        return
    gpp = find_gpp()
    if not gpp and os.name == "nt":
        say("Installing the C++ compiler (one time, a few minutes) ...")
        subprocess.run(["winget", "install", "-e", "--id", "BrechtSanders.WinLibs.POSIX.UCRT",
                        "--accept-package-agreements", "--accept-source-agreements"])
        gpp = find_gpp()
    if not gpp:
        sys.exit("No C++ compiler found. Install it with: winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT")
    os.environ["PATH"] = os.path.dirname(gpp) + os.pathsep + os.environ["PATH"]
    say("Compiling omp.cpp ...")
    cmd = [gpp, "-O2", "-fopenmp", "-o", EXE, src]
    if os.name == "nt" and subprocess.run(cmd[:-1] + ["-static", src]).returncode == 0:
        return
    subprocess.run(cmd, check=True)


def export_inputs(path, num):
    shin = os.path.join(HERE, "work", "shingles_{}.bin".format(num))
    coef = os.path.join(HERE, "work", "coef.bin")
    os.makedirs(os.path.dirname(shin), exist_ok=True)
    hasher = dedup.MinHasher()
    with open(coef, "wb") as fout:
        fout.write(hasher.a.astype("<u8").tobytes())
        fout.write(hasher.b.astype("<u8").tobytes())
    if os.path.isfile(shin):
        return shin, coef
    say("Preparing shingles for {:,} documents (one time) ...".format(num))
    docs = par.load_docs(path, num)
    arrs = [dedup.shingles(dedup.normalize(doc)) for doc in docs]
    offs = np.zeros(len(arrs) + 1, dtype="<u8")
    offs[1:] = np.cumsum([arr.size for arr in arrs])
    tmp = shin + ".tmp"
    with open(tmp, "wb") as fout:
        fout.write(np.array([len(arrs)], dtype="<u8").tobytes())
        fout.write(offs.tobytes())
        for arr in arrs:
            fout.write(arr.astype("<u4").tobytes())
    os.replace(tmp, shin)
    return shin, coef


def run_cpp(shin, coef, num, threads, repeats, modes="01234"):
    cmd = [EXE, shin, coef, str(num), ",".join(map(str, threads)), str(repeats), modes]
    rows = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        head = None
        for line in proc.stdout:
            line = line.strip()
            if not head:
                head = line.split(",")
                continue
            row = dict(zip(head, line.split(",")))
            row = {key: (val if key == "version" else float(val) if key == "seconds" else int(val))
                   for key, val in row.items()}
            rows.append(row)
            say("  C++ {:<10} {:>2} threads  rep {}  {:.3f} s".format(row["version"], row["threads"], row["rep"], row["seconds"]))
    if proc.returncode:
        sys.exit("omp.exe failed")
    return rows


def summarize(rows, base_key):
    groups = {}
    for row in rows:
        groups.setdefault((row["version"], row["threads"]), []).append(row)
    seq = groups[base_key]
    ref = {key: seq[0][key] for key in CHECK_KEYS}
    base = statistics.mean(row["seconds"] for row in seq)
    table = []
    for (version, threads), runs in groups.items():
        secs = [row["seconds"] for row in runs]
        bad = [row for row in runs if any(row[key] != ref[key] for key in CHECK_KEYS)]
        table.append({
            "version": version, "threads": threads, "repeats": len(runs),
            "mean": statistics.mean(secs), "std": statistics.stdev(secs) if len(secs) > 1 else 0.0,
            "speedup": base / statistics.mean(secs),
            "correct_runs": len(runs) - len(bad),
            "total_err": [row["total"] - ref["total"] for row in runs],
            "coll_err": [row["collisions"] - ref["collisions"] for row in runs],
            "hist_ok": [row["hist_sum"] == ref["hist_sum"] for row in runs],
        })
    return ref, table


def py_runs(path, num, procs_list, repeats):
    rows = []
    for rep in range(repeats):
        res = par.sequential(par.load_docs(path, num))
        res.update(version="sequential", threads=1, rep=rep + 1)
        rows.append(res)
        say("  Py  sequential  1 proc   rep {}  {:.3f} s".format(rep + 1, res["seconds"]))
    for version in ("race", "lock", "atomic", "reduction"):
        for procs in procs_list:
            for rep in range(repeats):
                res = par.parallel(path, num, version, procs)
                res.update(version=version, threads=procs, rep=rep + 1)
                rows.append(res)
                say("  Py  {:<10} {:>2} procs  rep {}  {:.3f} s".format(version, procs, rep + 1, res["seconds"]))
    return rows


def write_text(result, path):
    buf = io.StringIO()
    for lang in ("cpp", "python"):
        part = result[lang]
        buf.write("\n{} ({:,} documents)\n".format("C++ OpenMP" if lang == "cpp" else "Python multiprocessing", part["docs"]))
        buf.write("{:<11} {:>7} {:>9} {:>8} {:>8} {:>8}  {}\n".format("version", "threads", "mean_s", "std_s", "speedup", "correct", "total error / collision error per run"))
        for row in part["table"]:
            buf.write("{:<11} {:>7} {:>9.3f} {:>8.3f} {:>8.2f} {:>5}/{:<2}  {} / {}\n".format(
                row["version"], row["threads"], row["mean"], row["std"], row["speedup"],
                row["correct_runs"], row["repeats"], row["total_err"], row["coll_err"]))
    buf.write("\nCross-check C++ vs Python sequential on {:,} docs: {}\n".format(result["python"]["docs"], result["cross_check"]))
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(buf.getvalue())
    say(buf.getvalue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--data")
    ap.add_argument("--cpp-docs", type=int, default=100000)
    ap.add_argument("--py-docs", type=int, default=10000)
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--cpp-repeats", type=int, default=5)
    ap.add_argument("--py-repeats", type=int, default=3)
    ap.add_argument("--skip-python", action="store_true")
    args = ap.parse_args()

    name = "".join(ch if ch.isalnum() else "_" for ch in (args.name or input("Your name (English): "))).strip("_") or "member"
    threads = [int(val) for val in args.threads.split(",")]
    path = find_data(args.data)
    build_exe()
    system = sysinfo.collect()
    say("CPU: {} | {} cores / {} threads".format(system["cpu_model"], system["physical_cores"], system["logical_cores"]))

    started = time.perf_counter()
    shin, coef = export_inputs(path, args.cpp_docs)
    say("\nC++ OpenMP on {:,} documents".format(args.cpp_docs))
    cpp_rows = run_cpp(shin, coef, args.cpp_docs, threads, args.cpp_repeats)
    cpp_ref, cpp_table = summarize(cpp_rows, ("sequential", 1))

    result = {
        "member": name, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": system,
        "cpp": {"docs": args.cpp_docs, "reference": cpp_ref, "table": cpp_table, "runs": cpp_rows},
        "python": {"docs": args.py_docs, "table": [], "runs": []},
        "cross_check": "skipped",
    }

    if not args.skip_python:
        say("\nPython multiprocessing on {:,} documents".format(args.py_docs))
        py_rows = py_runs(path, args.py_docs, threads, args.py_repeats)
        py_ref, py_table = summarize(py_rows, ("sequential", 1))
        result["python"].update(reference=py_ref, table=py_table, runs=py_rows)
        small, _ = export_inputs(path, args.py_docs)
        check = run_cpp(small, coef, args.py_docs, [1], 1, "0")[0]
        same = all(check[key] == py_ref[key] for key in CHECK_KEYS)
        result["cross_check"] = "identical" if same else "DIFFERENT {} vs {}".format(
            {key: check[key] for key in CHECK_KEYS}, py_ref)

    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "A2_{}.json".format(name)), "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=1)
    write_text(result, os.path.join(OUT, "A2_{}.txt".format(name)))
    say("Done in {:.1f} min. Send results/A2_{}.json".format(result["elapsed_s"] / 60, name))


if __name__ == "__main__":
    mp.freeze_support()
    main()
