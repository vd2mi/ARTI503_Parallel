import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_var] = "1"

import sys

if sys.version_info < (3, 8):
    print("This needs Python 3.8 or newer. You have Python {}.{}.".format(*sys.version_info[:2]))
    print("Install Python from https://www.python.org/downloads/ and try again.")
    sys.exit(1)

import argparse
import cProfile
import datetime
import gc
import gzip
import hashlib
import io
import json
import math
import platform
import pstats
import re
import subprocess
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
CORPUS = os.path.join(DATA_DIR, "corpus.jsonl.gz")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")

SCHEMA = 1
DEFAULT_PLAN = [(1000, 5), (10000, 5), (50000, 5), (100000, 3)]
PROFILE_SIZE = 10000


def say(text=""):
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        sys.stdout.write(str(text).encode(enc, errors="replace").decode(enc) + "\n")
    except Exception:
        sys.stdout.write(str(text).encode("ascii", errors="replace").decode("ascii") + "\n")
    sys.stdout.flush()


def fail(message):
    say()
    say("ERROR: " + message)
    sys.exit(1)


def fmt_secs(s):
    if s < 1:
        return "{:.0f} ms".format(s * 1000)
    if s < 120:
        return "{:.1f} s".format(s)
    return "{:.1f} min".format(s / 60)


def ensure_numpy():
    try:
        import numpy
        return
    except ImportError:
        pass
    say("NumPy is not installed. Installing it now (one time only)...")
    for extra in ([], ["--user"]):
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "numpy"] + extra
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            continue
        try:
            import importlib
            import site
            importlib.invalidate_caches()
            if extra:
                user_site = site.getusersitepackages()
                if user_site not in sys.path:
                    sys.path.append(user_site)
            import numpy
            say("NumPy installed.")
            return
        except ImportError:
            continue
    fail("Could not install NumPy automatically.\n"
         "Open a terminal and run:   {} -m pip install numpy\n"
         "then run this program again.".format(sys.executable))


def disable_quick_edit():
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetStdHandle.restype = wintypes.HANDLE
        k32.GetStdHandle.argtypes = [wintypes.DWORD]
        handle = k32.GetStdHandle(wintypes.DWORD(-10 & 0xFFFFFFFF))
        mode = wintypes.DWORD()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x0040, 0x0080
            k32.SetConsoleMode(handle, (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE)
    except Exception:
        pass


def peak_memory_mb():
    try:
        if platform.system() == "Windows":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PMC()
            counters.cb = ctypes.sizeof(PMC)
            get_proc = ctypes.windll.kernel32.GetCurrentProcess
            get_proc.restype = wintypes.HANDLE
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            get_info.restype = wintypes.BOOL
            if get_info(get_proc(), ctypes.byref(counters), counters.cb):
                return round(counters.PeakWorkingSetSize / 1024 ** 2)
            return None
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / 1024 ** 2) if platform.system() == "Darwin" else round(peak / 1024)
    except Exception:
        return None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_data():
    prep = os.path.join(HERE, "..", "tools", "prepare_data.py")
    if (not os.path.isfile(CORPUS) or not os.path.isfile(MANIFEST)) and os.path.isfile(prep):
        say("No data yet. Building it now (one time only, needs internet)...")
        try:
            import datasets
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "datasets"])
        subprocess.run([sys.executable, prep, "--out", DATA_DIR])
    if not os.path.isfile(CORPUS) or not os.path.isfile(MANIFEST):
        fail("The data folder is missing or incomplete.\n"
             "Make sure you unzipped the WHOLE zip file and that the folder 'data'\n"
             "(with corpus.jsonl.gz and manifest.json) is next to run.py.")
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    say("Checking the data file...")
    digest = sha256_of(CORPUS)
    if digest != manifest["corpus_sha256"]:
        fail("The data file is damaged (checksum does not match).\n"
             "Download the zip again and unzip it fresh.")
    docs = []
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    if len(docs) != manifest["n_docs"]:
        fail("The data file has {} documents but should have {}.".format(len(docs), manifest["n_docs"]))
    return docs, manifest


def mean_std(values):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def clean_name(raw):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", (raw or "").strip()).strip("_")
    return name[:40]


def write_results(result, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "RESULTS_{}.json".format(name))
    txt_path = os.path.join(RESULTS_DIR, "RESULTS_{}.txt".format(name))
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, json_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_text(result))
    return json_path, txt_path


def render_text(r):
    from dedup import STAGES
    out = io.StringIO()
    w = out.write
    w("ARTI 503 Project Stage 1 - sequential benchmark (group 9MA2G2)\n")
    w("Member: {}\nDate:   {}\n".format(r["member"], r["timestamp"]))
    w("Status: {}\n\n".format("COMPLETE" if r["complete"] else "INCOMPLETE (" + r.get("stopped_reason", "") + ")"))
    w("System\n")
    for k, v in r["system"].items():
        w("  {:<18} {}\n".format(k, v))
    w("  {:<18} {}\n\n".format("peak_memory_mb", r.get("peak_memory_mb")))

    if r["sizes"]:
        w("Total time per run (seconds)\n")
        w("  {:>8}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}\n".format("docs", "repeats", "mean", "std", "min", "max"))
        for s in r["sizes"]:
            w("  {:>8,}  {:>7}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>9.3f}\n".format(
                s["docs"], s["repeats"], s["total_mean"], s["total_std"], s["total_min"], s["total_max"]))
        w("\nMean time per stage (seconds) and share of total\n")
        header = "  {:>8}".format("docs") + "".join("  {:>11}".format(st) for st in STAGES)
        w(header + "\n")
        for s in r["sizes"]:
            row = "  {:>8,}".format(s["docs"])
            for st in STAGES:
                row += "  {:>11}".format("{:.3f} {:>3.0f}%".format(s["stage_mean"][st], s["stage_share_pct"][st]))
            w(row + "\n")
        w("\nResult check (identical on every machine)\n")
        w("  {:>8}  {:>10}  {:>10}  {:>9}  {:>10}  {:>9}  {:>9}  {:>12}  {:>12}\n".format(
            "docs", "candidates", "dup_pairs", "clusters", "max_bucket", "recall", "precision",
            "sig_checksum", "edge_checksum"))
        for s in r["sizes"]:
            st, acc = s["stats"], s["accuracy"]
            w("  {:>8,}  {:>10,}  {:>10,}  {:>9,}  {:>10,}  {:>8.1f}%  {:>8.1f}%  {:>12}  {:>12}\n".format(
                s["docs"], st["candidates"], st["verified_pairs"], st["clusters"], st["max_bucket"],
                acc["recall"] * 100, acc["precision"] * 100, st["sig_checksum"], st["edge_checksum"]))
    if r.get("profile_text"):
        w("\ncProfile of one run on {:,} documents (top functions by own time)\n".format(r["profile_docs"]))
        w(r["profile_text"])
    return out.getvalue()


def profile_run(docs, hasher, run_pipeline):
    prof = cProfile.Profile()
    gc.collect()
    prof.enable()
    run_pipeline(docs, hasher)
    prof.disable()
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("tottime").print_stats(15)
    text = stream.getvalue()
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "ncalls" in l), 0)
    table = "\n".join(lines[start:]).strip() + "\n"
    table = table.replace(HERE + os.sep, "").replace(os.path.dirname(os.__file__) + os.sep, "")
    return table


def main():
    parser = argparse.ArgumentParser(description="ARTI 503 Stage 1 sequential benchmark")
    parser.add_argument("--name", help="your name (used in the result file name)")
    parser.add_argument("--sizes", help="comma-separated sizes, e.g. 1000,10000 (default: 1000,10000,50000,100000)")
    parser.add_argument("--repeats", type=int, help="repeats per size (default: 5, and 3 for 100,000)")
    parser.add_argument("--no-profile", action="store_true", help="skip the cProfile run")
    args = parser.parse_args()

    say("=" * 64)
    say(" ARTI 503 Stage 1 - sequential MinHash/LSH benchmark (9MA2G2)")
    say("=" * 64)

    name = clean_name(args.name)
    while not name:
        try:
            name = clean_name(input("Type your first and last name in English, then press Enter: "))
        except EOFError:
            fail("No name given. Run again with:  python run.py --name YourName")
        if not name:
            say("Please use English letters only.")

    ensure_numpy()
    sys.path.insert(0, HERE)
    import numpy as np
    import dedup
    import sysinfo

    plan = list(DEFAULT_PLAN)
    if args.sizes:
        try:
            sizes = [int(x.replace("_", "").strip()) for x in args.sizes.split(",") if x.strip()]
        except ValueError:
            fail("--sizes must be numbers separated by commas, e.g. 1000,10000")
        plan = [(n, 5 if n < 100000 else 3) for n in sizes]
    if args.repeats:
        plan = [(n, max(1, args.repeats)) for n, _ in plan]

    docs, manifest = load_data()
    plan = [(min(n, len(docs)), r) for n, r in plan if n > 0]
    say("Data: {:,} documents, {}".format(len(docs), manifest.get("source")))
    if manifest.get("synthetic"):
        say("  WARNING: this is SYNTHETIC test data, not the real data set.")

    say("Reading system details...")
    system = sysinfo.collect()
    say("  CPU:    {}".format(system["cpu_model"]))
    say("  Cores:  {} physical / {} logical".format(system["physical_cores"], system["logical_cores"]))
    say("  RAM:    {} GB".format(system["ram_gb"]))
    say("  Python: {} | NumPy {}".format(system["python"], system["numpy"]))
    if system.get("on_ac_power") is False:
        say("  WARNING: you are on battery. Plug in the charger for stable timings.")
    say()
    say("Keep the laptop plugged in and close other programs until this finishes.")
    say()
    disable_quick_edit()

    result = {
        "schema": SCHEMA,
        "member": name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": system,
        "settings": {
            "shingle_size": dedup.SHINGLE_SIZE, "num_perm": dedup.NUM_PERM,
            "bands": dedup.BANDS, "rows": dedup.ROWS,
            "jaccard_threshold": dedup.JACCARD_THRESHOLD, "prime": dedup.PRIME,
            "seed": dedup.SEED, "gc_disabled_during_runs": True, "threads": 1,
        },
        "data": {
            "n_docs": manifest["n_docs"], "corpus_sha256": manifest["corpus_sha256"],
            "source": manifest.get("source"),
        },
        "plan": [{"docs": n, "repeats": r} for n, r in plan],
        "sizes": [],
        "complete": False,
        "peak_memory_mb": None,
    }

    hasher = dedup.MinHasher()
    groups = manifest["groups"]
    started = time.perf_counter()

    try:
        say("Warm-up run (not recorded)...")
        dedup.run_pipeline(docs[:min(1000, len(docs))], hasher)

        per_doc = None
        for idx, (n, repeats) in enumerate(plan):
            if per_doc is not None:
                remaining = sum(m * r for m, r in plan[idx:]) * per_doc
                say("Estimated time left: about {}".format(fmt_secs(remaining)))
            subset = docs[:n]
            runs = []
            reference = None
            for rep in range(repeats):
                gc.collect()
                gc.disable()
                try:
                    res = dedup.run_pipeline(subset, hasher)
                finally:
                    gc.enable()
                stats = res["stats"]
                if reference is None:
                    reference = stats
                    clusters = res["clusters"]
                elif stats != reference:
                    raise RuntimeError("Repeated runs gave different results on {} docs".format(n))
                runs.append({"times": {k: round(v, 6) for k, v in res["times"].items()},
                             "total": round(res["total"], 6), "wall": round(res["wall"], 6)})
                say("  {:>7,} docs  run {}/{}  {}".format(n, rep + 1, repeats, fmt_secs(res["total"])))
                del res

            totals = [x["total"] for x in runs]
            tm, ts = mean_std(totals)
            stage_mean = {st: mean_std([x["times"][st] for x in runs])[0] for st in dedup.STAGES}
            stage_std = {st: mean_std([x["times"][st] for x in runs])[1] for st in dedup.STAGES}
            acc = dedup.pair_scores(clusters, groups, n)
            del clusters
            result["sizes"].append({
                "docs": n, "repeats": repeats, "runs": runs,
                "total_mean": round(tm, 6), "total_std": round(ts, 6),
                "total_min": round(min(totals), 6), "total_max": round(max(totals), 6),
                "stage_mean": {k: round(v, 6) for k, v in stage_mean.items()},
                "stage_std": {k: round(v, 6) for k, v in stage_std.items()},
                "stage_share_pct": {k: round(100 * v / tm, 2) if tm else 0.0 for k, v in stage_mean.items()},
                "stats": reference,
                "accuracy": acc,
            })
            say("  {:>7,} docs  mean {} (std {})  recall {:.1f}%  precision {:.1f}%".format(
                n, fmt_secs(tm), fmt_secs(ts), acc["recall"] * 100, acc["precision"] * 100))
            say()
            per_doc = tm / n
            result["peak_memory_mb"] = peak_memory_mb()
            write_results(result, name)

        if not args.no_profile:
            p = min(PROFILE_SIZE, len(docs))
            say("Profiling one run on {:,} documents...".format(p))
            result["profile_docs"] = p
            result["profile_text"] = profile_run(docs[:p], hasher, dedup.run_pipeline)

        result["complete"] = True
    except KeyboardInterrupt:
        result["stopped_reason"] = "stopped by user"
        say()
        say("Stopped. The sizes that finished are saved.")
    except MemoryError:
        result["stopped_reason"] = "out of memory"
        say()
        say("Your computer ran out of memory. The sizes that finished are saved.")
    except Exception as exc:
        result["stopped_reason"] = "error: {}".format(exc)
        result["traceback"] = traceback.format_exc()
        say()
        say("Something went wrong. The sizes that finished are saved, and the error")
        say("details are inside the results file. Send it to the team anyway.")

    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    result["peak_memory_mb"] = peak_memory_mb()
    json_path, txt_path = write_results(result, name)

    say("=" * 64)
    say(" DONE in {}".format(fmt_secs(result["elapsed_s"])) if result["complete"] else " FINISHED WITH PROBLEMS")
    say(" Send this file to the group:")
    say("   " + json_path)
    say(" (A readable copy is next to it: " + os.path.basename(txt_path) + ")")
    say("=" * 64)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
