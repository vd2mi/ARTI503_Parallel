import argparse
import datetime
import gzip
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "arti503_stage1", "data")

DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "arb_Arab"
TIERS = [1000, 10000, 50000, 100000]
DUP_RATE = 0.10
MIN_WORDS, MAX_WORDS = 50, 10000
MIN_LANG_SCORE = 0.8

ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
HARAKAT = ["َ", "ُ", "ِ", "ّ", "ْ", "ً"]
TATWEEL = "ـ"
PUNCT = ["،", ".", "؟", "!", ":", "؛", "(", ")", "«", "»", "-"]


def stream_fineweb(count, log):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("The 'datasets' library is missing. Run:  pip install datasets")
    log("Connecting to Hugging Face ({}, subset {})...".format(DATASET, CONFIG))
    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    seen = kept = 0
    for row in ds:
        seen += 1
        text = row.get("text") or ""
        score = row.get("language_score")
        if score is not None and score < MIN_LANG_SCORE:
            continue
        n_words = len(text.split())
        if not (MIN_WORDS <= n_words <= MAX_WORDS):
            continue
        kept += 1
        yield {"text": text, "id": row.get("id"), "url": row.get("url"), "dump": row.get("dump")}
        if kept % 5000 == 0:
            log("  {:,} / {:,} documents kept ({:,} read)".format(kept, count, seen))
        if kept >= count:
            break
    if kept < count:
        sys.exit("Only {:,} usable documents found, needed {:,}.".format(kept, count))


def synthetic_docs(count, rng):
    vocab = []
    for _ in range(40000):
        n = rng.randint(2, 8)
        vocab.append("".join(rng.choice(ARABIC_LETTERS) for _ in range(n)))
    weights = [1.0 / (r + 1) for r in range(len(vocab))]
    cum = []
    total = 0.0
    for w in weights:
        total += w
        cum.append(total)
    for i in range(count):
        n_words = int(min(MAX_WORDS, max(MIN_WORDS, rng.lognormvariate(6.0, 0.8))))
        words = rng.choices(vocab, cum_weights=cum, k=n_words)
        out = []
        for j, w in enumerate(words):
            if rng.random() < 0.03:
                w = w[0] + rng.choice(HARAKAT) + w[1:]
            out.append(w)
            if rng.random() < 0.06:
                out.append(rng.choice(PUNCT))
            if rng.random() < 0.01:
                out.append("\n")
        yield {"text": " ".join(out), "id": "synthetic-{}".format(i), "url": None, "dump": None}


def surface_noise(word, rng):
    r = rng.random()
    if r < 0.35 and len(word) > 1:
        pos = rng.randint(1, len(word))
        return word[:pos] + rng.choice(HARAKAT) + word[pos:]
    if r < 0.55 and word.startswith("ا"):
        return rng.choice(["أ", "إ", "آ"]) + word[1:]
    if r < 0.75 and len(word) > 2:
        pos = rng.randint(1, len(word) - 1)
        return word[:pos] + TATWEEL + word[pos:]
    return word + rng.choice(PUNCT)


def make_duplicate(text, rng, vocab):
    words = text.split()
    rate = rng.uniform(0.01, 0.05)
    edits = max(1, round(rate * len(words)))
    for _ in range(edits):
        op = rng.random()
        pos = rng.randrange(len(words))
        if op < 0.4:
            words[pos] = rng.choice(vocab)
        elif op < 0.7 and len(words) > MIN_WORDS:
            del words[pos]
        else:
            words.insert(pos, rng.choice(vocab))
    for _ in range(max(1, len(words) // 30)):
        pos = rng.randrange(len(words))
        words[pos] = surface_noise(words[pos], rng)
    sep = "\n" if rng.random() < 0.5 else " "
    return sep.join(words), rate


def build(originals, tiers, dup_rate, rng, log):
    vocab = []
    for doc in originals[:2000]:
        vocab.extend(doc["text"].split()[:200])
    vocab = [w for w in vocab if any("ء" <= c <= "ي" for c in w)] or ["كلمة"]

    corpus, groups = [], {}
    used = 0
    prev = 0
    for tier in tiers:
        seg_size = tier - prev
        n_dup = round(seg_size * dup_rate)
        n_orig = seg_size - n_dup
        seg_orig = originals[used:used + n_orig]
        used += n_orig
        items = [(dict(d, dup_of=None), None) for d in seg_orig]
        for _ in range(n_dup):
            src_local = rng.randrange(n_orig)
            text, rate = make_duplicate(seg_orig[src_local]["text"], rng, vocab)
            items.append(({"text": text, "id": None, "url": None, "dump": None,
                           "edit_rate": round(rate, 4)}, src_local))
        order = list(range(len(items)))
        rng.shuffle(order)
        pos_of_local_orig = {}
        new_positions = {}
        for new_pos, k in enumerate(order):
            new_positions[k] = prev + new_pos
            if k < n_orig:
                pos_of_local_orig[k] = prev + new_pos
        seg_records = [None] * len(items)
        for k, (rec, src_local) in enumerate(items):
            gpos = new_positions[k]
            if src_local is not None:
                src_g = pos_of_local_orig[src_local]
                rec["dup_of"] = src_g
                groups.setdefault(src_g, [src_g]).append(gpos)
            seg_records[gpos - prev] = rec
        corpus.extend(seg_records)
        log("  tier {:>7,}: {:,} originals + {:,} near-duplicates".format(tier, n_orig, n_dup))
        prev = tier
    group_list = sorted(sorted(g) for g in groups.values())
    return corpus, group_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tiers", default=",".join(str(t) for t in TIERS))
    ap.add_argument("--dup-rate", type=float, default=DUP_RATE)
    ap.add_argument("--seed", type=int, default=503)
    ap.add_argument("--synthetic", action="store_true", help="fake text, no internet (testing only)")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    tiers = sorted({int(t) for t in args.tiers.split(",") if t.strip()})
    if not tiers or tiers[0] <= 0:
        sys.exit("--tiers must be positive numbers")
    if not (0 < args.dup_rate < 0.5):
        sys.exit("--dup-rate must be between 0 and 0.5")
    rng = random.Random(args.seed)

    needed, prev = 0, 0
    for t in tiers:
        seg = t - prev
        needed += seg - round(seg * args.dup_rate)
        prev = t

    log("Collecting {:,} original documents...".format(needed))
    if args.synthetic:
        originals = list(synthetic_docs(needed, rng))
        source = "SYNTHETIC test data (not real documents)"
    else:
        originals = list(stream_fineweb(needed, log))
        source = "FineWeb-2 ({}, {}), Common Crawl derived, ODC-By".format(DATASET, CONFIG)

    log("Adding near-duplicates...")
    corpus, groups = build(originals, tiers, args.dup_rate, rng, log)

    os.makedirs(args.out, exist_ok=True)
    corpus_path = os.path.join(args.out, "corpus.jsonl.gz")
    tmp = corpus_path + ".tmp"
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz:
            for rec in corpus:
                gz.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, corpus_path)

    h = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)

    words = sorted(len(r["text"].split()) for r in corpus)
    manifest = {
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "synthetic": bool(args.synthetic),
        "n_docs": len(corpus),
        "tiers": tiers,
        "dup_rate": args.dup_rate,
        "seed": args.seed,
        "filters": {"min_words": MIN_WORDS, "max_words": MAX_WORDS, "min_language_score": MIN_LANG_SCORE},
        "words_per_doc": {"min": words[0], "median": words[len(words) // 2],
                          "mean": round(sum(words) / len(words), 1), "max": words[-1]},
        "corpus_sha256": h.hexdigest(),
        "groups": groups,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    size_mb = os.path.getsize(corpus_path) / 1024 ** 2
    log("")
    log("Done: {:,} documents ({:.0f} MB compressed) in {}".format(len(corpus), size_mb, os.path.abspath(args.out)))
    log("Words per document: median {:,}, mean {:,.0f}, max {:,}".format(
        manifest["words_per_doc"]["median"], manifest["words_per_doc"]["mean"], manifest["words_per_doc"]["max"]))
    log("Near-duplicate groups: {:,}".format(len(groups)))
    log("Next: run the benchmark once yourself (python run.py), then python make_zip.py")


if __name__ == "__main__":
    main()
