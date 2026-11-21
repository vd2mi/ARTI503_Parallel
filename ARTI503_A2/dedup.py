import hashlib
import re
import zlib
from time import perf_counter

import numpy as np

SHINGLE_SIZE = 5
NUM_PERM = 128
BANDS = 32
ROWS = 4
JACCARD_THRESHOLD = 0.5
SEED = 503
PRIME = 4294967291

STAGES = ["normalize", "shingle", "minhash", "lsh", "candidates", "verify", "cluster"]

DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]+")
ALEF = re.compile("[آأإٱ]")
NON_WORD = re.compile(r"[\W_]+")


def normalize(text):
    text = ALEF.sub("ا", DIACRITICS.sub("", text)).lower()
    return [tok for tok in NON_WORD.split(text) if tok]


def shingles(toks, size=SHINGLE_SIZE):
    cnt = len(toks)
    if cnt == 0:
        return np.empty(0, dtype=np.uint32)
    if cnt < size:
        grams = [" ".join(toks)]
    else:
        grams = [" ".join(toks[pos:pos + size]) for pos in range(cnt - size + 1)]
    hashes = np.fromiter((zlib.crc32(gram.encode("utf-8")) for gram in grams), dtype=np.uint32, count=len(grams))
    return np.unique(hashes)


class MinHasher:

    def __init__(self, num_perm=NUM_PERM, seed=SEED):
        def coef(tag, idx, low):
            dig = hashlib.sha256("{}-{}-{}".format(seed, tag, idx).encode()).digest()
            return low + int.from_bytes(dig[:8], "little") % (PRIME - low)

        self.a = np.array([coef("a", idx, 1) for idx in range(num_perm)], dtype=np.uint64)
        self.b = np.array([coef("b", idx, 0) for idx in range(num_perm)], dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, arr):
        vals = arr.astype(np.uint64)
        mat = (np.multiply.outer(vals, self.a) + self.b) % np.uint64(PRIME)
        return mat.min(axis=0).astype(np.uint32)


def jaccard(set_a, set_b):
    inter = np.intersect1d(set_a, set_b, assume_unique=True).size
    union = set_a.size + set_b.size - inter
    return inter / union if union else 0.0


class UnionFind:
    def __init__(self, num):
        self.parent = list(range(num))
        self.size = [1] * num

    def find(self, node):
        par = self.parent
        while par[node] != node:
            par[node] = par[par[node]]
            node = par[node]
        return node

    def union(self, node_a, node_b):
        root_a, root_b = self.find(node_a), self.find(node_b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def run_pipeline(docs, hasher=None):
    if hasher is None:
        hasher = MinHasher()
    assert BANDS * ROWS == hasher.num_perm

    num = len(docs)
    times = dict.fromkeys(STAGES, 0.0)
    start = perf_counter()

    sets = [None] * num
    sigs = np.zeros((num, hasher.num_perm), dtype=np.uint32)
    has_sig = np.zeros(num, dtype=bool)
    for doc_id, doc in enumerate(docs):
        t0 = perf_counter()
        toks = normalize(doc)
        t1 = perf_counter()
        arr = shingles(toks)
        t2 = perf_counter()
        if arr.size:
            sigs[doc_id] = hasher.signature(arr)
            has_sig[doc_id] = True
        t3 = perf_counter()
        sets[doc_id] = arr
        times["normalize"] += t1 - t0
        times["shingle"] += t2 - t1
        times["minhash"] += t3 - t2

    cands = set()
    max_bucket = 0
    num_buckets = 0
    valid = np.flatnonzero(has_sig).tolist()
    width = ROWS * 4
    for band in range(BANDS):
        t0 = perf_counter()
        raw = np.ascontiguousarray(sigs[valid, band * ROWS:(band + 1) * ROWS]).tobytes()
        buckets = {}
        for pos, doc_id in enumerate(valid):
            key = raw[pos * width:(pos + 1) * width]
            bucket = buckets.get(key)
            if bucket is None:
                buckets[key] = [doc_id]
            else:
                bucket.append(doc_id)
        t1 = perf_counter()

        for grp in buckets.values():
            size = len(grp)
            if size < 2:
                continue
            if size > max_bucket:
                max_bucket = size
            for pa in range(size):
                first = grp[pa]
                for pb in range(pa + 1, size):
                    cands.add((first, grp[pb]))
        num_buckets += len(buckets)
        del buckets
        t2 = perf_counter()
        times["lsh"] += t1 - t0
        times["candidates"] += t2 - t1

    t0 = perf_counter()
    edges = []
    for doc_a, doc_b in sorted(cands):
        if jaccard(sets[doc_a], sets[doc_b]) >= JACCARD_THRESHOLD:
            edges.append((doc_a, doc_b))
    times["verify"] = perf_counter() - t0

    t0 = perf_counter()
    uf = UnionFind(num)
    for doc_a, doc_b in edges:
        uf.union(doc_a, doc_b)
    groups = {}
    for doc_id in range(num):
        root = uf.find(doc_id)
        if uf.size[root] > 1:
            groups.setdefault(root, []).append(doc_id)
    clusters = sorted(groups.values())
    times["cluster"] = perf_counter() - t0

    wall = perf_counter() - start
    edge_bytes = np.asarray(edges, dtype=np.int64).tobytes() if edges else b""

    return {
        "times": times,
        "total": sum(times.values()),
        "wall": wall,
        "clusters": clusters,
        "stats": {
            "docs": num,
            "empty_docs": int(num - has_sig.sum()),
            "shingles": int(sum(arr.size for arr in sets)),
            "buckets": num_buckets,
            "max_bucket": max_bucket,
            "candidates": len(cands),
            "verified_pairs": len(edges),
            "clusters": len(clusters),
            "docs_in_clusters": sum(len(grp) for grp in clusters),
            "sig_checksum": zlib.crc32(sigs.tobytes()),
            "edge_checksum": zlib.crc32(edge_bytes),
        },
    }


def pair_scores(pred, truth, num):
    pred_id = {}
    for cid, grp in enumerate(pred):
        for doc in grp:
            pred_id[doc] = cid
    true_id = {}
    for cid, grp in enumerate(truth):
        grp = [doc for doc in grp if doc < num]
        if len(grp) > 1:
            for doc in grp:
                true_id[doc] = cid

    def pairs(cnt):
        return cnt * (cnt - 1) // 2

    pred_size, true_size, both = {}, {}, {}
    for doc, cid in pred_id.items():
        pred_size[cid] = pred_size.get(cid, 0) + 1
    for doc, cid in true_id.items():
        true_size[cid] = true_size.get(cid, 0) + 1
        if doc in pred_id:
            key = (pred_id[doc], cid)
            both[key] = both.get(key, 0) + 1

    correct = sum(pairs(cnt) for cnt in both.values())
    predicted = sum(pairs(cnt) for cnt in pred_size.values())
    actual = sum(pairs(cnt) for cnt in true_size.values())
    return {
        "true_pairs": actual,
        "predicted_pairs": predicted,
        "correct_pairs": correct,
        "precision": correct / predicted if predicted else 1.0,
        "recall": correct / actual if actual else 1.0,
    }
