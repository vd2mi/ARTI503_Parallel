import gzip
import json
import multiprocessing as mp
import os
from time import perf_counter

import numpy as np

import dedup

BANDS = 32
ROWS = 4
SLOTS = 1 << 14
MASK = (1 << 64) - 1
CHUNK = 64

DOCS = None
HASHER = None
TOTAL = None
COLL = None
HIST = None
LOCK = None


def load_docs(path, num):
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as fin:
        for line in fin:
            docs.append(json.loads(line)["text"])
            if len(docs) >= num:
                break
    return docs


def slot_of(sig, band):
    hv = int(sig[band * ROWS])
    for row in range(1, ROWS):
        hv = ((hv * 1000003) & MASK) ^ int(sig[band * ROWS + row])
    hv ^= hv >> 29
    return hv % SLOTS


def doc_work(doc):
    arr = dedup.shingles(dedup.normalize(doc))
    sig = HASHER.signature(arr)
    return arr.size, [band * SLOTS + slot_of(sig, band) for band in range(BANDS)], int(sig.sum(dtype=np.uint64))


def init_worker(path, num, total, coll, hist, lock):
    global DOCS, HASHER, TOTAL, COLL, HIST, LOCK
    DOCS = load_docs(path, num)
    HASHER = dedup.MinHasher()
    TOTAL, COLL, HIST, LOCK = total, coll, hist, lock


def ping(_):
    import time
    time.sleep(0.2)
    return os.getpid()


def chunk_race(span):
    sig_sum = 0
    for doc_id in range(*span):
        cnt, idxs, ssum = doc_work(DOCS[doc_id])
        sig_sum += ssum
        for _ in range(cnt):
            TOTAL.value += 1
        for idx in idxs:
            if HIST[idx] > 0:
                COLL.value += 1
            HIST[idx] += 1
    return sig_sum


def chunk_lock(span):
    sig_sum = 0
    for doc_id in range(*span):
        cnt, idxs, ssum = doc_work(DOCS[doc_id])
        sig_sum += ssum
        LOCK.acquire()
        for _ in range(cnt):
            TOTAL.value += 1
        for idx in idxs:
            if HIST[idx] > 0:
                COLL.value += 1
            HIST[idx] += 1
        LOCK.release()
    return sig_sum


def chunk_atomic(span):
    sig_sum = 0
    for doc_id in range(*span):
        cnt, idxs, ssum = doc_work(DOCS[doc_id])
        sig_sum += ssum
        for _ in range(cnt):
            with TOTAL.get_lock():
                TOTAL.value += 1
        for idx in idxs:
            with HIST.get_lock():
                old = HIST[idx]
                HIST[idx] = old + 1
            if old > 0:
                with COLL.get_lock():
                    COLL.value += 1
    return sig_sum


def chunk_reduce(span):
    sig_sum = 0
    total = 0
    hist = np.zeros(BANDS * SLOTS, dtype=np.int32)
    for doc_id in range(*span):
        cnt, idxs, ssum = doc_work(DOCS[doc_id])
        sig_sum += ssum
        total += cnt
        hist[idxs] += 1
    used = np.flatnonzero(hist)
    return sig_sum, total, used, hist[used]


WORKERS = {"race": chunk_race, "lock": chunk_lock, "atomic": chunk_atomic, "reduction": chunk_reduce}


def hist_sum(hist):
    arr = np.asarray(hist, dtype=np.uint64)
    return int((arr * np.arange(1, arr.size + 1, dtype=np.uint64)).sum(dtype=np.uint64))


def sequential(docs):
    global HASHER
    HASHER = dedup.MinHasher()
    hist = np.zeros(BANDS * SLOTS, dtype=np.int64)
    total = coll = sig_sum = 0
    start = perf_counter()
    for doc in docs:
        cnt, idxs, ssum = doc_work(doc)
        sig_sum += ssum
        total += cnt
        for idx in idxs:
            if hist[idx] > 0:
                coll += 1
            hist[idx] += 1
    secs = perf_counter() - start
    return {"seconds": secs, "total": total, "collisions": coll,
            "hist_sum": hist_sum(hist), "sig_sum": sig_sum % (1 << 64)}


def parallel(path, num, version, procs):
    safe = version == "atomic"
    total = mp.Value("q", 0) if safe else mp.RawValue("q", 0)
    coll = mp.Value("q", 0) if safe else mp.RawValue("q", 0)
    hist = mp.Array("i", BANDS * SLOTS) if safe else mp.RawArray("i", BANDS * SLOTS)
    lock = mp.Lock()
    spans = [(lo, min(lo + CHUNK, num)) for lo in range(0, num, CHUNK)]
    with mp.Pool(procs, initializer=init_worker, initargs=(path, num, total, coll, hist, lock)) as pool:
        pool.map(ping, range(procs * 2), chunksize=1)
        start = perf_counter()
        outs = list(pool.imap_unordered(WORKERS[version], spans))
        if version == "reduction":
            merged = np.zeros(BANDS * SLOTS, dtype=np.int64)
            tot = 0
            for _, part_total, used, counts in outs:
                merged[used] += counts
                tot += part_total
            coll_val = int(np.maximum(merged - 1, 0).sum())
        secs = perf_counter() - start
    if version == "reduction":
        sig_sum = sum(out[0] for out in outs)
        return {"seconds": secs, "total": tot, "collisions": coll_val,
                "hist_sum": hist_sum(merged), "sig_sum": sig_sum % (1 << 64)}
    raw = hist.get_obj() if safe else hist
    return {"seconds": secs, "total": (total.value if not safe else total.value),
            "collisions": coll.value, "hist_sum": hist_sum(np.frombuffer(raw, dtype=np.int32)),
            "sig_sum": sum(outs) % (1 << 64)}
