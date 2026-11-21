# ARTI 503 - Parallel Near-Duplicate Detection (Group 9MA2G2)

MinHash + LSH near-duplicate detector for Arabic web text.

## Stage 1 - sequential version

- `ARTI503_Stage1_code/arti503_stage1/` - pipeline (`dedup.py`), benchmark runner (`run.py`), system info (`sysinfo.py`)
- `ARTI503_Stage1_code/tools/` - data preparation, packaging and result merging
- `ARTI503_Stage1_code/results/` - benchmark results from all 11 members

Run: `python run.py --name YourName`

## Assignment 2 - race conditions and synchronisation (multi-core V1)

- `ARTI503_A2/omp.cpp` - C++ OpenMP: sequential, race, critical, atomic and reduction versions
- `ARTI503_A2/par.py` - Python multiprocessing: race, lock, per-update lock and reduction versions
- `ARTI503_A2/run_a2.py` - compiles the C++ code, runs every version at 1, 2, 4 and 8 workers and checks each run against the sequential result
- `ARTI503_A2/results/` - results (Intel Core i5-14400F)

Run: `py run_a2.py --name YourName` (needs g++ with OpenMP; on Windows it installs WinLibs through winget if missing)

Data: FineWeb-2 (arb_Arab), not included in the repo because of size. `run_a2.py` looks for it in `../arti503_stage1/data/` or `ARTI503_A2/data/`.
