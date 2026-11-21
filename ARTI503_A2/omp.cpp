#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>

static const int NUM_PERM = 128;
static const int BANDS = 32;
static const int ROWS = 4;
static const int SLOTS = 1 << 14;
static const uint64_t PRIME = 4294967291ULL;

struct Corpus {
    uint64_t num = 0;
    std::vector<uint64_t> offs;
    std::vector<uint32_t> vals;
    uint64_t coef_a[NUM_PERM];
    uint64_t coef_b[NUM_PERM];
};

static bool load(const char* shingle_path, const char* coef_path, Corpus& data) {
    FILE* fin = fopen(shingle_path, "rb");
    if (!fin) return false;
    if (fread(&data.num, 8, 1, fin) != 1) return false;
    data.offs.resize(data.num + 1);
    if (fread(data.offs.data(), 8, data.num + 1, fin) != data.num + 1) return false;
    data.vals.resize(data.offs[data.num]);
    if (fread(data.vals.data(), 4, data.vals.size(), fin) != data.vals.size()) return false;
    fclose(fin);

    FILE* fcoef = fopen(coef_path, "rb");
    if (!fcoef) return false;
    if (fread(data.coef_a, 8, NUM_PERM, fcoef) != NUM_PERM) return false;
    if (fread(data.coef_b, 8, NUM_PERM, fcoef) != NUM_PERM) return false;
    fclose(fcoef);
    return true;
}

static inline void minhash(const Corpus& data, uint64_t doc, uint32_t* sig) {
    const uint32_t* arr = data.vals.data() + data.offs[doc];
    uint64_t cnt = data.offs[doc + 1] - data.offs[doc];
    for (int perm = 0; perm < NUM_PERM; perm++) {
        uint64_t best = UINT64_MAX, ca = data.coef_a[perm], cb = data.coef_b[perm];
        for (uint64_t pos = 0; pos < cnt; pos++) {
            uint64_t val = (ca * arr[pos] + cb) % PRIME;
            if (val < best) best = val;
        }
        sig[perm] = (uint32_t)best;
    }
}

static inline uint32_t slot_of(const uint32_t* sig, int band) {
    uint64_t hv = sig[band * ROWS];
    for (int row = 1; row < ROWS; row++) hv = hv * 1000003ULL ^ sig[band * ROWS + row];
    hv ^= hv >> 29;
    return (uint32_t)(hv % SLOTS);
}

struct Result {
    double secs;
    uint64_t total, coll, hist_sum, sig_sum;
};

static Result run(const Corpus& data, uint64_t num, int mode, int threads) {
    std::vector<uint32_t> sigs(num * NUM_PERM);
    std::vector<uint32_t> hist((size_t)BANDS * SLOTS, 0);
    uint32_t* hp = hist.data();
    uint64_t total = 0, coll = 0;
    long long cnt_docs = (long long)num;

    double start = omp_get_wtime();

    if (mode == 0) {
        for (long long doc = 0; doc < cnt_docs; doc++) {
            uint32_t* sig = &sigs[doc * NUM_PERM];
            minhash(data, doc, sig);
            total += data.offs[doc + 1] - data.offs[doc];
            for (int band = 0; band < BANDS; band++) {
                size_t idx = (size_t)band * SLOTS + slot_of(sig, band);
                if (hp[idx] > 0) coll++;
                hp[idx]++;
            }
        }
    } else if (mode == 1) {
        #pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
        for (long long doc = 0; doc < cnt_docs; doc++) {
            uint32_t* sig = &sigs[doc * NUM_PERM];
            minhash(data, doc, sig);
            total += data.offs[doc + 1] - data.offs[doc];
            for (int band = 0; band < BANDS; band++) {
                size_t idx = (size_t)band * SLOTS + slot_of(sig, band);
                if (hp[idx] > 0) coll++;
                hp[idx]++;
            }
        }
    } else if (mode == 2) {
        #pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
        for (long long doc = 0; doc < cnt_docs; doc++) {
            uint32_t* sig = &sigs[doc * NUM_PERM];
            minhash(data, doc, sig);
            #pragma omp critical
            {
                total += data.offs[doc + 1] - data.offs[doc];
                for (int band = 0; band < BANDS; band++) {
                    size_t idx = (size_t)band * SLOTS + slot_of(sig, band);
                    if (hp[idx] > 0) coll++;
                    hp[idx]++;
                }
            }
        }
    } else if (mode == 3) {
        #pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
        for (long long doc = 0; doc < cnt_docs; doc++) {
            uint32_t* sig = &sigs[doc * NUM_PERM];
            minhash(data, doc, sig);
            uint64_t cnt = data.offs[doc + 1] - data.offs[doc];
            #pragma omp atomic
            total += cnt;
            for (int band = 0; band < BANDS; band++) {
                size_t idx = (size_t)band * SLOTS + slot_of(sig, band);
                uint32_t old;
                #pragma omp atomic capture
                old = hp[idx]++;
                if (old > 0) {
                    #pragma omp atomic
                    coll++;
                }
            }
        }
    } else {
        #pragma omp parallel num_threads(threads)
        {
            std::vector<uint32_t> local((size_t)BANDS * SLOTS, 0);
            #pragma omp for schedule(dynamic, 64) reduction(+:total)
            for (long long doc = 0; doc < cnt_docs; doc++) {
                uint32_t* sig = &sigs[doc * NUM_PERM];
                minhash(data, doc, sig);
                total += data.offs[doc + 1] - data.offs[doc];
                for (int band = 0; band < BANDS; band++)
                    local[(size_t)band * SLOTS + slot_of(sig, band)]++;
            }
            int nth = omp_get_num_threads(), tid = omp_get_thread_num();
            size_t len = local.size(), part = (len + nth - 1) / nth;
            for (int step = 0; step < nth; step++) {
                int blk = (tid + step) % nth;
                size_t lo = blk * part, hi = lo + part < len ? lo + part : len;
                for (size_t idx = lo; idx < hi; idx++) {
                    #pragma omp atomic
                    hp[idx] += local[idx];
                }
            }
        }
        for (size_t idx = 0; idx < hist.size(); idx++)
            if (hp[idx] > 1) coll += hp[idx] - 1;
    }

    double secs = omp_get_wtime() - start;

    uint64_t hist_sum = 0, sig_sum = 0;
    for (size_t idx = 0; idx < hist.size(); idx++) hist_sum += (uint64_t)hp[idx] * (idx + 1);
    for (uint32_t val : sigs) sig_sum += val;
    return {secs, total, coll, hist_sum, sig_sum};
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: omp shingles.bin coef.bin num_docs threads(e.g. 1,2,4,8) repeats [modes]\n");
        return 1;
    }
    Corpus data;
    if (!load(argv[1], argv[2], data)) {
        fprintf(stderr, "could not read input files\n");
        return 1;
    }
    uint64_t num = strtoull(argv[3], nullptr, 10);
    if (num == 0 || num > data.num) num = data.num;
    std::vector<int> thread_list;
    std::string arg = argv[4];
    size_t pos = 0;
    while (pos < arg.size()) {
        size_t next = arg.find(',', pos);
        if (next == std::string::npos) next = arg.size();
        thread_list.push_back(atoi(arg.substr(pos, next - pos).c_str()));
        pos = next + 1;
    }
    int repeats = atoi(argv[5]);
    std::string modes = argc > 6 ? argv[6] : "01234";
    const char* names[] = {"sequential", "race", "critical", "atomic", "reduction"};

    printf("version,threads,rep,seconds,total,collisions,hist_sum,sig_sum\n");
    for (char mch : modes) {
        int mode = mch - '0';
        for (int threads : thread_list) {
            if (mode == 0 && threads != thread_list[0]) continue;
            int used = mode == 0 ? 1 : threads;
            for (int rep = 0; rep < repeats; rep++) {
                Result res = run(data, num, mode, used);
                printf("%s,%d,%d,%.6f,%llu,%llu,%llu,%llu\n", names[mode], used, rep + 1, res.secs,
                       (unsigned long long)res.total, (unsigned long long)res.coll,
                       (unsigned long long)res.hist_sum, (unsigned long long)res.sig_sum);
                fflush(stdout);
            }
        }
    }
    return 0;
}
