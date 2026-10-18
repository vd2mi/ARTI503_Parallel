import csv
import glob
import json
import os
import statistics
import sys

STAGES = ["normalize", "shingle", "minhash", "lsh", "candidates", "verify", "cluster"]
CHECK_KEYS = ["candidates", "verified_pairs", "clusters", "sig_checksum", "edge_checksum"]

STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BAR = "#2a78d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def load(folder):
    members = []
    for path in sorted(glob.glob(os.path.join(folder, "RESULTS_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            r["_file"] = os.path.basename(path)
            members.append(r)
        except Exception as exc:
            print("Could not read {}: {}".format(path, exc))
    return members


def find_problems(members):
    problems = []
    hashes = {m["data"]["corpus_sha256"] for m in members}
    if len(hashes) > 1:
        problems.append("Members used DIFFERENT data files: " + ", ".join(
            "{}={}".format(m["member"], m["data"]["corpus_sha256"][:10]) for m in members))
    ref = {}
    for m in members:
        for s in m["sizes"]:
            key = tuple(s["stats"][k] for k in CHECK_KEYS)
            ref.setdefault(s["docs"], {}).setdefault(key, []).append(m["member"])
    for docs, variants in sorted(ref.items()):
        if len(variants) > 1:
            ranked = sorted(variants.items(), key=lambda kv: -len(kv[1]))
            for key, who in ranked[1:]:
                problems.append("{:,} docs: results differ from the majority for {} (check their run)".format(
                    docs, ", ".join(who)))
    for m in members:
        if not m.get("complete"):
            problems.append("{} did not finish: {}".format(m["member"], m.get("stopped_reason", "unknown")))
        if m["system"].get("on_ac_power") is False:
            problems.append("{} ran on battery power (timings may be slower/noisier)".format(m["member"]))
        for s in m["sizes"]:
            if s["total_mean"] and s["total_std"] / s["total_mean"] > 0.10:
                problems.append("{} at {:,} docs: high variation between runs (std {:.0f}% of mean)".format(
                    m["member"], s["docs"], 100 * s["total_std"] / s["total_mean"]))
    names = [m["member"] for m in members]
    for n in set(names):
        if names.count(n) > 1:
            problems.append("Two result files use the same name: " + n)
    return problems


def write_csvs(members, out):
    sizes = sorted({s["docs"] for m in members for s in m["sizes"]})
    with open(os.path.join(out, "team_times.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["member", "cpu_model", "physical_cores", "logical_cores", "max_freq_mhz", "ram_gb",
                "os", "python", "numpy", "on_ac_power", "complete"]
        for n in sizes:
            head += ["mean_s_{}".format(n), "std_s_{}".format(n), "repeats_{}".format(n)]
        w.writerow(head)
        for m in members:
            sy = m["system"]
            row = [m["member"], sy.get("cpu_model"), sy.get("physical_cores"), sy.get("logical_cores"),
                   sy.get("max_freq_mhz"), sy.get("ram_gb"), sy.get("os"), sy.get("python"),
                   sy.get("numpy"), sy.get("on_ac_power"), m.get("complete")]
            by = {s["docs"]: s for s in m["sizes"]}
            for n in sizes:
                s = by.get(n)
                row += [round(s["total_mean"], 3), round(s["total_std"], 3), s["repeats"]] if s else ["", "", ""]
            w.writerow(row)
    with open(os.path.join(out, "stage_breakdown.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["member", "docs"] + ["{}_s".format(s) for s in STAGES] + ["{}_pct".format(s) for s in STAGES])
        for m in members:
            for s in m["sizes"]:
                w.writerow([m["member"], s["docs"]] + [round(s["stage_mean"][k], 4) for k in STAGES]
                           + [round(s["stage_share_pct"][k], 1) for k in STAGES])
    return sizes


def write_markdown(members, sizes, out):
    lines = ["## Team execution times (mean ± std over repeats, seconds)", ""]
    head = "| Member | CPU | Cores (P/L) | RAM (GB) | " + " | ".join("{:,} docs".format(n) for n in sizes) + " |"
    lines += [head, "|" + "---|" * (4 + len(sizes))]
    for m in members:
        sy = m["system"]
        by = {s["docs"]: s for s in m["sizes"]}
        cells = ["{:.2f} ± {:.2f}".format(by[n]["total_mean"], by[n]["total_std"]) if n in by else "n/a"
                 for n in sizes]
        lines.append("| {} | {} | {}/{} | {} | {} |".format(
            m["member"], sy.get("cpu_model"), sy.get("physical_cores"), sy.get("logical_cores"),
            sy.get("ram_gb"), " | ".join(cells)))

    lines += ["", "## Share of time per stage (median over members, %)", ""]
    lines += ["| Docs | " + " | ".join(STAGES) + " |", "|" + "---|" * (1 + len(STAGES))]
    for n in sizes:
        rows = [s for m in members for s in m["sizes"] if s["docs"] == n]
        if rows:
            med = [statistics.median(r["stage_share_pct"][k] for r in rows) for k in STAGES]
            lines.append("| {:,} | ".format(n) + " | ".join("{:.1f}".format(v) for v in med) + " |")

    lines += ["", "## Detection quality vs injected near-duplicates", ""]
    lines += ["| Docs | True pairs | Predicted pairs | Correct | Precision | Recall |", "|---|---|---|---|---|---|"]
    for n in sizes:
        acc = next((s["accuracy"] for m in members for s in m["sizes"] if s["docs"] == n), None)
        if acc:
            lines.append("| {:,} | {:,} | {:,} | {:,} | {:.1f}% | {:.1f}% |".format(
                n, acc["true_pairs"], acc["predicted_pairs"], acc["correct_pairs"],
                acc["precision"] * 100, acc["recall"] * 100))
    with open(os.path.join(out, "report_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def figures(members, sizes, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures (pip install matplotlib)")
        return
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})

    common = [n for n in sizes if all(any(s["docs"] == n for s in m["sizes"]) for m in members)]
    if common:
        n = common[-1]
        rows = sorted(((m["member"], next(s for s in m["sizes"] if s["docs"] == n)) for m in members),
                      key=lambda x: x[1]["total_mean"])
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.2))
        y = range(len(rows))
        ax.barh(list(y), [r[1]["total_mean"] for r in rows], xerr=[r[1]["total_std"] for r in rows],
                color=BAR, height=0.6, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})
        ax.set_yticks(list(y))
        ax.set_yticklabels([r[0] for r in rows])
        for i, r in enumerate(rows):
            ax.text(r[1]["total_mean"] + r[1]["total_std"], i, "  {:.1f} s".format(r[1]["total_mean"]),
                    va="center", fontsize=8, color=INK)
        ax.set_xlabel("Mean sequential time (s), error bar = std")
        ax.set_title("Sequential run time per member, {:,} documents".format(n))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_team_times.png"), dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    med, lo, hi, xs = [], [], [], []
    for n in sizes:
        vals = [s["total_mean"] for m in members for s in m["sizes"] if s["docs"] == n]
        if vals:
            xs.append(n)
            med.append(statistics.median(vals))
            lo.append(min(vals))
            hi.append(max(vals))
    ax.fill_between(xs, lo, hi, color=BAR, alpha=0.15, linewidth=0, label="fastest to slowest member")
    ax.plot(xs, med, color=BAR, linewidth=2, marker="o", markersize=6, label="median member")
    for x, v in zip(xs, med):
        ax.annotate("{:.1f} s".format(v), (x, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["{:,}".format(x) for x in xs])
    ax.set_xlabel("Number of documents")
    ax.set_ylabel("Total time (s), log scale")
    ax.set_title("Sequential time vs problem size")
    ax.grid(color=GRID, linewidth=0.8, which="major")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_scaling.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(sizes) + 1.8))
    for yi, n in enumerate(sizes):
        rows = [s for m in members for s in m["sizes"] if s["docs"] == n]
        shares = [statistics.median(r["stage_share_pct"][k] for r in rows) for k in STAGES]
        total = sum(shares) or 1
        shares = [100 * v / total for v in shares]
        left = 0
        for k, (st, v) in enumerate(zip(STAGES, shares)):
            ax.barh(yi, v, left=left, color=STAGE_COLORS[k], height=0.6, edgecolor="white", linewidth=1.5,
                    label=st if yi == 0 else None)
            if v >= 7:
                ax.text(left + v / 2, yi, "{:.0f}%".format(v), ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels(["{:,} docs".format(n) for n in sizes])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total time (%)")
    ax.set_title("Where the sequential time goes")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_stage_share.png"), dpi=200)
    plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python merge_results.py <folder with RESULTS_*.json>")
    folder = sys.argv[1]
    members = load(folder)
    if not members:
        sys.exit("No RESULTS_*.json files found in " + folder)
    members.sort(key=lambda m: m["member"].lower())
    out = os.path.join(folder, "merged")
    os.makedirs(out, exist_ok=True)

    problems = find_problems(members)
    with open(os.path.join(out, "checks.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(problems) + "\n" if problems else "No problems found.\n")
    sizes = write_csvs(members, out)
    write_markdown(members, sizes, out)
    figures(members, sizes, out)

    print("Merged {} members -> {}".format(len(members), out))
    print("\n".join("  ! " + p for p in problems) if problems else "  No problems found.")


if __name__ == "__main__":
    main()
