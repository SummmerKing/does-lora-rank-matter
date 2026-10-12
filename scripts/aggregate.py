"""Aggregate all evaluated runs into results/summary.md, results/summary.json
and figures in results/figures/.

Comparisons (per benchmark, Holm-corrected together):
  * every fine-tuned configuration vs the base model
  * lora_r4 vs lora_r16 and lora_r64 vs lora_r16       (does rank matter?)
  * loraplus_r16 vs lora_r16                            (does LoRA+ matter?)
"""

import json

import numpy as np
from _common import GRID, RESULTS_DIR, RUNS_DIR, read_jsonl

from lora_code.stats import holm, mcnemar_exact, paired_comparison, summarize

BENCHES = ["humaneval", "mbpp"]
PAIRS = [("lora_r4", "lora_r16"), ("lora_r64", "lora_r16"), ("loraplus_r16", "lora_r16")]

# Reference categorical palette (light mode), plus a neutral for the baseline.
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def matrix(run_dirs, bench):
    rows = []
    task_order = None
    for d in run_dirs:
        res = read_jsonl(d / f"{bench}.jsonl")
        ids = [r["task_id"] for r in res]
        if task_order is None:
            task_order = ids
        elif ids != task_order:
            raise ValueError(f"task order mismatch in {d}")
        rows.append([r["passed"] for r in res])
    return np.array(rows), task_order


def collect():
    configs = {"base": [RESULTS_DIR / "base"]}
    for name in GRID["configs"]:
        dirs = sorted((RESULTS_DIR / name).glob("seed*"))
        if dirs:
            configs[name] = dirs
    return configs


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    configs = collect()
    out = {"benchmarks": {}, "loss_vs_pass": {}}
    lines = ["# Results", "", f"Model: `{GRID['model_name']}`. Greedy decoding, pass@1. "
             "95% CIs bootstrap over problems; p-values are paired sign-flip tests, Holm-corrected "
             "within each benchmark.", ""]

    for bench in BENCHES:
        mats = {name: matrix(dirs, bench)[0] for name, dirs in configs.items()}
        summ = {name: summarize(m) for name, m in mats.items()}
        comps = [(n, "base") for n in mats if n != "base"] + [p for p in PAIRS if p[0] in mats and p[1] in mats]
        tests = [paired_comparison(mats[a], mats[b]) for a, b in comps]
        adj = holm([t["p_value"] for t in tests])
        for t, p in zip(tests, adj):
            t["p_holm"] = p
        mcn = {n: mcnemar_exact(mats[n][0], mats["base"][0]) for n in mats if n != "base"}
        out["benchmarks"][bench] = {"summary": summ,
                                    "comparisons": {f"{a} vs {b}": t for (a, b), t in zip(comps, tests)},
                                    "mcnemar_seed0_vs_base": mcn}

        n_prob = summ["base"]["n_problems"]
        lines += [f"## {bench} ({n_prob} problems)", "",
                  "| config | seeds | pass@1 | 95% CI | seed SD |", "|---|---|---|---|---|"]
        for name, s in summ.items():
            lines.append(f"| {name} | {s['n_seeds']} | {100*s['pass@1']:.1f} | "
                         f"[{100*s['ci_low']:.1f}, {100*s['ci_high']:.1f}] | {100*s['seed_sd']:.1f} |")
        lines += ["", "| comparison | Δ pass@1 (pts) | 95% CI | p | p (Holm) |", "|---|---|---|---|---|"]
        for (a, b), t in zip(comps, tests):
            lines.append(f"| {a} vs {b} | {100*t['diff']:+.1f} | [{100*t['ci_low']:+.1f}, {100*t['ci_high']:+.1f}] "
                         f"| {t['p_value']:.3f} | {t['p_holm']:.3f} |")
        lines.append("")

    # Is validation loss a good proxy for functional correctness?
    xs, ys = {b: [] for b in BENCHES}, {b: [] for b in BENCHES}
    for name, dirs in configs.items():
        if name == "base":
            continue
        for d in dirs:
            metrics = RUNS_DIR / name / d.name / "metrics.json"
            if not metrics.exists():
                continue
            loss = json.loads(metrics.read_text())["final_val_loss"]
            for b in BENCHES:
                xs[b].append(loss)
                ys[b].append(np.mean([r["passed"] for r in read_jsonl(d / f"{b}.jsonl")]))
    lines += ["## Validation loss vs pass@1", ""]
    for b in BENCHES:
        if len(xs[b]) >= 3:
            rho = spearman(xs[b], ys[b])
            out["loss_vs_pass"][b] = {"spearman": rho, "n_runs": len(xs[b])}
            lines.append(f"- {b}: Spearman ρ = {rho:+.2f} over {len(xs[b])} runs "
                         "(negative means lower loss goes with higher pass@1).")
    lines.append("")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))
    plot(out, configs)
    print("\n".join(lines))


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(out, configs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paired gain over the base model (the quantity that is actually tested).
    #    Absolute pass@1 CIs overlap heavily because they include problem
    #    difficulty, which pairing cancels; plotting them would mislead.
    names = [n for n in configs if n != "base"]
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(10, 0.6 * len(names) + 1.6), sharey=True)
    for ax, bench in zip(np.atleast_1d(axes), BENCHES):
        comps = out["benchmarks"][bench]["comparisons"]
        base = out["benchmarks"][bench]["summary"]["base"]["pass@1"]
        ax.axvline(0, color=MUTED, lw=1.5, ls="--")
        for i, n in enumerate(names):
            c = comps[f"{n} vs base"]
            color = SERIES[1] if n.startswith("loraplus") else SERIES[0]
            ax.plot([100 * c["ci_low"], 100 * c["ci_high"]], [i, i], color=color, lw=2, solid_capstyle="round")
            ax.plot(100 * c["diff"], i, "o", color=color, ms=8)
            ax.text(100 * c["ci_high"], i, f"  {100 * c['diff']:+.1f}", color=INK, fontsize=9, va="center")
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_title(f"{bench} (base model: {100 * base:.1f}%)", color=INK, loc="left", fontsize=11)
        ax.set_xlabel("Δ pass@1 vs base model (points)", color=INK)
        ax.margins(x=0.15)
        _style(ax)
    fig.suptitle("Gain over the base model, paired by problem, with 95% bootstrap CI", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "gain_over_base.png", dpi=200)
    plt.close(fig)

    # 2. validation loss curves (seed 0) per configuration.
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ends = []
    for i, n in enumerate(names):
        m = RUNS_DIR / n / "seed0" / "metrics.json"
        if not m.exists():
            continue
        h = json.loads(m.read_text())["history"]
        ax.plot(h["val_step"], h["val_loss"], color=SERIES[i % len(SERIES)], lw=2, label=n)
        ends.append([h["val_loss"][-1], h["val_step"][-1], n])
    # Direct labels at line ends, nudged apart so they never overlap.
    if ends:
        ends.sort()
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        for y, x, n in ends:
            ax.text(x, y, f"  {n}", color=INK, fontsize=8, va="center")
        ax.margins(x=0.18)
    ax.set_xlabel("optimizer step", color=INK)
    ax.set_ylabel("validation loss (response tokens)", color=INK)
    ax.set_title("Validation loss, seed 0", color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRIDC, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "val_loss.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
