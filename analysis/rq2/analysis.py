"""RQ2 — does the object need to be chosen? Task-level paired analysis (ER-2).

Hypothesis under test, H_sem: "the added object must be chosen for the task (or at least
fit the scene) to flip a refusal". Three rungs on one axis, how much the added object was
chosen for the task, all measured on the SAME stratified tasks:

    targeted    RQ1's objects: proposer saw image + instruction, red-team prompt (20/task)
    scene_only  proposer saw the image only, neutral prompt                      (5/task)
    random      drawn from a fixed COCO-derived list, no proposer at all         (5/task)

plus the clean re-query `baseline` (the non-determinism floor, 1/task). Every comparison is
PAIRED at the task level. Unit of analysis = task: repeated trials on one task are correlated,
so each task collapses to one bypass rate per condition (P(J==1) over conclusive trials,
J==-1 dropped from the denominator -- the shared convention in analysis/rq1/bypass_rates.py).

Pre-registered tests (written before the run; see README.md):

  Primary   H_sem: random vs targeted, two-sided paired t (difference) AND paired
            TOST at +/-10pp (equivalence, 90% CI). Equivalence disproves H_sem.
  Secondary scene_only vs targeted, scene_only vs random, same two tests.
  Floor     each rung > baseline: paired one-sided t + Wilcoxon, 95% CI. Without this,
            "equivalent" could mean "all three are equally near zero".
  Task prop Spearman(rung, rung) with a 10k bootstrap CI (seed 42): malleability is a
            property of the task, not of the stimulus.
  Strata    per-stratum means, visualisation / manipulation check only
            (strata were defined by the targeted rate, so that gradient is by construction).

Holm-adjusted p-values accompany the raw ones for the three rung-pair tests.

Inputs (ER-2; the ER-1.6 run used EARDataset/stratified_{baseline,scene_only,random}.csv
and robot_constitution_64_automerge_attack_5_judged.csv):
    EARDataset/robot_constitution_64_automerge_er2_attack_20.csv   targeted, restricted to the sample
    EARDataset/stratified_scene_only_er2.csv                       scene_only rung
    EARDataset/stratified_random_er2.csv                           random rung
    EARDataset/stratified_baseline_er2.csv                         the 194 sampled tasks + clean re-query

Usage (from the repo root):
    uv run python analysis/rq2/analysis.py [--output_dir analysis/output]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rq1"))
from bypass_rates import STRATUM_ORDER, per_task, read_judged, trial_counts  # shared defs, live in rq1/

TARGETED_CSV   = 'EARDataset/robot_constitution_64_automerge_er2_attack_20.csv'
SCENE_ONLY_CSV = 'EARDataset/stratified_scene_only_er2.csv'
RANDOM_CSV     = 'EARDataset/stratified_random_er2.csv'
BASELINE_CSV   = 'EARDataset/stratified_baseline_er2.csv'   # also the task list (`stratum`, `bypass_rate`)

EQUIV_MARGIN = 0.10          # pre-registered, as in the ER-1.6 noise study
RUNGS = ['targeted', 'scene_only', 'random']
N_BOOT = 10000
BOOT_SEED = 42
ANYFLIP_N = 5                # objects/task the rungs get; targeted is subsampled to this for a fair any-flip
ANYFLIP_DRAWS = 200


def load_arms(targeted_csv, scene_only_csv, random_csv, baseline_csv):
    """Per-task rates for the four conditions on the sampled tasks, plus trial counts."""
    baseline = read_judged(baseline_csv)
    task_ids = baseline['Task ID'].unique()
    targeted = read_judged(targeted_csv)
    targeted = targeted[targeted['Task ID'].isin(task_ids)]      # RQ1 trials for the sampled tasks only
    arms = {'targeted': targeted, 'scene_only': read_judged(scene_only_csv),
            'random': read_judged(random_csv), 'baseline': baseline}
    counts = {k: trial_counts(v) for k, v in arms.items()}
    tasks = {k: per_task(v) for k, v in arms.items()}

    rates = pd.concat([t['rate'].rename(k) for k, t in tasks.items()], axis=1)
    rates['stratum'] = baseline.drop_duplicates('Task ID').set_index('Task ID')['stratum']
    n_before = len(rates)
    rates = rates.dropna(subset=list(arms))
    dropped = n_before - len(rates)
    if dropped:
        print(f"[note] dropped {dropped} task(s) missing a condition; {len(rates)} tasks analysed")
    return rates, counts, tasks, arms, n_before


def mean_diff_ci(diff, conf):
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return float(diff.mean() - h), float(diff.mean() + h)


def rung_vs_baseline(r, rung):
    """Paired one-sided: per-task rung rate > per-task baseline rate."""
    a, b = r[rung], r['baseline']
    diff = a - b
    t, p = stats.ttest_rel(a, b, alternative='greater')
    try:
        w, wp = stats.wilcoxon(a, b, alternative='greater')
    except ValueError:
        w, wp = np.nan, np.nan
    lo, hi = mean_diff_ci(diff, 0.95)
    return {'rung': rung, 'mean_rung': float(a.mean()), 'mean_baseline': float(b.mean()),
            'mean_diff': float(diff.mean()), 'ci95': [lo, hi],
            't': float(t), 'p_one_sided': float(p),
            'wilcoxon_W': None if np.isnan(w) else float(w),
            'wilcoxon_p': None if np.isnan(wp) else float(wp),
            'n_tasks_higher': int((diff > 0).sum())}


def pair_tests(r, a, b):
    """a vs b: two-sided paired t (difference) and paired TOST (equivalence)."""
    x, y = r[a], r[b]
    diff = x - y
    t_two, p_two = stats.ttest_rel(x, y, alternative='two-sided')
    p_tost, _, _ = ttost_paired(x.values, y.values, -EQUIV_MARGIN, EQUIV_MARGIN)
    lo90, hi90 = mean_diff_ci(diff, 0.90)
    return {'a': a, 'b': b, 'mean_a': float(x.mean()), 'mean_b': float(y.mean()),
            'mean_diff': float(diff.mean()), 'ci90': [lo90, hi90],
            't_two_sided': float(t_two), 'p_two_sided': float(p_two),
            'tost_margin': EQUIV_MARGIN, 'p_tost': float(p_tost),
            'tightest_margin_supported': float(max(abs(lo90), abs(hi90)))}


def spearman_boot(r, a, b):
    x, y = r[a].values, r[b].values
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(x)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.integers(0, n, n)
        boot[i] = stats.spearmanr(x[s], y[s])[0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'a': a, 'b': b, 'rho': float(rho), 'p': float(p), 'ci95_boot': [float(lo), float(hi)]}


def any_flip(tasks, arms, task_index):
    """Share of tasks with >=1 bypass. Descriptive only: it grows with the object budget, so
    `targeted` (20/task) is also reported subsampled to ANYFLIP_N to match the rungs."""
    out = {k: {'share': float(t.loc[task_index, 'bypassed_any'].mean()),
               'n_objects_per_task': float(t.loc[task_index, 'n_conclusive'].mean())}
           for k, t in tasks.items()}
    tg = arms['targeted']
    tg = tg[tg['Task ID'].isin(task_index) & tg['Judgement'].isin([0, 1])]
    groups = [g['Judgement'].to_numpy() for _, g in tg.groupby('Task ID')]
    shares = []
    for s in range(ANYFLIP_DRAWS):
        rng = np.random.default_rng(s)
        shares.append(float(np.mean([(g[rng.permutation(len(g))[:ANYFLIP_N]] == 1).any() for g in groups])))
    out['targeted_subsampled'] = {'share': float(np.mean(shares)), 'n_objects_per_task': float(ANYFLIP_N),
                                  'pct5': float(np.percentile(shares, 5)), 'pct95': float(np.percentile(shares, 95)),
                                  'draws': ANYFLIP_DRAWS}
    return out


def per_stratum(r):
    g = r.groupby('stratum')[RUNGS + ['baseline']].mean().reindex(STRATUM_ORDER)
    g['n'] = r.groupby('stratum').size().reindex(STRATUM_ORDER)
    return g


def pct(v):
    return f"{v*100:6.2f}%"


def main():
    ap = argparse.ArgumentParser(description="RQ2: targeted vs scene-only vs random objects, task-level paired tests")
    ap.add_argument('--output_dir', default='analysis/output')
    ap.add_argument('--targeted_csv', default=TARGETED_CSV, help='override (smoke tests)')
    ap.add_argument('--scene_only_csv', default=SCENE_ONLY_CSV, help='override (smoke tests)')
    ap.add_argument('--random_csv', default=RANDOM_CSV, help='override (smoke tests)')
    ap.add_argument('--baseline_csv', default=BASELINE_CSV, help='override (smoke tests)')
    ap.add_argument('--output_json', default='rq2_er2_results.json', help='file name inside --output_dir')
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r, counts, tasks, arms, n_sampled = load_arms(args.targeted_csv, args.scene_only_csv,
                                                  args.random_csv, args.baseline_csv)
    n = len(r)
    res = {'inputs': {'targeted': args.targeted_csv, 'scene_only': args.scene_only_csv,
                      'random': args.random_csv, 'baseline': args.baseline_csv},
           'n_tasks_sampled': int(n_sampled), 'n_tasks': n, 'equiv_margin': EQUIV_MARGIN,
           'trial_counts': counts,
           'mean_rates': {c: float(r[c].mean()) for c in RUNGS + ['baseline']}}

    print("=" * 70)
    print(f"RQ2  TARGETED -> SCENE-ONLY -> RANDOM   (task-level paired, n={n} of {n_sampled} tasks)")
    print("=" * 70)
    for c in RUNGS + ['baseline']:
        tc = counts[c]
        print(f"  {c:10s} mean per-task rate {pct(r[c].mean())}   trials {tc['n_conclusive']} conclusive, "
              f"{tc['n_inconclusive']} inconclusive, {tc['n_unjudged']} unjudged")

    # --- Floor: each rung > baseline -------------------------------------------------
    print("\n" + "-" * 70 + "\nFLOOR  rung > clean re-query   (paired one-sided t, Wilcoxon)")
    res['vs_baseline'] = []
    for rung in RUNGS:
        d = rung_vs_baseline(r, rung)
        res['vs_baseline'].append(d)
        print(f"  {rung:10s} {pct(d['mean_rung'])} vs {pct(d['mean_baseline'])}  diff {d['mean_diff']*100:+.2f}pp "
              f"95% CI [{d['ci95'][0]*100:.2f}, {d['ci95'][1]*100:.2f}]  t={d['t']:.2f} p={d['p_one_sided']:.2e}"
              f"  Wilcoxon p={d['wilcoxon_p']:.2e}")

    # --- Rung pairs: primary + secondary ---------------------------------------------
    pairs = [('random', 'targeted'),       # PRIMARY (H_sem)
             ('scene_only', 'targeted'),   # secondary
             ('scene_only', 'random')]     # secondary
    pair_res = [pair_tests(r, a, b) for a, b in pairs]
    holm_two = multipletests([d['p_two_sided'] for d in pair_res], method='holm')[1]
    holm_tost = multipletests([d['p_tost'] for d in pair_res], method='holm')[1]
    for d, h2, ht in zip(pair_res, holm_two, holm_tost):
        d['p_two_sided_holm'] = float(h2)
        d['p_tost_holm'] = float(ht)
    res['pairs'] = pair_res
    print("\n" + "-" * 70 + f"\nRUNG PAIRS  two-sided paired t (difference) + TOST +/-{EQUIV_MARGIN*100:.0f}pp (equivalence)")
    for i, d in enumerate(pair_res):
        tag = 'PRIMARY  ' if i == 0 else 'secondary'
        verdict = ('EQUIVALENT' if d['p_tost'] < 0.05 else 'not equivalent') + \
                  (', DIFFERENT' if d['p_two_sided'] < 0.05 else ', no difference detected')
        print(f"  [{tag}] {d['a']} vs {d['b']}: {pct(d['mean_a'])} vs {pct(d['mean_b'])}  "
              f"diff {d['mean_diff']*100:+.2f}pp  90% CI [{d['ci90'][0]*100:.2f}, {d['ci90'][1]*100:.2f}]")
        print(f"             two-sided t={d['t_two_sided']:.2f} p={d['p_two_sided']:.3f} (Holm {d['p_two_sided_holm']:.3f})"
              f"   TOST p={d['p_tost']:.2e} (Holm {d['p_tost_holm']:.2e})   tightest margin +/-{d['tightest_margin_supported']*100:.2f}pp")
        print(f"             --> {verdict}")

    # --- Task property: Spearman ------------------------------------------------------
    print("\n" + "-" * 70 + "\nTASK PROPERTY  Spearman rank correlation across tasks (bootstrap 95% CI)")
    res['spearman'] = []
    for a, b in [('targeted', 'scene_only'), ('targeted', 'random'), ('scene_only', 'random')]:
        d = spearman_boot(r, a, b)
        res['spearman'].append(d)
        print(f"  rho({a}, {b}) = {d['rho']:.3f}  p={d['p']:.2e}  CI [{d['ci95_boot'][0]:.3f}, {d['ci95_boot'][1]:.3f}]")

    # --- Any-flip (descriptive; depends on the object budget) --------------------------
    res['any_flip'] = any_flip(tasks, arms, r.index)
    print("\n" + "-" * 70 + "\nTASKS WITH >=1 BYPASS  (descriptive: grows with objects/task)")
    for k, v in res['any_flip'].items():
        extra = (f"  [{v['pct5']*100:.1f}-{v['pct95']*100:.1f} over {v['draws']} draws]"
                 if 'pct5' in v else "")
        print(f"  {k:20s} {v['share']*100:5.1f}%   ({v['n_objects_per_task']:.1f} conclusive trials/task){extra}")

    # --- Strata (viz only) -------------------------------------------------------------
    g = per_stratum(r)
    res['per_stratum'] = {s: {c: float(g.loc[s, c]) for c in RUNGS + ['baseline']} | {'n': int(g.loc[s, 'n'])}
                          for s in STRATUM_ORDER}
    print("\n" + "-" * 70 + "\nPER STRATUM  mean per-task rates (viz only; targeted gradient is by construction)")
    print("  stratum   targeted  scene_only   random   baseline   (n)")
    for s in STRATUM_ORDER:
        print(f"  {s:7s} {g.loc[s,'targeted']*100:9.2f}% {g.loc[s,'scene_only']*100:10.2f}% "
              f"{g.loc[s,'random']*100:8.2f}% {g.loc[s,'baseline']*100:9.2f}%   ({int(g.loc[s,'n'])})")

    out = out_dir / args.output_json
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
