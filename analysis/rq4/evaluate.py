"""
Evaluate any per-task proxy score against ground-truth bypass rates.

The evaluator is column-agnostic: `evaluate()` takes two already-aligned numeric
vectors (proxy score, ground-truth score) and reports how well they line up.
Anything approach-specific (how a proxy produces one score per task) happens
upstream; each approach just emits a CSV with a task-id column and a score
column, then calls this.

Two complementary panels are printed per run:
  1. CONTINUOUS  — how well the proxy ranks the exact bypass RATE
                   (Spearman [primary], Pearson, RMSE), via evaluate().
  2. CLASSIFICATION — how well the proxy DETECTS "task ever bypasses"
                   (ROC-AUC, PR-AUC, precision@k + lift), via
                   evaluate_classification(). The bypass rate is zero-inflated
                   (~2/3 of tasks never bypass), so a proxy with a weak global
                   correlation can still have a trustworthy top-of-ranking; the
                   classification panel is what surfaces that.
The classification panel needs the binary 'bypassed_any' column from
ground_truth.py; it is skipped (not an error) on older ground-truth CSVs.

Usage:
  uv run python analysis/rq4/evaluate.py \
    --input analysis/output/proxy_approach2_scores.csv \
    --score_col score

  # override column / ground-truth names if they differ:
  uv run python analysis/rq4/evaluate.py \
    --input scores.csv --score_col my_score \
    --id_col "Task ID" --gt analysis/output/proxy_ground_truth.csv --gt_col bypass_rate

Ground truth defaults to the ER-2 labels, analysis/output/rq4_er2_ground_truth.csv
(run ground_truth.py first). Pass --gt analysis/output/proxy_ground_truth.csv to score
against the ER-1.6 labels instead.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from argparse import ArgumentParser

GROUND_TRUTH_CSV = 'analysis/output/rq4_er2_ground_truth.csv'


def evaluate(proxy, truth, label=''):
    """Compare two aligned per-task vectors.

    Args:
        proxy: array-like of proxy scores, one per task.
        truth: array-like of ground-truth scores, aligned element-wise to proxy.
        label: label for the printed report.

    Reports Spearman (ranking, primary), Pearson (linear association), and RMSE
    (absolute closeness). Returns the metrics as a dict.
    """
    proxy = np.asarray(proxy, dtype=float)
    truth = np.asarray(truth, dtype=float)

    if proxy.shape != truth.shape:
        raise ValueError(f"proxy and truth must be the same length: {proxy.shape} vs {truth.shape}")

    mask = ~(np.isnan(proxy) | np.isnan(truth))
    proxy, truth = proxy[mask], truth[mask]

    print(f"\n{'=' * 60}")
    print(f"PROXY EVALUATION: {label}")
    print(f"{'=' * 60}")
    print(f"  Tasks evaluated: {len(proxy)}")

    if len(proxy) < 2:
        print("  ERROR: need at least 2 tasks to correlate")
        return None

    # Correlation is undefined when either side has no spread. Say so plainly
    # instead of returning a bare nan (e.g. all-zero ground truth).
    for name, vec in (('proxy', proxy), ('ground truth', truth)):
        if np.ptp(vec) == 0:
            print(f"\n  WARNING: {name} is constant ({vec[0]:.4f} for all "
                  f"{len(vec)} tasks) -> correlation is undefined (nan).")

    spearman_r, spearman_p = stats.spearmanr(proxy, truth)
    pearson_r, pearson_p = stats.pearsonr(proxy, truth)
    rmse = float(np.sqrt(np.mean((proxy - truth) ** 2)))

    print(f"\n  Spearman rho: {spearman_r:.4f} (p={spearman_p:.2e})   [ranking, primary]")
    print(f"  Pearson r:    {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  RMSE:         {rmse:.4f}   [closeness]")

    print(f"\n  Proxy score range:  [{proxy.min():.4f}, {proxy.max():.4f}]")
    print(f"  Ground truth range: [{truth.min():.4f}, {truth.max():.4f}]")

    return {
        'spearman_r': spearman_r,
        'pearson_r': pearson_r,
        'rmse': rmse,
        'n_tasks': int(len(proxy)),
    }


def evaluate_classification(proxy, y_binary, topk=(50, 100), label=''):
    """Score a proxy as a binary DETECTOR of "task ever bypasses".

    Complements evaluate(): where evaluate() asks "does the proxy rank the exact
    bypass RATE?", this asks "does the proxy separate the tasks that break from
    the ones that never do?". The proxy score stays continuous — we only threshold
    the GROUND TRUTH (bypassed_any). Higher proxy score MUST mean more likely to
    bypass; an AUC < 0.5 signals a flipped score, not a failed proxy.

    Args:
        proxy: array-like of proxy scores, one per task (higher = more vulnerable).
        y_binary: array-like of 0/1 ground-truth labels, aligned to proxy.
        topk: iterable of k values for precision@k / lift.
        label: label for the printed report.

    Reports ROC-AUC, PR-AUC (average precision), base rate, and precision@k with
    lift over the base rate. Returns the metrics as a dict (None if not scorable).
    """
    proxy = np.asarray(proxy, dtype=float)
    y = np.asarray(y_binary, dtype=float)

    if proxy.shape != y.shape:
        raise ValueError(f"proxy and labels must be the same length: {proxy.shape} vs {y.shape}")

    mask = ~(np.isnan(proxy) | np.isnan(y))
    proxy, y = proxy[mask], y[mask].astype(int)

    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION (detector): {label}")
    print(f"{'=' * 60}")
    print(f"  Tasks evaluated: {len(proxy)}")

    n_pos = int(y.sum())
    if len(proxy) < 2 or n_pos == 0 or n_pos == len(y):
        print(f"  ERROR: need both classes present (got {n_pos} positive of "
              f"{len(y)}) -> ROC-AUC undefined.")
        return None

    base = n_pos / len(y)
    auc = roc_auc_score(y, proxy)
    ap = average_precision_score(y, proxy)

    print(f"  Positives (bypassed): {n_pos}/{len(y)}   base rate: {base:.4f}")
    print(f"\n  ROC-AUC: {auc:.4f}   [ranking separability; 0.5 = chance]")
    print(f"  PR-AUC:  {ap:.4f}   [avg precision; base rate {base:.3f} = chance]")
    if auc < 0.5:
        print("  NOTE: AUC < 0.5 -> proxy score likely points the WRONG way "
              "(higher should mean more vulnerable).")

    order = np.argsort(-proxy)  # highest proxy score first
    prec_at = {}
    print()
    for k in topk:
        if k > len(proxy):
            print(f"  precision@{k}: n/a (only {len(proxy)} tasks)")
            continue
        prec = float(y[order[:k]].mean())
        lift = prec / base if base > 0 else float('nan')
        prec_at[k] = prec
        print(f"  precision@{k}: {prec:.4f}   (lift vs base: {lift:.2f}x)")

    return {
        'roc_auc': auc,
        'pr_auc': ap,
        'base_rate': base,
        'precision_at_k': prec_at,
        'n_tasks': int(len(proxy)),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Per-task score CSV')
    parser.add_argument('--score_col', type=str, required=True, help='Proxy score column in --input')
    parser.add_argument('--id_col', type=str, default='Task ID', help='Task-id column (both files)')
    parser.add_argument('--gt', type=str, default=GROUND_TRUTH_CSV, help='Ground-truth CSV')
    parser.add_argument('--gt_col', type=str, default='bypass_rate', help='Continuous ground-truth column')
    parser.add_argument('--gt_binary_col', type=str, default='bypassed_any',
                        help='Binary ground-truth column for the classification panel')
    parser.add_argument('--topk', type=str, default='50,100',
                        help='Comma-separated k values for precision@k')
    parser.add_argument('--label', type=str, default='', help='Label for the report')
    args = parser.parse_args()

    scores = pd.read_csv(args.input)
    gt = pd.read_csv(args.gt)
    topk = tuple(int(k) for k in args.topk.split(',') if k.strip())
    has_binary = args.gt_binary_col in gt.columns

    # Rename to fixed internal names so score_col == gt_col (e.g. both
    # 'bypass_rate') doesn't collide into pandas _x/_y suffixes on merge.
    proxy_side = scores[[args.id_col, args.score_col]].rename(columns={args.score_col: '_proxy'})
    truth_cols = [args.id_col, args.gt_col] + ([args.gt_binary_col] if has_binary else [])
    truth_side = gt[truth_cols].rename(
        columns={args.gt_col: '_truth', args.gt_binary_col: '_ybin'})
    merged = proxy_side.merge(truth_side, on=args.id_col, how='inner')

    if len(merged) == 0:
        raise SystemExit(f"No overlapping '{args.id_col}' between {args.input} and {args.gt}")

    n_proxy, n_gt = scores[args.id_col].nunique(), gt[args.id_col].nunique()
    if len(merged) < min(n_proxy, n_gt):
        print(f"WARNING: only {len(merged)} tasks overlap "
              f"(proxy has {n_proxy}, ground truth has {n_gt})")

    label = args.label or f"{args.input}:{args.score_col}"

    # Panel 1: continuous ranking / regression against the bypass RATE.
    evaluate(merged['_proxy'], merged['_truth'], label=label)

    # Panel 2: binary detection against "ever bypassed". Skipped (not failed) if
    # the ground-truth CSV predates the binary column.
    if has_binary:
        evaluate_classification(merged['_proxy'], merged['_ybin'], topk=topk, label=label)
    else:
        print(f"\n[skip] classification panel: no '{args.gt_binary_col}' column in "
              f"{args.gt}. Re-run analysis/rq4/ground_truth.py to add it.")


if __name__ == '__main__':
    main()
