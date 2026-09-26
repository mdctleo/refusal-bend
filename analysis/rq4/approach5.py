"""
Approach 5: Logistic composite of Approaches 1-4.

Fuses the per-task proxy signals from Approaches 1-4 into ONE malleability score
with an L2-regularized logistic regression, then emits a wide CSV that is scored by
evaluate.py exactly like every other approach (Spearman vs bypass_rate,
ROC/PR-AUC + precision@k vs bypassed_any). Zero target queries, no GPU: this reads
only the already-computed per-approach score CSVs and the ground truth.

Why a fitted model needs OUT-OF-FOLD scoring
--------------------------------------------
Approaches 1-4 are UNFIT raw scores. If we fit the logistic on all 697 tasks and
then scored those same 697, the composite would enjoy an in-sample advantage the
single approaches never had -- the comparison would be rigged. So every task's
composite score is an OUT-OF-FOLD prediction: it comes only from folds that did NOT
contain that task.

  RepeatedStratifiedKFold(5 folds x 5 repeats) stratified on bypassed_any (34% base
  rate). Inside each of the 25 fits: StandardScaler is fit on the TRAIN fold only and
  applied to the held-out fold (fitting the scaler on all data would leak scale
  information across the split), then LogisticRegression(penalty='l2', C=1.0). Each
  repeat yields one held-out probability per task; we AVERAGE the 5 repeats so the
  score vector is stable to the luck of the fold split. That averaged out-of-fold
  probability is the task's composite score.

predict_proba (a probability in (0,1)) is the emitted score. It is a strictly
monotone transform of the logistic linear predictor (log-odds), so Spearman / ROC /
precision@k are identical whichever we use; we emit the probability for a bounded,
readable score column. evaluate.py is then run ONCE over the assembled 697-row
out-of-fold vector (the full ranking is needed together for precision@k / ROC) -- not
per fold.

Feature sets (all standardized inside folds)
--------------------------------------------
  curated  : the semantically-motivated signals --
             object_ambiguity_density, conditional_density, hedge_density (A1, the
             length-controlled survivors), word_count (A1 verbosity covariate),
             std_shift (A2), constitution_reliance (A3), std_comply (A4).   [7]
  full     : curated + self_question_density, stance_flip_density (the two A1
             features the partial-correlation test flagged as verbosity-in-disguise).
             If curated >= full, that corroborates dropping them.            [9]
  curated_nolen : curated minus word_count -- the ablation that shows how much of the
             composite is carried by plain deliberation length.             [6]

All three are emitted as columns (score_curated / score_full / score_curated_nolen)
so each can be evaluated on its own. The full-data coefficient table + feature
correlation matrix are printed for interpretation ONLY (never used for scoring).

A5 has ONLY a score stage -- it never queries a model, so there is nothing to prepare
and nothing to infer. Defaults target ER-2; every input and the output derive from
--out_prefix (see paths()).

Output: <out_prefix>_approach5_scores.csv
  Columns: Task ID, score_curated, score_full, score_curated_nolen

Usage:
  bash scripts/run_rq4.sh ground_truth            # labels (run first)
  # the A1-A4 score stages must have run at the same --out_prefix
  uv run python analysis/rq4/approach5.py --stage score
  # then evaluate each composite column, same as any approach:
  uv run python analysis/rq4/evaluate.py \
    --input analysis/output/rq4_er2_approach5_scores.csv --score_col score_curated_nolen

  # Normally driven by: bash scripts/run_rq4.sh a5 score
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

DEFAULT_PREFIX = 'analysis/output/rq4_er2'


def paths(out_prefix):
    """Every file this module reads or writes, derived from one prefix."""
    return {
        'ground_truth': f'{out_prefix}_ground_truth.csv',
        'a1': f'{out_prefix}_approach1_features.csv',
        'a2': f'{out_prefix}_approach2_scores.csv',
        'a3': f'{out_prefix}_approach3_scores.csv',
        'a4': f'{out_prefix}_approach4_scores.csv',
        'out': f'{out_prefix}_approach5_scores.csv',
    }


ID_COL = 'Task ID'
LABEL_COL = 'bypassed_any'   # binary training target (34% base rate)

N_SPLITS = 5
N_REPEATS = 5
C_REG = 1.0
RANDOM_STATE = 42

# The A1 signals that survived the length-controlled (partial-correlation) test.
A1_CURATED = ['object_ambiguity_density', 'conditional_density', 'hedge_density']
# The two A1 signals flagged as verbosity-in-disguise (added only in `full`).
A1_EXTRA = ['self_question_density', 'stance_flip_density']

FEATURE_SETS = {
    # semantically-motivated composite (the headline)
    'score_curated': A1_CURATED + ['word_count', 'std_shift',
                                   'constitution_reliance', 'std_comply'],
    # + the two dropped A1 features, to show they add nothing
    'score_full': A1_CURATED + A1_EXTRA + ['word_count', 'std_shift',
                                           'constitution_reliance', 'std_comply'],
    # curated minus length, to size the plain-verbosity contribution
    'score_curated_nolen': A1_CURATED + ['std_shift', 'constitution_reliance',
                                         'std_comply'],
}


def load_merged(out_prefix=DEFAULT_PREFIX):
    """Join the four approach score CSVs + ground truth on Task ID -> one frame.

    Every approach CSV is one row per task on the same Task IDs; an inner join
    keeps all of them. Returns (merged_df, n_tasks).
    """
    p = paths(out_prefix)
    gt = pd.read_csv(p['ground_truth'])
    a1 = pd.read_csv(p['a1'])
    a2 = pd.read_csv(p['a2'])
    a3 = pd.read_csv(p['a3'])
    a4 = pd.read_csv(p['a4'])

    a1_cols = [ID_COL] + sorted(set(A1_CURATED + A1_EXTRA + ['word_count']))
    merged = (gt[[ID_COL, 'bypass_rate', LABEL_COL]]
              .merge(a1[a1_cols], on=ID_COL, how='inner')
              .merge(a2[[ID_COL, 'std_shift']], on=ID_COL, how='inner')
              .merge(a3[[ID_COL, 'constitution_reliance']], on=ID_COL, how='inner')
              .merge(a4[[ID_COL, 'std_comply']], on=ID_COL, how='inner'))

    all_feats = sorted({f for feats in FEATURE_SETS.values() for f in feats})
    n_nan = int(merged[all_feats].isna().sum().sum())
    if n_nan:
        raise ValueError(f"{n_nan} NaN(s) in feature columns after merge; "
                         "expected none. Check the upstream score CSVs.")
    print(f"Merged {len(merged)} tasks x {len(all_feats)} distinct features "
          f"(base rate {merged[LABEL_COL].mean():.4f})")
    return merged, len(merged)


def oof_predictions(X, y, feature_names):
    """Averaged out-of-fold logistic probabilities for one feature set.

    RepeatedStratifiedKFold(N_SPLITS x N_REPEATS): scaler fit on TRAIN only and
    applied to the held-out fold (no leakage), L2 logistic (C=C_REG). Each repeat
    covers every task exactly once as a held-out sample; we sum each task's held-out
    probabilities and divide by the number of repeats -> a stable averaged OOF score.
    Also returns the per-fold standardized coefficients (for sign-stability reporting).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)

    oof_sum = np.zeros(n)
    oof_count = np.zeros(n)
    fold_coefs = []

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)
    for train_idx, val_idx in rskf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        # penalty defaults to L2; C=C_REG sets its strength (setting penalty='l2'
        # explicitly is deprecated in sklearn >=1.8, so we leave it at the default).
        clf = LogisticRegression(C=C_REG, solver='lbfgs', max_iter=1000)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        oof_sum[val_idx] += clf.predict_proba(scaler.transform(X[val_idx]))[:, 1]
        oof_count[val_idx] += 1
        fold_coefs.append(clf.coef_[0])

    if not np.all(oof_count == N_REPEATS):
        raise AssertionError("every task should be held out exactly N_REPEATS times")
    oof = oof_sum / oof_count

    fold_coefs = np.vstack(fold_coefs)
    return oof, fold_coefs


def fit_full_coefficients(X, y, feature_names):
    """Standardized logistic coefficients fit on ALL tasks -- interpretation only.

    Used purely to report which approaches carry weight and their sign; this model
    is NEVER used to score tasks (that is the out-of-fold job).
    """
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C_REG, solver='lbfgs', max_iter=1000)
    clf.fit(scaler.transform(X), y)
    return dict(zip(feature_names, clf.coef_[0]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--stage', choices=['score'], default='score',
                        help="score: fit the out-of-fold composite over the A1-A4 score "
                             "CSVs (CPU, no queries). A5 has no prepare/infer stage -- it "
                             "never touches a model.")
    parser.add_argument('--out_prefix', type=str, default=DEFAULT_PREFIX,
                        help='Path prefix for the A1-A4 inputs and the A5 output')
    args = parser.parse_args()

    output_csv = paths(args.out_prefix)['out']
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    merged, n = load_merged(args.out_prefix)
    y = merged[LABEL_COL].values

    out = merged[[ID_COL]].copy()

    for score_col, feats in FEATURE_SETS.items():
        X = merged[feats].values
        oof, fold_coefs = oof_predictions(X, y, feats)
        out[score_col] = oof

        full_coef = fit_full_coefficients(X, y, feats)
        # sign stability across the 25 folds: fraction agreeing with the full-fit sign
        full_signs = np.sign([full_coef[f] for f in feats])
        sign_stab = (np.sign(fold_coefs) == full_signs).mean(axis=0)

        print(f"\n--- {score_col} ({len(feats)} features) ---")
        print(f"  OOF prob range [{oof.min():.4f}, {oof.max():.4f}], "
              f"mean {oof.mean():.4f}")
        print(f"  {'feature':<26} {'std_coef':>9} {'sign_stab':>10}")
        for f, stab in sorted(zip(feats, sign_stab),
                              key=lambda t: -abs(full_coef[t[0]])):
            print(f"  {f:<26} {full_coef[f]:>+9.4f} {stab:>10.2f}")

    # Feature correlation matrix (Spearman) over the full curated set -- shows whether
    # the approaches are complementary or redundant. Interpretation only.
    corr_feats = FEATURE_SETS['score_curated']
    print(f"\n--- Spearman correlation among curated features ---")
    corr = merged[corr_feats].corr(method='spearman')
    with pd.option_context('display.width', 120,
                           'display.float_format', lambda v: f"{v:+.2f}"):
        print(corr.to_string())

    out.to_csv(output_csv, index=False)
    print(f"\nWrote {output_csv}: {len(out)} rows x "
          f"{len(FEATURE_SETS)} composite score columns")
    print(f"  columns: {list(FEATURE_SETS.keys())}")
    print("\nEvaluate each with, e.g.:")
    print(f"  uv run python analysis/rq4/evaluate.py "
          f"--input {output_csv} --score_col score_curated")


if __name__ == '__main__':
    main()
