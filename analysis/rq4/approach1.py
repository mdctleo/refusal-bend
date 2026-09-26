"""
Approach 1: proxy trace text features. Two stages.

  --stage infer  (GPU)  Query the local proxy VLM once per task on the CLEAN image,
                        WITH the full 63-rule constitution at T=0, and persist its
                        Pass-1 thinking trace.
                        -> <out_prefix>_approach1_traces.csv
                           Task ID, thinking_trace, yes_lp, no_lp, yes_p, no_p
                        yes_lp/no_lp are single-token logprobs; the proxy prompt asks
                        "Will you reject?", so yes = reject, no = comply. They are
                        captured for free -- A1 mines the trace TEXT, not these.

  --stage score  (CPU)  Mine that trace text into per-task marker features.
                        -> <out_prefix>_approach1_features.csv
                           Task ID + {object_ambiguity, conditional, hedge,
                           self_question, stance_flip} x {_raw, _density} + word_count
                        Restricted to tasks present in <out_prefix>_ground_truth.csv.

Hypothesis: the proxy's deliberation leaks task-intrinsic ambiguity. A task near the
safety boundary produces a trace unsure what the object/action even is ("is it a food
can or...? could be sealed...") and hedges its judgment ("if it's a weapon... unless").
That ambiguity is a property of the (instruction, scene) pair and should transfer to
the target even though proxy and target draw their boundaries differently. Purely
local: 0 target queries.

Every marker family is counted the same way and emitted BOTH as a raw count (_raw,
verbosity-confounded, kept for comparison) and a per-100-words density (_density, the
canonical scored feature). word_count is emitted un-normalized as the verbosity
BASELINE: a density feature carries real signal only if it out-predicts bare length.

The A3/A5 inference paths generate this thinking trace internally but discard it,
keeping only the yes/no logprobs; this module keeps it.

Defaults target ER-2. Paths derive from --out_prefix (see paths()).

Usage:
  uv run python analysis/rq4/approach1.py --stage infer          # GPU
  uv run python analysis/rq4/approach1.py --stage score          # CPU, no GPU

  # Sharded (each shard concats onto the shared CSV under a file lock; NO de-dup,
  # so never re-run a slice that already succeeded):
  uv run python analysis/rq4/approach1.py --stage infer --rang 0,423
  uv run python analysis/rq4/approach1.py --stage infer --rang 423,846

  # Normally driven by: bash scripts/run_rq4.sh a1 infer|score
"""

import sys
import os
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from filelock import FileLock

# NB: `from vlm_2 import VLM` (heavy: pulls in vLLM/torch) is imported lazily inside
# run_infer(), so the 'score' stage runs on CPU without a GPU/vLLM.

DEFAULT_ATTACK_CSV = 'EARDataset/robot_constitution_64_automerge_er2_attack_20.csv'
DEFAULT_BASELINE_CSV = 'EARDataset/dataset_baseline_er2.csv'
DEFAULT_PREFIX = 'analysis/output/rq4_er2'
DEFAULT_VLM = 'Qwen/Qwen3-VL-30B-A3B-Thinking-FP8'
IMAGE_ROOT = 'EARDataset/images'
BATCH_SIZE = 32


def paths(out_prefix):
    """Every file this module reads or writes, derived from one prefix.

    Defaulting the prefix to 'rq4_er2' means the ER-1.6 outputs (analysis/output/
    proxy_*.csv) are unreachable unless explicitly asked for.
    """
    return {
        'traces': f'{out_prefix}_approach1_traces.csv',        # 'infer' output
        'ground_truth': f'{out_prefix}_ground_truth.csv',      # 'score' input
        'features': f'{out_prefix}_approach1_features.csv',    # 'score' output
    }


def run_infer(args):
    """Stage 'infer': capture one clean-image thinking trace per task (GPU)."""
    # Heavy imports (vLLM/torch) live here so the 'score' stage stays GPU-free.
    from vlm_2 import VLM
    from system_prompts_2 import vlm_proxy_system_prompt, vlm_proxy_user_prompt

    out_csv = paths(args.out_prefix)['traces']
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    # Get task list
    attack = pd.read_csv(args.attack_csv)
    task_ids = attack['Task ID'].unique()
    print(f"Total tasks: {len(task_ids)}")

    if args.rang:
        start, end = map(int, args.rang.split(','))
        task_ids = task_ids[start:end]
        print(f"Processing task range [{start}, {end}): {len(task_ids)} tasks")

    # Get image paths and instructions
    baseline = pd.read_csv(args.baseline_csv)
    baseline_lookup = baseline.set_index('Task ID')

    # Build task list with valid images
    tasks = []
    for task_id in task_ids:
        if task_id not in baseline_lookup.index:
            continue
        row = baseline_lookup.loc[task_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        image_path = os.path.join(IMAGE_ROOT, row['Matched Image Path'])
        if not os.path.exists(image_path):
            continue
        tasks.append({
            'task_id': task_id,
            'instruction': row['Instruction'],
            'image_path': image_path,
        })

    print(f"Tasks with valid images: {len(tasks)}")

    # Single pass: clean image, WITH constitution, T=0. Keep the thinking trace.
    system_prompt = vlm_proxy_system_prompt()
    vlm = VLM(
        model_name=args.vlm_name,
        system_prompt=system_prompt,
        max_new_tokens=4096,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=42,
    )

    all_results = []
    for batch_start in tqdm(range(0, len(tasks), args.batch_size), desc="Clean-image trace"):
        batch = tasks[batch_start:batch_start + args.batch_size]
        prompts = [vlm_proxy_user_prompt(instruction=t['instruction']) for t in batch]
        image_paths = [t['image_path'] for t in batch]

        responses = vlm.query_local_vlm_logits(prompts=prompts, image_paths=image_paths)

        for i, task in enumerate(batch):
            resp = responses[i]
            all_results.append({
                'Task ID': task['task_id'],
                'thinking_trace': resp['thinking_trace'],
                'yes_lp': resp['yes_lp'],
                'no_lp': resp['no_lp'],
                'yes_p': resp['yes_p'],
                'no_p': resp['no_p'],
            })

    results_df = pd.DataFrame(all_results)

    lock_path = out_csv + '.lock'
    with FileLock(lock_path):
        if os.path.exists(out_csv) and args.rang:
            existing = pd.read_csv(out_csv)
            results_df = pd.concat([existing, results_df], ignore_index=True)
        results_df.to_csv(out_csv, index=False)

    print(f"\nResults saved to {out_csv}: {len(results_df)} rows")
    n_empty = int((results_df['thinking_trace'].fillna('').str.strip() == '').sum())
    print(f"  Empty thinking traces: {n_empty}/{len(results_df)}")


# ---------------------------------------------------------------------------
# Stage 'score': mine the trace text into per-task marker features.
#
# Marker patterns. Every marker feature is counted the SAME way (standardized):
#   re.findall(pattern, text, re.IGNORECASE) with \b word boundaries, summed
#   over the feature's pattern list. No str.count() substring matching anywhere.
# Stemming policy (Option B): inflectable CONTENT words carry an explicit \w*
# stem so a behavior is caught in every form ("suggest" == "suggested" ==
# "suggestion"); function words, modals, and fixed discourse markers ("if",
# "might", "wait", "hmm", "maybe") stay bare because they don't inflect and a
# \w* there would only add noise ("might" must NOT match "mighty"). The \b
# anchors also fix two old substring bugs: bare "caution" no longer fires inside
# "precaution", and "might" no longer fires inside "mighty".
# ---------------------------------------------------------------------------

# Hedging vocabulary. Content verbs/adjectives stemmed; connectives left bare.
HEDGE_PATTERNS = [
    r'\bhowever\b',
    r'\balternativ\w*\b',                 # alternatively / alternative
    r'\bif you (?:mean|meant|meaning)\b',
    r'\bcould you clarif\w*\b',           # clarify / clarified / clarification
    r'\bsafer\b',
    r'\binstead\b',
    r'\brather\b',
    r'\bperhaps\b',
    r'\bmight\b',                         # modal, bare (excludes "mighty")
    r'\bpossibl\w*\b',                    # possibly / possible
    r'\brecommend\w*\s+against\b',        # recommend(s|ed) against
    r'\bsuggest\w*\b',                    # suggest / suggested / suggestion
    r'\bwould advis\w*\b',               # would advise / advised
    r'\bcaution\w*\b',                    # caution(s|ary) (excludes "precaution")
]

# Conditional-judgment markers. Function words bare; content words stemmed.
CONDITIONAL_PATTERNS = [
    r'\bif\b',
    r'\bunless\b',
    r'\bprovid\w*\s+that\b',              # provided / providing that
    r'\bassum\w*\b',                      # assuming / assume / assumption
    r'\bin case\b',
    r'\bwhen\b[^\n]*?\bsafe\b',           # "when … safe" co-occurrence (one line)
]

# The proxy expressing uncertainty about what the object/action is (modal
# phrases — no inflection, kept bare).
OBJECT_AMBIGUITY_PATTERNS = [
    r'\bis it\b', r'\bis this\b', r'\bcould be\b', r'\bmight be\b',
    r'\bmaybe\b', r'\beither\b', r'\bor is it\b',
]

# The proxy visibly changing course mid-deliberation. Discourse markers bare;
# only the content verb "reconsider" is stemmed.
STANCE_FLIP_PATTERNS = [
    r'\bwait\b', r'\bhmm+\b', r'\breconsider\w*\b',
    r'\bon the other hand\b', r'\blet me (?:re)?think\b',
]

# Column order for the emitted wide CSV (after 'Task ID'). Every marker family
# emits both a raw count (_raw, kept for transparency / the length-confound
# comparison) and a length-normalized density (_density, per 100 words — the
# CANONICAL scored feature, on one ruler across the family). word_count is kept
# on its own as the verbosity baseline / covariate, NOT normalized.
FEATURE_COLUMNS = [
    'object_ambiguity_raw',
    'object_ambiguity_density',
    'conditional_raw',
    'conditional_density',
    'hedge_raw',
    'hedge_density',
    'self_question_raw',
    'self_question_density',
    'stance_flip_raw',
    'stance_flip_density',
    'word_count',
]


def _count(patterns, text):
    """Total boundaried, case-insensitive matches of a feature's pattern list."""
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def extract_features(response_text):
    """Extract the trace text features from a single thinking-trace string.

    Every marker family is counted identically (boundaried, case-insensitive
    re.findall summed over its pattern list) and emitted twice: a raw count and
    a per-100-words density. The density is the canonical per-task score; the
    raw count is retained to expose the length confound. word_count is the
    verbosity baseline. All are continuous per-task scores evaluated
    independently by evaluate.py.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return {c: 0 for c in FEATURE_COLUMNS}

    word_count = len(response_text.split())
    denom = max(word_count, 1)

    object_ambiguity = _count(OBJECT_AMBIGUITY_PATTERNS, response_text)
    conditional = _count(CONDITIONAL_PATTERNS, response_text)
    hedge = _count(HEDGE_PATTERNS, response_text)
    stance_flip = _count(STANCE_FLIP_PATTERNS, response_text)
    self_question = response_text.count('?')

    def density(raw):
        return 100.0 * raw / denom

    return {
        'object_ambiguity_raw': object_ambiguity,
        'object_ambiguity_density': density(object_ambiguity),
        'conditional_raw': conditional,
        'conditional_density': density(conditional),
        'hedge_raw': hedge,
        'hedge_density': density(hedge),
        'self_question_raw': self_question,
        'self_question_density': density(self_question),
        'stance_flip_raw': stance_flip,
        'stance_flip_density': density(stance_flip),
        'word_count': word_count,
    }


def mine_features(out_prefix=DEFAULT_PREFIX, traces_csv=None,
                  ground_truth_csv=None, features_csv=None):
    """Stage 'score': mine the captured traces into the wide feature CSV (CPU).

    Reads only the traces CSV (plus the ground truth, to restrict to labelled
    tasks), so it is safe to call without a VLM/GPU — the A1 analog of Approach
    2/4's collapse_scores(). Individual paths override the prefix when given.
    """
    p = paths(out_prefix)
    traces_csv = traces_csv or p['traces']
    ground_truth_csv = ground_truth_csv or p['ground_truth']
    features_csv = features_csv or p['features']
    os.makedirs(os.path.dirname(features_csv) or '.', exist_ok=True)

    # Restrict to tasks that have ground truth.
    gt = pd.read_csv(ground_truth_csv)
    task_ids_with_gt = set(gt['Task ID'].values)
    print(f"Ground truth tasks: {len(task_ids_with_gt)}")

    # Load proxy clean-image thinking traces.
    traces = pd.read_csv(traces_csv)
    traces = traces[traces['Task ID'].isin(task_ids_with_gt)].copy()
    print(f"Proxy traces for tasks with ground truth: {len(traces)}")

    # Mine features from each trace.
    rows = []
    for _, row in traces.iterrows():
        feats = extract_features(row.get('thinking_trace', ''))
        feats['Task ID'] = row['Task ID']
        rows.append(feats)

    features_df = pd.DataFrame(rows)[['Task ID'] + FEATURE_COLUMNS]
    features_df.to_csv(features_csv, index=False)
    print(f"Wrote {features_csv}: {len(features_df)} rows x {len(FEATURE_COLUMNS)} feature columns")
    print(f"  columns: {FEATURE_COLUMNS}")
    return features_df


def main():
    parser = ArgumentParser()
    parser.add_argument('--stage', choices=['infer', 'score'], default='infer',
                        help="infer: capture one clean-image thinking trace per task "
                             "(GPU). score: mine those traces into the marker feature "
                             "CSV (CPU, no GPU).")
    parser.add_argument('--vlm_name', type=str, default=DEFAULT_VLM)
    parser.add_argument('--attack_csv', type=str, default=DEFAULT_ATTACK_CSV,
                        help='Attack CSV defining the task list')
    parser.add_argument('--baseline_csv', type=str, default=DEFAULT_BASELINE_CSV,
                        help='Baseline CSV supplying image paths + instructions')
    parser.add_argument('--out_prefix', type=str, default=DEFAULT_PREFIX,
                        help='Output path prefix; see paths()')
    parser.add_argument('--rang', type=str, default=None, help='Task index range, format: start,end')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.stage == 'infer':
        run_infer(args)
    else:
        mine_features(args.out_prefix)


if __name__ == '__main__':
    main()
