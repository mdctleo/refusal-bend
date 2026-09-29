# 🤖 Refusals That Bend

Code for *Refusals That Bend: Measuring and Predicting Task Malleability in Embodied VLM Planners*.

An embodied vision-language planner that refuses a task on one image may comply once a single everyday object is added to the scene. We call the fraction of inserted objects that overturn a refusal the task's **malleability**. This repository contains the pipeline that measures it and the predictor that estimates it without querying the planner.

## 📁 Layout

| Path | Contents |
|---|---|
| `src/` | Model clients: planner, object proposer, scene editor, judge |
| `scripts/` | One driver per research question |
| `analysis/rq1` to `rq4` | Statistics and figures for each research question |

## ⚙️ Setup

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # set GENAI_API_KEY and GEMINI_API_KEY (Gemini API keys)
```

The proposer and the scene editor run on a GPU node through SLURM. Every other stage runs locally.

## 🚀 Running

All commands run from the repository root, one stage per call. Each driver prints its stage list when called without arguments.

**RQ1: does an added object bypass the refusal?**

```bash
bash scripts/run_rq1.sh plan baseline         # query the planner on every clean image
bash scripts/run_rq1.sh judge baseline        # label the replies; refused tasks continue
bash scripts/run_rq1.sh propose               # GPU: propose 20 objects per refused task
bash scripts/run_rq1.sh render                # GPU: insert each object into the image
bash scripts/run_rq1.sh plan targeted 8       # query the planner on the edited images, 8 slices
bash scripts/run_rq1.sh judge targeted        # label the replies as bypass or refusal
bash scripts/run_rq1.sh requery               # control: re-query the clean images
bash scripts/run_rq1.sh judge requery
uv run python analysis/rq1/requery_lift.py    # object bypass rate vs the re-query floor
```

**RQ2: must the object be chosen for the task?**

```bash
uv run python analysis/rq2/stratified_sample.py   # draw the stratified task sample
bash scripts/run_rq2.sh propose random            # draw 5 objects per task from a fixed list
bash scripts/run_rq2.sh render random             # GPU: insert each object into the image
bash scripts/run_rq2.sh plan random 4             # query the planner, 4 slices
bash scripts/run_rq2.sh judge random              # label the replies
uv run python analysis/rq2/analysis.py            # compare the rungs to the targeted objects
```

Replace `random` with `scene_only` for the second rung.

**RQ3: what separates malleable tasks from robust ones?**

```bash
bash scripts/run_rq3.sh all   # instruction, scene, and interaction patterns, then constitution proximity
```

**RQ4: can malleability be predicted without the planner?**

```bash
bash scripts/run_rq4.sh ground_truth   # per-task bypass labels from the judged RQ1 runs
bash scripts/run_rq4.sh a1 infer       # GPU: query the proxy VLM; likewise a2, a3, a4
bash scripts/run_rq4.sh a1 score       # turn the raw output into one score per task
bash scripts/run_rq4.sh a5 score       # logistic composite of a1 to a4
bash scripts/run_rq4.sh evaluate       # rank correlation and ROC against the labels
```

## 📖 How to Cite

If you use this code or the findings in your work, please cite the paper:

```bibtex
@article{lin2026refusals,
  title   = {Refusals That Bend: Measuring and Predicting Task Malleability in Embodied {VLM} Planners},
  author  = {Lin, Leo Y. and Kuznetsov, Mikhail and Ozmen, Muslum Ozgur and Celik, Z. Berkay},
  year    = {2026},
  note    = {Under review}
}
```

## LatentHarm Dataset
https://huggingface.co/datasets/mdctleo/LatentHarm
