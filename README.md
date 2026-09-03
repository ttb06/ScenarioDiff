# ScenarioDiff

Implementation of our paper **[ScenarioDiff: A Scenario-level Guidance Framework
for Multimodal Time Series Forecasting](https://arxiv.org/abs/2608.17164)**,
which is accepted at **ICDM'26**.

ScenarioDiff combines three cached guidance levels with a conditional diffusion
forecaster:

1. Historical Context Agent: stepwise summaries of observed documents.
2. Scenario Agent: a qualitative description of the forecast horizon.
3. Anchor Guidance Agent: sparse future intervals for Anchor Blended Sampling.

All three agents use `gemini-2.5-flash`. The prompting pipeline removes calendar
dates and timestamps before each model call and represents sequence positions
with relative indices.

## Setup

Use Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository includes the five Time-MMD domains used in the paper:
`Economy`, `Energy`, `Security`, `SocialGood`, and `Traffic`.

## Cached Guidance

Set a Gemini API key and run all three phases for one dataset:

```bash
export GEMINI_API_KEY="..."
python data/run_llm_pipeline.py \
  --root_path Time-MMD \
  --data_path Economy/Economy.csv \
  --seq_len 36 \
  --pred_len 18 \
  --phase 0 \
  --resume
```

Use `--phase 1`, `--phase 2`, or `--phase 3` to run an individual agent.
Outputs are cached under `Time-MMD/textual/<domain>/`.

## Forecasting

Run the complete 15-task horizon suite:

```bash
bash scripts/run_all_horizons_parallel.sh
```

Run one task:

```bash
ROOT_PATH=Time-MMD \
DATA_PATH=Economy/Economy.csv \
CONFIG=economy_36_18.yaml \
SEQ_LEN=36 PRED_LEN=18 TEXT_LEN=36 FREQ=m \
bash scripts/train.sh
```

The default seed is `2025`. Dataset rows and cached guidance rows retain their
original ordering; removing unused domains and metadata does not change the
sampling order or random-number consumption of an experiment.

## Ablations

```bash
bash scripts/run_source_text_perturb_ablation.sh
CHECKPOINT_DIRS="outputs/run_a outputs/run_b" bash scripts/run_noisy_anchor_ablation.sh
```

## Acknowledgements

The implementation builds on CSDI, MCD-TSF, MM-TSF, and the Time-MMD benchmark.
