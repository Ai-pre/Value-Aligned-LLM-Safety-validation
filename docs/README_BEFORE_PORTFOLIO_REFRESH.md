# Value-Aligned LLM Safety Validation

This repository reproduces and extends the Value Injection Method (VIM) experiment from
**Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights**.

The goal is simple and deliberately controlled:

> If two LLMs are trained with the same base model and the same VIM pipeline, but only the
> human value **Power** is changed, does the safety behavior of the model change?

We test this with a counterfactual-style setting:

- **Power High**: Power = 6.0, all other values = 3.0
- **Power Low**: Power = 1.0, all other values = 3.0

The final finding is not that Power simply makes the model "more unsafe" everywhere.
Instead, increasing Power shifts the type of risk: the model becomes more vulnerable to
control-, manipulation-, dominance-, and system-abuse-oriented harmful behavior.

## Motivation

Value alignment is often treated as an unambiguously positive intervention: align a model
more strongly with human values, and safety should improve. The paper challenges that
intuition by showing that different values can amplify different downstream risks.

This reproduction focuses on one value from Schwartz's human value theory:

**Power**: social status, prestige, and control or dominance over people and resources.

The experiment asks whether emphasizing Power changes the model's response style and
safety profile under harmful prompts. This matters because personalized or value-aligned
LLMs may not merely become "more aligned"; they may become aligned toward a value profile
that increases specific kinds of unsafe behavior.

## Research Question

This project validates the following hypothesis:

> Value alignment does not monotonically improve safety. Emphasizing a specific value can
> increase specific safety risks, even when the overall harmfulness rate changes only
> slightly.

For this reproduction, the target question is:

> Does increasing the Power value make a VIM-trained model more likely to produce
> dominance-, control-, manipulation-, or system-abuse-related harmful outputs?

## Experimental Design

Only one value is manipulated.

| Distribution | Achievement | Benevolence | Conformity | Hedonism | Power | Security | Self-direction | Stimulation | Tradition | Universalism |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `power_low` | 3.0 | 3.0 | 3.0 | 3.0 | 1.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| `power_high` | 3.0 | 3.0 | 3.0 | 3.0 | 6.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |

This setup makes Power the main explanatory variable while holding the rest of the value
profile constant.

## Pipeline

The reproduction follows the paper's VIM flow with three main stages.

```text
Value distribution
        |
        v
ValueEval-based preprocessing
        |
        v
Argument Generation training
        |
        v
Survey Alignment training
        |
        v
Safety inference on BeaverTails and HEx-PHI
        |
        v
GPT-4o / Duo Judge evaluation
        |
        v
High-vs-Low category analysis
```

### 1. Argument Generation Training

The preprocessing step uses `data/extreme_distributions.csv` and `data/valueEval_10.csv`
to split ValueEval examples into positive and negative argument data for each target
distribution.

The argument-generation model is then LoRA-tuned so that it learns what the target value
profile would or would not endorse.

Relevant files:

- `src/preprocessing.py`
- `src/train_argument.py`
- `src/dataset/d_argument.py`
- `scripts/initialize.sh`
- `scripts/train_vim.sh`

### 2. Survey Alignment Training

The survey stage validates whether the trained model actually responds according to the
target value profile. The model is trained to answer ValueEval-style survey questions on
a 1-to-6 scale.

Relevant files:

- `src/train_argument_survey.py`
- `src/dataset/d_survey.py`

Observed Power means from the survey validation:

| Model | Predicted Power Mean |
| --- | ---: |
| `power_low` | ~2.90 |
| `power_high` | ~3.27 |

The absolute values are compressed toward the middle, but the ordering is correct:
`power_high` shows a higher Power tendency than `power_low`.

### 3. Safety Inference

After survey alignment, the two value-aligned models are evaluated on harmful/adversarial
prompts.

Safety datasets:

- `BeaverTails-Evaluation`: harmfulness classification by category
- `HEx-PHI`: risk severity evaluation with Duo Score

Inference uses free-form generation with an instruction-style prompt.

Relevant files:

- `src/inference_VIM.py`
- `src/inference_vanilla.py`
- `src/inference_ICL.py`
- `utils/utils.py`
- `utils/prompt_templates.py`

## Environment

The original README lists Python 3.13, but the reproduction was run with Python 3.11
because several GPU and package dependencies were not stable under Python 3.13.

Final reproduction environment:

| Component | Setting |
| --- | --- |
| Python | 3.11 |
| Base model | `mistralai/Mistral-7B-v0.1` |
| Training | LoRA / PEFT |
| Inference | vLLM |
| Precision | FP16 |
| Batch size | 8 for safety inference |
| Max tokens | 64 for BeaverTails and HEx-PHI |
| Judge model | GPT-4o |

## Model Access Note

The original implementation targets `meta-llama/Llama-2-7b-hf`, but that model is gated
on Hugging Face. During reproduction, access issues with the LLaMA 2 gated repository
blocked the run, so the base model was changed to:

```text
mistralai/Mistral-7B-v0.1
```

The experiment keeps the same VIM logic while changing the base model to avoid delaying
the reproduction.

## Checkpoints and Results

The final safety inference used the Mistral VIM checkpoints:

```text
ckpt/argument_survey/mistral/min_TH_3/
├── power_high_631/epoch_3
└── power_low_631/epoch_3
```

BeaverTails inference outputs:

```text
results/
├── mistral-7b-VIM-beavertails_results_high/
│   └── beavertails_results.json
└── mistral-7b-VIM-beavertails_results_low/
    └── beavertails_results.json
```

GPT-4o evaluation outputs:

```text
results/
├── mistral-7b-VIM-beavertails_results_high/gpt-4o_eval_output/
└── mistral-7b-VIM-beavertails_results_low/gpt-4o_eval_output/
```

## Main Results

### BeaverTails

Overall harmfulness changed only slightly.

| Model | Unsafe Rate |
| --- | ---: |
| Power Low | 0.727 |
| Power High | 0.730 |

The important result is the category-level shift.

Categories where Power High increased risk:

| Category | Delta |
| --- | ---: |
| Terrorism / Organized Crime | +0.100 |
| Financial Crime / Property Crime / Theft | +0.080 |
| Hate Speech / Offensive Language | +0.078 |
| Privacy Violation | +0.042 |
| Adult Content | +0.041 |
| Non-violent Unethical Behavior | +0.040 |

Categories where Power High decreased risk:

| Category | Delta |
| --- | ---: |
| Drug Abuse / Weapons / Banned Substance | -0.090 |
| Controversial Topics / Politics | -0.086 |
| Misinformation regarding Ethics, Laws, and Safety | -0.086 |
| Animal Abuse | -0.071 |

Interpretation:

Power does not meaningfully increase the total harmfulness rate on BeaverTails, but it
does increase risk in categories related to domination, crime, privacy, and coercive
social behavior.

### HEx-PHI

HEx-PHI shows a clearer increase in average harmfulness under Power High.

| Model | Harmfulness Score | Harmfulness Rate |
| --- | ---: | ---: |
| Power Low | 2.43 | 15.9% |
| Power High | 2.66 | 18.9% |

Risk areas where Power High increased average score:

| Category | Delta Avg Score |
| --- | ---: |
| Fraud and Deception | +0.81 |
| Malware | +0.70 |
| Economic Harm | +0.66 |
| Political Manipulation | +0.50 |
| Privacy Violation | +0.46 |

Risk areas where Power High decreased average score:

| Category | Delta Avg Score |
| --- | ---: |
| Physical Harm | -0.43 |
| Hate / Harassment / Violence | -0.24 |
| Illegal Activity | -0.18 |

Interpretation:

The risk profile shifts away from direct physical harm and toward manipulation,
deception, system exploitation, and control-oriented behavior.

## Qualitative Observation

Manual inspection of model generations showed a consistent style difference:

| Model | Observed Response Tendency |
| --- | --- |
| `power_low` | More cautious, more likely to refuse, more norm- or ethics-oriented |
| `power_high` | More assertive, more authority-oriented, more likely to accept the premise of harmful requests |

For safety reasons, this README summarizes qualitative examples instead of reproducing
actionable harmful generations verbatim.

## Comparison with the Paper

The reproduction supports the paper's central claim:

> Emphasizing a value can increase particular types of safety risk.

The strongest match is not at the level of every individual category, but at the broader
mechanism level: value alignment changes the model's behavioral disposition, and this
can amplify specific harmful behaviors.

| Finding | Reproduction Result |
| --- | --- |
| Value injection changes model behavior | Supported |
| Power High produces higher Power tendency in survey validation | Supported |
| Power High monotonically increases all harmfulness | Not supported |
| Power High increases control/manipulation/system-abuse risk | Supported |
| Risk type shifts across categories | Supported |

## How to Run

### 1. Install Dependencies

```bash
conda create -n vim-power python=3.11 -y
conda activate vim-power
pip install -r requirements.txt
```

### 2. Prepare Value Data

```bash
bash scripts/initialize.sh
```

### 3. Train VIM Models

The repository contains the original training scripts. For this reproduction, the base
model was changed to Mistral and only the two Power distributions were used.

```bash
python src/train_argument.py \
  --distribution_name power_high \
  --GPU_NUM 0 \
  --model_name mistral \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --train_base_dir ./data/values \
  --batch_size 8

python src/train_argument_survey.py \
  --distribution_name power_high \
  --GPU_NUM 0 \
  --model_name mistral \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --argument_generation_dir ./data/argument_generation/value_split \
  --batch_size 8
```

Repeat the same commands for `power_low`.

### 4. Run Safety Inference

```bash
python src/inference_VIM.py \
  --dataset_name beavertails \
  --home_directory . \
  --batch_size 8 \
  --max_tokens 64 \
  --peft_path ./ckpt/argument_survey/mistral/min_TH_3/power_high_631/epoch_3 \
  --output_path ./results/mistral-7b-VIM-beavertails_results_high

python src/inference_VIM.py \
  --dataset_name beavertails \
  --home_directory . \
  --batch_size 8 \
  --max_tokens 64 \
  --peft_path ./ckpt/argument_survey/mistral/min_TH_3/power_low_631/epoch_3 \
  --output_path ./results/mistral-7b-VIM-beavertails_results_low
```

Use `--dataset_name HEx-PHI` for HEx-PHI inference.

### 5. Run GPT-4o Safety Evaluation

```bash
python evaluate/gpt4_eval.py \
  --dataset_name beavertails \
  --openai_modelname gpt-4o \
  --target_result_path ./results/mistral-7b-VIM-beavertails_results_high

python evaluate/gpt4_eval.py \
  --dataset_name beavertails \
  --openai_modelname gpt-4o \
  --target_result_path ./results/mistral-7b-VIM-beavertails_results_low
```

## Important Notes

- Keep API keys out of the repository. Use environment variables or a local `.env` file.
- Large checkpoint files should be tracked with Git LFS if they are pushed to GitHub.
- Some scripts in the original reproduction code still contain LLaMA-oriented defaults.
  For the Power experiment, update paths and model names to Mistral before running.
- The safety evaluation contains harmful prompts and model outputs. Handle generated
  result files carefully when making the repository public.

## Repository Structure

```text
.
├── data/                  # ValueEval, BeaverTails, HEx-PHI, and auxiliary datasets
├── evaluate/              # GPT-4o, Duo Judge, Perspective, and bias evaluation scripts
├── scripts/               # Shell entrypoints for training, inference, and evaluation
├── src/                   # VIM preprocessing, training, and inference code
├── utils/                 # Prompt templates, constants, dataset loading, utilities
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{choi-etal-2025-unintended,
    title = "Unintended Harms of Value-Aligned {LLM}s: Psychological and Empirical Insights",
    author = "Choi, Sooyung and Lee, Jaehyeok and Yi, Xiaoyuan and Yao, Jing and Xie, Xing and Bak, JinYeong",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    url = "https://aclanthology.org/2025.acl-long.1532/",
    doi = "10.18653/v1/2025.acl-long.1532"
}
```
