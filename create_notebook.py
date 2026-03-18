#!/usr/bin/env python3
import json
import os


def create_kaggle_notebook():
    with open("phase1_dino_ema_train.py", "r", encoding="utf-8") as f:
        train_lines = f.readlines()

    with open("evaluate.py", "r", encoding="utf-8") as f:
        eval_lines = f.readlines()

    with open("submission_mbr.py", "r", encoding="utf-8") as f:
        submission_lines = f.readlines()

    cells = []

    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

    def add_code(lines_or_str):
        if isinstance(lines_or_str, str):
            source = [line + "\n" for line in lines_or_str.split("\n")]
        else:
            source = lines_or_str
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        })

    # ── Cell 1: Title ──────────────────────────────────────────
    add_md(
        "# Deep Past Initiative: DINO EMA Pretraining & Evaluation (Multi-GPU)\n"
        "Full pipeline for Dual-Objective Sequence-Level KD DINO EMA Pretraining "
        "with multi-GPU support via Accelerate, built for direct execution on Kaggle."
    )

    # ── Cell 2: Install dependencies ───────────────────────────
    add_md("## 1. Install Dependencies")
    add_code("!pip install -q transformers sacrebleu kagglehub accelerate bitsandbytes")

    # ── Cell 3: Write training script ──────────────────────────
    add_md(
        "## 2. Write Training Script\n"
        "Write the training logic to `train_script.py` and launch via `accelerate launch` "
        "to avoid Jupyter's forked-process CUDA limitations."
    )
    train_cell = ["%%writefile train_script.py\n"]
    for line in train_lines:
        train_cell.append(line)
    add_code(train_cell)

    # ── Cell 4: Path setup ─────────────────────────────────────
    add_md("## 3. Detect Paths & Download Model")
    setup_code = """\
import os
import kagglehub

# Find competition dataset directory (contains published_texts.csv)
BASE_DATA_DIR = "/kaggle/input/deep-past-initiative-machine-translation"
for root, dirs, files in os.walk("/kaggle/input"):
    if "published_texts.csv" in files:
        BASE_DATA_DIR = root
        break

# Download baseline model via kagglehub
print("Downloading baseline model via kagglehub...")
dataset_path = kagglehub.dataset_download("assiaben/final-byt5")
print(f"Downloaded to: {dataset_path}")

# Locate model folder (contains config.json)
MODEL_PATH = ""
for root, dirs, files in os.walk(dataset_path):
    if "config.json" in files:
        MODEL_PATH = root
        break

if not MODEL_PATH:
    raise FileNotFoundError(f"config.json not found inside {dataset_path}")

OUTPUT_DIR = "/kaggle/working/dino_ema_output"
PUBLISHED_CSV = os.path.join(BASE_DATA_DIR, "published_texts.csv")
TRAIN_CSV = os.path.join(BASE_DATA_DIR, "train.csv")

os.environ["MODEL_PATH"] = MODEL_PATH
os.environ["PUBLISHED_CSV"] = PUBLISHED_CSV
os.environ["OUTPUT_DIR"] = OUTPUT_DIR
os.environ["TRAIN_CSV"] = TRAIN_CSV

print(f"BASE_DATA_DIR : {BASE_DATA_DIR}")
print(f"MODEL_PATH    : {MODEL_PATH}")
print(f"PUBLISHED_CSV : {PUBLISHED_CSV}")
print(f"OUTPUT_DIR    : {OUTPUT_DIR}")
"""
    add_code(setup_code)

    # ── Cell 5: Run training ───────────────────────────────────
    add_md(
        "## 4. Run DINO EMA Pretraining (Multi-GPU)\n"
        "Using `accelerate launch` with T4x2. "
        "Adjust `--num_processes`, `--epochs`, `--batch_size`, `--grad_accum` as needed."
    )
    add_code(
        "!accelerate launch --multi_gpu --num_processes=2 train_script.py \\\n"
        "  --model_path \"$MODEL_PATH\" \\\n"
        "  --data_path \"$PUBLISHED_CSV\" \\\n"
        "  --train_data_path \"$TRAIN_CSV\" \\\n"
        "  --output_dir \"$OUTPUT_DIR\" \\\n"
        "  --epochs 1 \\\n"
        "  --batch_size 2 \\\n"
        "  --grad_accum 8 \\\n"
        "  --use_bf16 true \\\n"
        "  --gradient_checkpointing true \\\n"
        "  --sample_check_every 80 \\\n"
        "  --save_every_steps 1000"
    )

    # ── Cell 6: Evaluation utilities ──────────────────────────
    add_md("## 5. Evaluation Utilities")

    eval_header = (
        "import sacrebleu\nimport pandas as pd\nimport os\nimport torch\nimport sys\n"
        "from torch.utils.data import Dataset, DataLoader\n"
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n"
        "from tqdm.auto import tqdm\n"
        "import numpy as np\n\n"
        "sys.path.append('.')\n"
        "from train_script import OptimizedPreprocessor\n"
    )
    eval_core_lines = [eval_header]
    skip = True
    for line in eval_lines:
        if "class TranslationDataset" in line:
            skip = False
        if "def main():" in line:
            break
        if not skip:
            eval_core_lines.append(line.rstrip())

    add_code("\n".join(eval_core_lines))

    # ── Cell 7: Run evaluation ─────────────────────────────────
    add_md("## 6. Predict & Evaluate")
    eval_run = """\
print("\\n=> Preparing Evaluation Phase...")
TRAIN_CSV = os.environ.get("TRAIN_CSV", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "")

if not os.path.exists(TRAIN_CSV):
    print(f"Warning: {TRAIN_CSV} not found. Skipping evaluation.")
else:
    eval_df = pd.read_csv(TRAIN_CSV)
    eval_df = eval_df.sample(n=min(400, len(eval_df)), random_state=42).reset_index(drop=True)
    dataset = TranslationDataset(eval_df, OptimizedPreprocessor())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\\nEvaluating Original Baseline Model...")
    try:
        res_orig = evaluate_model(MODEL_PATH, dataset, "Original", batch_size=8, max_new_tokens=256, device=device)
    except Exception as e:
        print("Failed to evaluate original model:", e)
        res_orig = None

    print("\\nEvaluating DINO EMA Fine-tuned Model...")
    dino_model_path = os.path.join(OUTPUT_DIR, "final")
    try:
        res_dino = evaluate_model(dino_model_path, dataset, "DINO EMA", batch_size=8, max_new_tokens=256, device=device)
    except Exception as e:
        print("Failed to evaluate DINO model:", e)
        res_dino = None

    if res_orig is not None and res_dino is not None:
        comp = show_comparison(res_orig, res_dino, n_examples=10)
"""
    add_code(eval_run)

    # ── Cell 8: Submission (MBR) ───────────────────────────────
    add_md(
        "## 7. MBR Ensemble Submission\n"
        "DINO-pretrained student (Model A) + mattiaangeli/byt5-akkadian-mbr (Model B)  \n"
        "Cross-model candidate pooling → post-processing → chrF++ MBR → `submission.csv`"
    )
    submission_cell = ["%%writefile submission_script.py\n"]
    for line in submission_lines:
        submission_cell.append(line)
    add_code(submission_cell)

    add_code(
        "!python submission_script.py \\\n"
        "  --dino_model_path \"$OUTPUT_DIR/final/student\" \\\n"
        "  --output_dir \"$OUTPUT_DIR\" \\\n"
        "  --batch_size 2 \\\n"
        "  --num_beams 8 \\\n"
        "  --max_new_tokens 384"
    )

    # ── Write notebook ─────────────────────────────────────────
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_name = "Kaggle_DINO_EMA_Pipeline.ipynb"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Successfully wrote {out_name}")


if __name__ == "__main__":
    create_kaggle_notebook()
