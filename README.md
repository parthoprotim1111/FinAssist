# LLM-Driven Personal Financial Assistant

Educational prototype: a Streamlit app that guides users through a structured dialogue (task, personal context, requirements, preferences) and produces **personalized, justified** financial-style recommendations using a **local** Hugging Face model with optional **LoRA (PEFT)** adapters.

**Disclaimer:** This is not financial advice. It is a coursework demonstration. Do not enter real account numbers or sensitive identifiers.

## Requirements

- Python 3.10+
- NVIDIA GPU recommended; CPU/MPS possible with smaller models and longer latency
- ~8 GB+ VRAM for 4-bit inference of small instruct models (1B–3B class); less if using CPU offload

## Setup

**1. Virtual environment (recommended)** — Create your own venv in the project directory, activate it, then install everything with *that* environment’s `pip`. The repository does not ship a venv; `.venv/` and `venv/` are listed in `.gitignore` so they stay local and are not committed.

```bash
cd assignment-2
python -m venv .venv
```

Activate:

- **Windows (cmd/PowerShell):** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

**2. PyTorch with CUDA (for GPU inference)** — With the venv **activated**, install `torch` and `torchvision` first. Use the official selector and copy the `pip` command for your OS and CUDA version: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). CUDA wheels come from PyTorch’s index, not a generic CPU-only PyPI install.

Example (replace the `--index-url` with the one matching your CUDA toolkit):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**3. Project dependencies** — Still with the venv activated:

```bash
pip install -e .
```

Or: `pip install -r requirements.txt` (after step 2 if you use that file instead of the editable install for the rest of the stack).

## 📁 Project Structure

### Top-level directories

| Directory | Purpose |
|-----------|---------|
| `src/` | Main Python package: dialogue, LLM backends, finance layer, Streamlit app |
| `configs/` | YAML for models, dialogue, evaluation, and training |
| `tests/` | Pytest suite |
| `notebooks/` | Report and exploratory notebooks |
| `scripts/` | CLI utilities (e.g. LoRA training, report assets) |
| `data/` | Training and export datasets (e.g. `data/processed/` JSONL) |
| `reports/` | Generated artifacts (e.g. evaluation CSV) |

### Core modules (`src/`)

| Module | Purpose |
|--------|---------|
| `app/` | Streamlit entry point, pages (Chat, Benchmarks, Data & Exports), UI components |
| `dialogue/` | Dialogue state machine, slot model, guided flow, validation |
| `llm/` | Abstract backend, Hugging Face local inference, mock backend, Jinja prompts |
| `finassist/` | Recommendation schemas, parsing, justification, deterministic calculations |
| `evaluation/` | Fixtures, metrics, batch evaluation CLI |
| `data_collection/` | Consent and anonymized export helpers |
| `utils/` | Project paths, config loading, device and shared helpers |

**Inference:** primary and alternate checkpoints use `HFLocalBackend` (`transformers` + optional PEFT); benchmarks and `python -m evaluation.run_eval` support **mock**, **HF primary**, and **HF alt**.

### System flow

**User input** → **Dialogue engine** → **Slot extraction** → **LLM backend** → **Deterministic calculations** (metrics in prompt) → **Structured recommendation** (schema-validated) → **UI output**

Slot-filling uses the LLM for JSON updates; the recommendation step combines Jinja templates with Python-computed debt metrics before parsing and display.

## Run the app

From the project root (after `pip install -e .`):

```bash
streamlit run src/app/streamlit_app.py
```

**Pages:** Chat (home), Benchmarks (model × technique on fixtures), Data & Exports (consent + anonymized JSON export).

## LoRA fine-tuning (optional)

The project supports **LoRA (PEFT)** for parameter-efficient fine-tuning.  
By default, no adapter is applied (`adapter_path: null`), so the app runs on the base instruction-tuned models.

To enable LoRA:

1. Prepare a dataset (e.g., JSONL) under `data/processed/` (see `configs/training.yaml`).
2. Train an adapter:

```bash
python scripts/train_lora.py --config configs/training.yaml
```

## Evaluation and report assets

```bash
python -m evaluation.run_eval
python scripts/generate_report_assets.py
```

## Data collection

Users may opt in to export **anonymized** session data from the Data & Exports page. No real PII should be used in demos.

## License

Project code: coursework use. Third-party models and datasets are subject to their respective licenses.
