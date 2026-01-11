# Quantitative strategies on High Frequency Data (Group 1)

This folder contains the Group 1 research notebook(s), the final strategy implementation script, and Quarto sources for the presentation/report.

## Setup (Windows / PowerShell)

From the repo root (this folder):

```powershell
# Create and activate a virtual environment
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once per terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## What to run

### 1) Generate the Group 1 outputs (CSV + equity plots)

This is the reproducible “one button” script used by the Quarto documents.

```powershell
python final_strategy_group1.py
```

It writes:
- `outputs/group1/summary_data1_all_quarters.csv`
- `outputs/group1/data1_<YYYY_Qx>_NQ.png`
- `outputs/group1/data1_<YYYY_Qx>_SP.png`

### 2) Render the slides and report

```powershell
quarto render "presentation_revealjs.qmd"
quarto render "final_report_word.qmd"
```

Quarto is a separate install (not a pip package). Verify with:

```powershell
quarto --version
```

## Repo layout

- `notebooks/group1_data_preparation.ipynb`: data cleaning/prep.
- `notebooks/group1_strategies.ipynb`: research notebook with parameter sweeps and quarter-by-quarter evaluation.
- `final_strategy_group1.py`: final strategy generator (used by Quarto).
- `functions/position_VB.py`: helper used for the volatility-breakout position state machine.
- `outputs/group1/`: generated CSV/plots (ignored by git).

## Data

Input data lives under `data/`. By default, large parquet files are ignored by git via `.gitignore`.

If your assessment requires committing parquet data, remove or adjust the `data/*.parquet` rule in `.gitignore`.

## Notes

- The Quarto sources read all generated figures/tables from `outputs/group1/`.
- Whether rendered deliverables (`*.html`, `*.docx`, `*.pdf`) are committed is controlled by `.gitignore`.