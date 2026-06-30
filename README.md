# 🧪 ChemGPT Studio

> ⚠️ **Work in progress.** This is a teaching demo and may change rapidly, with breaking changes between commits. It exists to illustrate a *generate → analyze* loop end to end — **for education only**. It is **not** validated and must **not** be used for clinical, regulatory, research, or drug-discovery decisions.

A small Streamlit app that computes a molecule's physicochemical / drug-likeness properties, renders an interactive 3D structure, and can sample brand-new molecules with a tiny chemical language model (ChemGPT) — then analyze those too. Everything analyzed is cached, tagged by source, and exportable to CSV.

---

## What it does

- **Analyze a molecule** from a SMILES string and get real [RDKit](https://www.rdkit.org/) descriptors: molecular weight, calculated LogP, H-bond donors/acceptors, TPSA, rotatable bonds, ring count, plus a Lipinski Rule-of-5 check.
- **View it in 3D** — coordinates are embedded locally with RDKit (no external service) and shown with py3Dmol/stmol.
- **Generate new molecules** with [ChemGPT](https://huggingface.co/ncfrey) (`4.7M` or `19M` params), decode the SELFIES output to SMILES, keep only the chemically valid ones, and analyze any candidate.
- **Track & export**: an in-memory DuckDB table logs every analyzed molecule, tagged by source (`sample` / `custom` / `generated`), with a filter and one-click CSV download.

## Honest scope & limitations

This is intentionally simple, and the UI tries to say so where it matters:

- **Descriptors are calculated, not measured.** LogP is a Crippen *estimate* (cLogP), labelled "LogP (calc.)".
- **Lipinski's Rule of 5 is a crude heuristic** for oral bioavailability. Passing it does **not** mean a molecule is a real, safe, or good drug (ethanol passes with zero violations).
- **The 3D view is one quick low-energy conformer**, illustrative only — not necessarily the bioactive or global-minimum shape.
- **Generation is *de novo*** and **not conditioned on any protein target**. The "Target / sequence" field is a free-text annotation that is logged but does **not** influence the chemistry. Target-conditioned design needs a different, much heavier class of structure-based models.
- **Generated molecules are only checked for valid SMILES** — not novelty, synthesizability, stability, or safety.

## Install

```bash
pip install -r requirements.txt
```

Notes:
- The RDKit package is `rdkit` (the old `rdkit-pypi` is deprecated).
- The ChemGPT generation feature needs `torch`, `transformers`, and `selfies`. They're listed in `requirements.txt` but are **optional** — without them the app still runs as an analyzer/3D viewer and the Generate button is disabled.

## Run

```bash
streamlit run main.py
```

## Usage

1. Under **Molecule source**, pick a built-in **sample** (shown instantly) or choose **Custom SMILES** to type your own and click **Analyze**.
2. Optionally open **Generate** to have ChemGPT sample new molecules, seeded from the current molecule. Pick a candidate and analyze it.
3. Review the **Analysis log** at the bottom, filter by source, and **download the results as CSV**.

The first time you generate, the small model downloads from Hugging Face and is then cached in memory; it runs on CPU.

## Privacy & data

- Descriptor computation and 3D embedding run **locally** — molecules never leave your machine.
- ChemGPT weights are downloaded once from Hugging Face, then inference is local.
- The cache/log lives in an in-memory DuckDB database and is wiped when the app stops.

## Project structure

Currently a single file for convenience, organized into clearly marked blocks (`# models.py`, `# metrics.py`, `# generate.py`, `# main.py`) intended to be split into real modules as it grows.

## Roadmap (subject to change)

- Split the single file into proper modules.
- Optional synthesizability (SA score) and additional descriptors.
- Persisted (on-disk) cache and richer export.
- Clearer separation of "valid sample" vs "novel generation" in the log.
