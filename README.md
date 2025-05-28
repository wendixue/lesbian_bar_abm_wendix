# Lesbian Bar ABM

This repository accompanies the final project *"Belonging and Drift: An Agent-Based Model of Lesbian Space Transformation"*. The project explores how strategic orientation and institutional adaptability shape identity dynamics in lesbian bars using an original agent-based model implemented in Mesa.

## Repository Structure

### `codes/`
Contains all model implementation files:
- `app.py` – Launches the Mesa GUI interface to interactively visualize simulation dynamics.
- `agent.py` – Defines the `PersonAgent` and `Bar`class with identity attributes and belonging logic.
- `model.py` – Defines the `LesbianBarABM` model, including bars’ cultural adaptation and agent-bar interactions.
- `batch_run.py` – Runs systematic batch simulations across different `gamma` values and outputs results for analysis.
- `batch_run_results.csv` – The results of batch_run.py.

### `figures/`
Includes all simulation visualizations used in the report, such as agent spatial distributions, QW ratio plots, and effective affinity boxplots.

### `final_report.tex`
Main LaTeX file of the written report, prepared in Overleaf. It contains the full theoretical background, model explanation, experimental results, and discussion.

### `references.bib`
BibTeX file listing all scholarly references cited in the report.

### `batch_run_results.csv`
The results of `batch_run.py`.
