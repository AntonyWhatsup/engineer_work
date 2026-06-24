# Credit Risk Assessment System (XAI & Hybrid AI)

The repository contains an engineering project of a scoring system that combines machine learning (Random Forest) with strict rules of the Polish Financial Supervision Authority (KNF).

## 🚀 How to run the project?

### 1. Install dependencies:
Ensure you have Python installed (recommended 3.10+). In the terminal, type:
```bash
pip install -r requirements.txt
```

### 2. Run the application:
Type the following command in the main project folder to start the Flask server:
```bash
python run_app.py
```
After starting, open the browser at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### 3. Model analysis and training (optional):
Notebooks are located in the `notebooks/` folder. If you want to refresh the EDA analysis or retrain the model, run the appropriate `.ipynb` files in the Jupyter environment:
- `notebooks/analiza_danych.ipynb` (EDA analysis and data cleaning)
- `notebooks/model_training.ipynb` (Random Forest classifier training process)

---

## 📂 Directory and file structure

- **`notebooks/`**
  - `analiza_danych.ipynb` — EDA (Exploratory Data Analysis) process on a dataset of 200,000 records. Contains Data Leakage elimination and correlation analysis.
  - `model_training.ipynb` — training of a balanced Random Forest model saving weights and columns.
- **`models/`**
  - `credit_model.pkl` — binary file of the trained Random Forest classifier (excluded from repository due to size).
  - `model_columns.pkl` — list of mapped columns (One-Hot Encoding) used for input validation in the web app.
- **`web_app/`**
  - `app.py` — Flask application engine combining Legal Gatekeeper, KNF rules (DSTI), expert weights, and SHAP interpretability.
  - `templates/index.html` — interactive, modern UI template in Dark Mode with an explanation panel.
  - `static/current_shap.png` — generated explanation chart (SHAP Force Plot) for the last prediction.
- **`data/`**
  - `accepted_2007_to_2018Q4.csv` — input LendingClub dataset (1.6 GB, excluded from repository).
- **`run_app.py`** — script to easily run the application directly from the root folder.
- **`requirements.txt`** — list of dependent libraries.
- **`patch_notes.txt`** — project release history (changelog).

