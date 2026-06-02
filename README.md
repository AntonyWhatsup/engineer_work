# System Oceny Ryzyka Kredytowego (XAI & Hybrid AI)

Repozytorium zawiera inżynierski projekt systemu scoringowego, który łączy uczenie maszynowe (Random Forest) z rygorystycznymi zasadami polskiego nadzoru finansowego (KNF).

## 🚀 Jak uruchomić projekt?

### 1. Instalacja bibliotek:
Upewnij się, że masz zainstalowanego Pythona (zalecany 3.10+). W terminalu wpisz:
```bash
pip install -r requirements.txt
```

### 2. Uruchomienie aplikacji:
Wpisz poniższą komendę w głównym folderze projektu, aby uruchomić serwer Flask:
```bash
python run_app.py
```
Po uruchomieniu, otwórz w przeglądarce adres: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### 3. Analiza i trening modelu (opcjonalnie):
Notebooki znajdują się w folderze `notebooks/`. Jeśli chcesz odświeżyć analizę EDA lub przetrenować model, uruchom odpowiednie pliki `.ipynb` w środowisku Jupyter:
- `notebooks/analiza_danych.ipynb` (analiza EDA i czyszczenie danych)
- `notebooks/model_training.ipynb` (proces trenowania klasyfikatora Random Forest)

---

## 📂 Struktura katalogów i plików

- **`notebooks/`**
  - `analiza_danych.ipynb` — proces EDA (Exploratory Data Analysis) na zbiorze 200,000 rekordów. Zawiera eliminację Data Leakage i analizę korelacji.
  - `model_training.ipynb` — trening zbalansowanego modelu Random Forest z zapisem wag i kolumn.
- **`models/`**
  - `credit_model.pkl` — plik binarny wytrenowanego klasyfikatora Random Forest (wykluczony z repozytorium z uwagi na rozmiar).
  - `model_columns.pkl` — lista zmapowanych kolumn (One-Hot Encoding) wykorzystywana przy walidacji wejść w web app.
- **`web_app/`**
  - `app.py` — silnik aplikacji w Flasku łączący Legal Gatekeeper, reguły KNF (DSTI), wagi eksperckie i interpretowalność SHAP.
  - `templates/index.html` — interaktywny, nowoczesny szablon UI w ciemnym motywie (Dark Mode) z panelem objaśnień.
  - `static/current_shap.png` — wygenerowany wykres wyjaśniający (SHAP Force Plot) dla ostatniej predykcji.
- **`data/`**
  - `accepted_2007_to_2018Q4.csv` — wejściowy zbiór danych LendingClub (1.6 GB, wykluczony z repozytorium).
- **`run_app.py`** — skrypt ułatwiający uruchomienie aplikacji directly z głównego folderu.
- **`requirements.txt`** — lista bibliotek zależnych.
- **`patch_notes.txt`** — historia wydań projektu (changelog).
