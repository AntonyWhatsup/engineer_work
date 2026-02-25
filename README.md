System Oceny Ryzyka Kredytowego (XAI & Hybrid AI)
Repozytorium zawiera inżynierski projekt systemu scoringowego, który łączy uczenie maszynowe (Random Forest) z rygorystycznymi zasadami polskiego nadzoru finansowego (KNF).

🚀 Jak uruchomić projekt?
1.Instalacja bibliotek:
Upewnij się, że masz zainstalowanego Pythona (zalecany 3.10+). W terminalu wpisz:
pip install -r requirements.txt

2.Uruchomienie aplikacji:
Wpisz poniższą komendę, aby otworzyć interaktywny interfejs w przeglądarce:
streamlit run app.py

3.Trening modelu (opcjonalnie):
Jeśli chcesz odświeżyć model, uruchom plik model_training.ipynb w środowisku Jupyter.

📂 Struktura plików
-- app.py --
- Główny punkt wejścia aplikacji (Dashboard Streamlit).
- Implementacja modułu Legal Gatekeeper (weryfikacja statusu pobytu obcokrajowców).
- Logika Compliance KNF (automatyczne sprawdzanie limitów DSTI i minimum socjalnego).
- Silnik Hybrid AI, łączący matematyczny wynik modelu ML z korektami eksperckimi (typ umowy, Graph Risk).
- Moduł XAI (Explainable AI) generujący wykresy SHAP, które tłumaczą powody podjęcia konkretnej decyzji.

-- model_training.ipynb --
- Skrypt odpowiedzialny za budowę modelu predykcyjnego.
- Przetwarzanie danych (One-Hot Encoding, usuwanie brakujących wartości).
- Trening klasyfikatora Random Forest z balansem klas (class_weight='balanced').
- Eksport gotowego modelu do pliku binarnego.

-- analiza_danych.ipynb --
- Proces EDA (Exploratory Data Analysis) na zbiorze 200,000 rekordów.
- Identyfikacja i usunięcie zjawiska Data Leakage (zmienne typu hardship i settlement).
- Analiza korelacji między dochodem, wynikiem BIK a spłacalnością kredytu.

-- credit_model.pkl --
- Zapisany stan wytrenowanego modelu.
- Przechowuje wagi i strukturę lasu losowego, gotową do natychmiastowego użycia w aplikacji.

-- explaint.txt --
- Dokumentacja techniczna i log zmian (Changelog).
- Szczegółowy opis wszystkich 115 cech zbioru danych (LendingClub).
- Historia wersji projektu (od v1.0.0 do v3.0.0).

-- inj_rozdział_1.docx --
- Pierwszy rozdział pracy dyplomowej ("Teoretyczne aspekty AI w bankowości").
- Opis problematyki "czarnej skrzynki" (Black Box) i metodologii XAI.

-- .gitignore --
- Plik konfiguracyjny dla systemu Git.
- Blokuje wysyłanie na serwer niepotrzebnych plików tymczasowych (__pycache__) oraz ciężkich wirtualnych środowisk (venv).
