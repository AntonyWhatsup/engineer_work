import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. KONFIGURACJA STRONY I INTERFEJSU
# ==============================================================================
st.set_page_config(
    page_title="System Scoringowy AI - Zgodność z KNF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Niestandardowe style CSS dla poprawy czytelności interfejsu
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        width: 100%; 
        background-color: #003366; 
        color: white; 
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
    .status-box {padding: 15px; border-radius: 10px; border: 1px solid #ddd;}
    h1, h2, h3 {color: #003366;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. FUNKCJE LOGICZNE I POMOCNICZE
# ==============================================================================

@st.cache_resource
def load_model():
    """Wczytuje wytrenowany model klasyfikatora z pliku pkl."""
    if os.path.exists("credit_model.pkl"):
        return joblib.load("credit_model.pkl")
    return None

def calculate_monthly_payment(principal, annual_rate, months):
    """Oblicza wysokość raty stałej (annuitetowej)."""
    if annual_rate == 0: 
        return principal / months
    monthly_rate = annual_rate / 12
    return (principal * monthly_rate) / (1 - (1 + monthly_rate) ** (-months))

# Inicjalizacja modelu i silnika wyjaśnień (SHAP)
model = load_model()
if model:
    # TreeExplainer jest dedykowany dla modeli opartych na drzewach (np. Random Forest)
    explainer = shap.TreeExplainer(model)

# ==============================================================================
# 3. PANEL BOCZNY (Sidebar) - Informacje o systemie
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/poland-circular.png", width=60)
    st.title("Polski Bank AI")
    st.subheader("Moduł Zarządzania Ryzykiem")
    st.markdown("---")
    
    st.info("""
    **Proces Weryfikacji Wniosku:**
    1. 🛂 **Weryfikacja Prawna:** Sprawdzenie statusu pobytu i obywatelstwa.
    2. ⚖️ **Zgodność z KNF:** Analiza wskaźnika DSTI oraz minimum socjalnego.
    3. 🧠 **Model Scoringowy:** Predykcja prawdopodobieństwa spłaty (ML).
    4. 🔍 **XAI:** Wyjaśnienie decyzji algorytmu.
    """)
    st.caption("Wersja systemu: 3.1.0 (Stable Branch)")

# ==============================================================================
# 4. INTERFEJS GŁÓWNY - Formularz wprowadzania danych
# ==============================================================================
st.title("🇵🇱 Inteligentny System Decyzyjny")
st.markdown("Analiza ryzyka kredytowego zgodna z **Rekomendacją T (KNF)** oraz standardami **Explainable AI (XAI)**")
st.divider()

if model is None:
    st.error("Błąd krytyczny: Nie odnaleziono pliku modelu 'credit_model.pkl'. Proszę wytrenować model.")
    st.stop()

# Układ kolumn dla formularza
col_input1, col_input2 = st.columns([1, 1.2], gap="large")

with col_input1:
    st.subheader("📄 Dane Wnioskodawcy")
    
    # Dane demograficzne i prawne
    citizenship = st.radio("Obywatelstwo", ["Polskie", "Inne"], horizontal=True)
    
    residency_status = "Stały Pobyt"
    if citizenship == "Inne":
        residency_status = st.selectbox(
            "Status pobytu w RP", 
            ["Stały Pobyt", "Czasowy (>12 mies.)", "Czasowy (<12 mies.)", "Brak / Wiza"]
        )

    # Parametry zatrudnienia
    emp_type = st.selectbox(
        "Forma zatrudnienia", 
        ["Umowa o pracę (Czas nieokreślony)", "Umowa o pracę (Czas określony)", 
         "Umowa Zlecenie/Dzieło", "Własna działalność (B2B)"]
    )
    
    biz_duration = 24
    if "Własna działalność" in emp_type:
        biz_duration = st.number_input("Staż działalności (w miesiącach)", min_value=0, value=12)

    st.subheader("💰 Dane Finansowe")
    income_netto = st.number_input("Miesięczny dochód netto (PLN)", min_value=0, value=5500)
    loan_req = st.number_input("Wnioskowana kwota (PLN)", min_value=1000, value=30000)
    loan_term = st.selectbox("Okres kredytowania (miesiące)", [12, 24, 36, 48, 60, 120], index=2)
    
    other_commitments = st.number_input("Suma innych zobowiązań miesięcznych (PLN)", min_value=0, value=500)
    family_members = st.number_input("Liczba osób na utrzymaniu", min_value=0, value=0)

    st.subheader("🏦 Historia Kredytowa")
    bik_score = st.slider("Scoring BIK (0 - 100 pkt)", 0, 100, 75)
    # Konwersja BIK na ekwiwalent FICO używany przez model
    fico_val = 300 + (bik_score * 5.5)

# ==============================================================================
# 5. SILNIK DECYZYJNY I ANALIZA
# ==============================================================================
with col_input2:
    st.subheader("📊 Analiza i Wynik")
    btn_calculate = st.button("🚀 URUCHOM ANALIZĘ RYZYKA")

    if btn_calculate:
        # --- KROK 1: FILTR PRAWNY (Gatekeeper) ---
        is_legal_ok = True
        if citizenship == "Inne" and residency_status in ["Brak / Wiza", "Czasowy (<12 mies.)"]:
            is_legal_ok = False
            st.error("⛔ **NEGATYWNA DECYZJA PRAWNA:** Niestabilny status pobytu wnioskodawcy.")
        else:
            st.success("✅ **WERYFIKACJA PRAWNA:** Pozytywna")

        if is_legal_ok:
            # --- KROK 2: ANALIZA FINANSOWA (Zasady KNF) ---
            monthly_rate_est = calculate_monthly_payment(loan_req, 0.11, loan_term) # Zakładane RRSO 11%
            total_monthly_burden = monthly_rate_est + other_commitments
            
            # Obliczenie wskaźnika DSTI (Debt Service to Income)
            dsti_ratio = (total_monthly_burden / income_netto) * 100
            dsti_limit = 65.0 if income_netto > 7500 else 50.0
            
            # Obliczenie minimum socjalnego (uproszczone)
            social_min = 1300 + (family_members * 900)
            funds_after_loan = income_netto - total_monthly_burden
            
            st.markdown("#### Wskaźniki Zdolności")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prognozowana rata", f"{monthly_rate_est:.2f} zł")
            m2.metric("Wskaźnik DSTI", f"{dsti_ratio:.1f}%", f"Limit: {dsti_limit}%")
            m3.metric("Pozostałe środki", f"{funds_after_loan:.0f} zł", f"Minimum: {social_min} zł")

            # Walidacja limitów KNF
            if dsti_ratio > dsti_limit:
                st.warning("⚠️ **UWAGA:** Przekroczono dopuszczalny limit zadłużenia DSTI.")
                is_fin_ok = False
            elif funds_after_loan < social_min:
                st.warning("⚠️ **UWAGA:** Środki po spłacie raty są niższe niż minimum socjalne.")
                is_fin_ok = False
            else:
                is_fin_ok = True

            # --- KROK 3: MODEL MATEMATYCZNY (AI SCORING) ---
            # Przygotowanie wektora cech dla modelu
            loan_to_income_val = loan_req / (income_netto * 12)
            features = ['annual_inc', 'loan_amnt', 'term', 'fico_range_low', 'dti', 'loan_to_income']
            input_vector = pd.DataFrame([[
                income_netto * 12, loan_req, loan_term, fico_val, dsti_ratio, loan_to_income_val
            ]], columns=features)
            
            # Predykcja bazowa z modelu Random Forest
            prob_ai = model.predict_proba(input_vector)[0][1]
            
            # --- KROK 4: KOREKTA EKSPERCKA (Heurystyka) ---
            adj_score = 0.0
            reasons = []

            if "Umowa o pracę (Czas nieokreślony)" in emp_type:
                adj_score += 0.12
                reasons.append("➕ Stabilna forma zatrudnienia (+12%)")
            elif "Zlecenie" in emp_type:
                adj_score -= 0.10
                reasons.append("➖ Mało stabilna forma zatrudnienia (-10%)")
            
            if "Własna działalność" in emp_type and biz_duration < 12:
                adj_score -= 0.25
                reasons.append("🚨 Krótki staż działalności gospodarczej (-25%)")

            final_score = max(0.0, min(1.0, prob_ai + adj_score))

            # Wyświetlenie decyzji końcowej
            st.markdown("---")
            st.subheader("🧠 Wynik Analizy Scoringowej")
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Scoring Bazowy AI", f"{prob_ai:.1%}")
            res_col2.metric("Scoring Skorygowany", f"{final_score:.1%}", delta=f"{adj_score:+.1%}")

            for r in reasons:
                st.caption(r)

            if final_score >= 0.65 and is_fin_ok:
                st.success("✅ **DECYZJA: KREDYT PRZYZNANY (WSTĘPNIE)**")
                st.balloons()
            else:
                st.error("❌ **DECYZJA: WNIOSEK ODRZUCONY**")
                if not is_fin_ok:
                    st.info("Powód: Brak wystarczającej zdolności kredytowej wg wytycznych KNF.")

            # --- KROK 5: WYJAŚNIALNOŚĆ MODELU (XAI - SHAP) ---
            st.markdown("---")
            st.subheader("🔍 Interpretacja Decyzji AI")
            st.write("Wykres pokazuje wpływ poszczególnych cech na końcową ocenę modelu.")
            
            try:
                # Obliczanie wartości SHAP dla wprowadzonego wniosku
                s_values = explainer.shap_values(input_vector)
                
                # Obsługa różnych formatów wyjścia SHAP (zależnie od wersji biblioteki)
                if isinstance(s_values, list):
                    contribution = s_values[1][0] # Klasa pozytywna
                else:
                    contribution = s_values[0]

                # Generowanie wykresu słupkowego wpływu cech
                fig_xai, ax_xai = plt.subplots(figsize=(10, 5))
                y_pos = np.arange(len(features))
                
                # Sortowanie dla lepszej wizualizacji
                sort_idx = np.argsort(contribution)
                
                colors = ['#d9534f' if x < 0 else '#5cb85c' for x in contribution[sort_idx]]
                ax_xai.barh(y_pos, contribution[sort_idx], color=colors)
                ax_xai.set_yticks(y_pos)
                ax_xai.set_yticklabels(np.array(features)[sort_idx])
                ax_xai.set_xlabel("Siła wpływu na decyzję (SHAP Value)")
                ax_xai.axvline(0, color='black', lw=0.8)
                
                st.pyplot(fig_xai)
                st.caption("Legenda: Kolor zielony wzmacnia szansę na kredyt, kolor czerwony ją obniża.")

            except Exception as e:
                st.warning(f"Moduł wyjaśnień SHAP jest chwilowo niedostępny: {e}")

# ==============================================================================
# STOPKA
# ==============================================================================
st.divider()
st.caption("System wsparcia decyzji kredytowych. Projekt Inżynierski © 2024")