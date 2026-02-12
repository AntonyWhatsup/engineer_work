import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components

# --- 1. НАЛАШТУВАННЯ СТОРІНКИ ---
st.set_page_config(
    page_title="System Oceny Ryzyka Kredytowego (KNF Compliant)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ВИДАЛЕНО ПРОБЛЕМНИЙ РЯДОК: st.set_option('deprecation.showPyplotGlobalUse', False)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; background-color: #003366; color: white; font-weight: bold;}
    .status-box {padding: 10px; border-radius: 5px; margin-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. ФУНКЦІЇ ---

@st.cache_resource
def load_model():
    if os.path.exists("credit_model.pkl"):
        return joblib.load("credit_model.pkl")
    return None

def calculate_monthly_payment(principal, annual_rate, months):
    if annual_rate == 0: return principal / months
    monthly_rate = annual_rate / 12
    return (principal * monthly_rate) / (1 - (1 + monthly_rate) ** (-months))

model = load_model()
if model:
    # check_additivity=False вимикає зайві перевірки SHAP, щоб не було помилок
    explainer = shap.TreeExplainer(model)

# --- 3. БІЧНА ПАНЕЛЬ ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/poland-circular.png", width=60)
    st.title("Polski Bank AI")
    st.markdown("**Moduł Ryzyka v3.1 (Stable)**")
    
    st.info("""
    **Etapy Weryfikacji:**
    1. 🛂 **Legal Check:** Weryfikacja pobytu.
    2. ⚖️ **KNF Rules:** Limity DSTI.
    3. 🧠 **AI Model:** Scoring.
    """)

# --- 4. ГОЛОВНИЙ ІНТЕРФЕЙС ---
st.title("🇵🇱 Kredyt Hipoteczny/Gotówkowy - System Decyzyjny")
st.markdown("Analiza zgodna z Rekomendacją T (KNF) oraz Explainable AI")
st.divider()

if model is None:
    st.error("Brak modelu! Uruchom najpierw `model_training.ipynb`.")
    st.stop()

# ==========================================
# ВХІДНІ ДАНІ
# ==========================================

col_main1, col_main2 = st.columns([1, 1.2], gap="large")

with col_main1:
    st.subheader("1. Status Prawny i Zatrudnienie")
    
    citizenship = st.radio("Obywatelstwo", ["Polskie", "Ukraińskie/Inne"], horizontal=True)
    
    residency_card = "Brak"
    if citizenship == "Ukraińskie/Inne":
        residency_card = st.selectbox(
            "Status pobytu (Karta Pobytu)", 
            ["Stały Pobyt (Pobyt Stały)", "Czasowy (>12 mies.)", "Czasowy (<12 mies.)", "Brak / Wiza turystyczna"]
        )

    emp_type = st.selectbox(
        "Forma zatrudnienia", 
        ["Umowa o pracę (Czas nieokreślony)", "Umowa o pracę (Czas określony)", "Umowa Zlecenie/Dzieło", "Własna działalność (B2B)"]
    )
    
    if "Własna działalność" in emp_type:
        business_duration = st.number_input("Staż działalności (miesiące)", min_value=0, value=12)
    else:
        business_duration = 24 

    st.subheader("2. Finanse")
    income_pln = st.number_input("Dochód netto miesięcznie (PLN)", min_value=0, value=5500, step=100)
    annual_inc_pln = income_pln * 12
    
    loan_amount_pln = st.number_input("Kwota kredytu (PLN)", min_value=5000, value=30000, step=1000)
    term_months = st.selectbox("Okres (miesiące)", [24, 36, 48, 60, 120], index=1)
    
    other_debts = st.number_input("Inne miesięczne raty (kredyty, limity)", min_value=0, value=500)
    dependents = st.number_input("Liczba osób na utrzymaniu", min_value=0, value=0)

    st.subheader("3. Historia Kredytowa (BIK)")
    bik_score = st.slider("Scoring BIK (punkty 0-100)", 0, 100, 75)
    fico_equivalent = 300 + (bik_score * 5.5)

with col_main2:
    st.subheader("📊 Raport Decyzyjny")
    run_analysis = st.button("🚀 PRZEANALIZUJ WNIOSEK")

    if run_analysis:
        # --- ETAP 1: LEGAL GATEKEEPER ---
        legal_status = True
        legal_msg = []

        if citizenship == "Ukraińskie/Inne":
            if residency_card in ["Brak / Wiza turystyczna", "Czasowy (<12 mies.)"]:
                legal_status = False
                legal_msg.append("⛔ **ODMOWA:** Brak stabilnego prawa pobytu.")
            else:
                legal_msg.append("✅ **Status OK:** Potwierdzono legalność pobytu.")
        else:
             legal_msg.append("✅ **Status OK:** Obywatel RP.")

        if not legal_status:
            st.error("\n".join(legal_msg))
        else:
            st.success("\n".join(legal_msg))
            
            # --- ETAP 2: KNF RULES ---
            knf_status = True
            estimated_installment = calculate_monthly_payment(loan_amount_pln, 0.12, term_months)
            total_debt_cost = estimated_installment + other_debts
            dsti_ratio = (total_debt_cost / income_pln) * 100
            dsti_limit = 65.0 if income_pln > 7000 else 50.0
            
            min_living_cost = 1200 + (dependents * 1000)
            disposable_income = income_pln - total_debt_cost
            
            st.markdown("#### 📉 Analiza Finansowa (Standard KNF)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rata kredytu", f"{estimated_installment:.0f} PLN")
            c2.metric("DSTI (Obciążenie)", f"{dsti_ratio:.1f}%", f"Limit: {dsti_limit}%", 
                      delta_color="normal" if dsti_ratio < dsti_limit else "inverse")
            c3.metric("Zostaje na życie", f"{disposable_income:.0f} PLN", f"Min: {min_living_cost} PLN",
                      delta_color="normal" if disposable_income > min_living_cost else "inverse")

            if dsti_ratio > dsti_limit:
                st.error(f"⛔ **ODMOWA (KNF):** Przekroczono wskaźnik DSTI.")
                knf_status = False
            elif disposable_income < min_living_cost:
                st.error(f"⛔ **ODMOWA (Zdolność):** Brak środków na życie.")
                knf_status = False
            
            if knf_status:
                # --- ETAP 3: AI MODEL ---
                loan_to_income = loan_amount_pln / annual_inc_pln
                feature_names = ['annual_inc', 'loan_amnt', 'term', 'fico_range_low', 'dti', 'loan_to_income']
                input_data = pd.DataFrame([[
                    annual_inc_pln, loan_amount_pln, term_months, fico_equivalent, dsti_ratio, loan_to_income
                ]], columns=feature_names)
                
                base_prob = model.predict_proba(input_data)[0][1]
                
                # --- SCORING ADJUSTMENT ---
                adjustment = 0.0
                adj_reasons = []

                if "Umowa o pracę (Czas nieokreślony)" in emp_type:
                    adjustment += 0.10
                    adj_reasons.append("📈 **Stabilność:** Umowa na czas nieokreślony (+10%)")
                elif "Zlecenie" in emp_type or "Dzieło" in emp_type:
                    adjustment -= 0.15
                    adj_reasons.append("📉 **Ryzyko:** Umowa cywilnoprawna (-15%)")
                elif "Własna działalność" in emp_type:
                    if business_duration < 12:
                        adjustment -= 0.30
                        adj_reasons.append("⛔ **Ryzyko:** Firma < 12 mies. (-30%)")
                    else:
                        adjustment += 0.05
                        adj_reasons.append("📈 **Biznes:** Firma > 12 mies. (+5%)")

                final_prob = max(0.0, min(1.0, base_prob + adjustment))
                
                st.markdown("---")
                st.subheader("🧠 Decyzja Końcowa")
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Wynik Bazowy AI", f"{base_prob:.1%}")
                col_res2.metric("Wynik Skorygowany", f"{final_prob:.1%}", delta=f"{adjustment:+.1%}")
                
                if adj_reasons:
                    for r in adj_reasons: st.write(r)

                if final_prob >= 0.60:
                    st.balloons()
                    st.success("✅ **KREDYT WSTĘPNIE PRZYZNANY**")
                else:
                    st.error("❌ **Wniosek Odrzucony**")

                # --- ETAP 4: XAI (GRAFIK БЕЗ POMYŁEK) ---
                st.markdown("---")
                st.caption("Wyjaśnienie czynników wpływających na decyzję (SHAP):")
                try:
                    shap_vals = explainer.shap_values(input_data)
                    
                    # Безпечне отримання значень
                    if isinstance(shap_vals, list):
                        vals = shap_vals[1][0]
                    else:
                        vals = shap_vals[0]

                    # Створюємо простий і надійний графік
                    fig, ax = plt.subplots(figsize=(8, 4))
                    features = feature_names
                    # Сортуємо для краси
                    indices = np.argsort(vals)
                    sorted_vals = np.array(vals)[indices]
                    sorted_features = np.array(features)[indices]
                    
                    colors = ['red' if x > 0 else 'blue' for x in sorted_vals]
                    ax.barh(sorted_features, sorted_vals, color=colors)
                    ax.set_xlabel("Wpływ na decyzję (Prawo = TAK, Lewo = NIE)")
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Nie udało się wygenerować wykresu: {e}")