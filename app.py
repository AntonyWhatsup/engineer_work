from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import shap
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# --- ЗАВАНТАЖЕННЯ МОДЕЛІ ---
MODEL_PATH = "credit_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
explainer = shap.TreeExplainer(model) if model else None

def calculate_rata(amount, term, rate=0.12):
    if term == 0: return 0
    m_rate = rate / 12
    return (amount * m_rate) / (1 - (1 + m_rate)**(-term))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Отримуємо всі дані з форми
            citizenship = request.form.get('citizenship', 'PL')
            residency = request.form.get('residency', 'Staly')
            income_m = float(request.form.get('income', 6000))
            loan_amt = float(request.form.get('loan_amount', 30000))
            term = int(request.form.get('term', 36))
            bik = float(request.form.get('bik_score', 50))
            debts_m = float(request.form.get('other_debts', 0))
            dependents = int(request.form.get('dependents', 0))
            emp = request.form.get('emp_type', 'nieokreslony')
        except ValueError:
            return render_template('index.html', error="Błąd: Wprowadź poprawne wartości liczbowe.")

        # --- LEGAL CHECK ---
        if citizenship == "UA" and residency in ["Brak", "Krotki"]:
            return render_template('index.html', error="⛔ ODMOWA: Brak stabilnego statusu pobytu.")

        # --- KNF COMPLIANCE ---
        rata = calculate_rata(loan_amt, term)
        total_monthly_costs = rata + debts_m
        
        dsti = (total_monthly_costs / income_m) * 100 if income_m > 0 else 100
        dsti_limit = 65.0 if income_m > 7500 else 50.0
        
        min_living_required = 2000.0 + (dependents * 1000.0)
        cash_left = income_m - total_monthly_costs

        if dsti > dsti_limit:
            return render_template('index.html', error=f"⛔ ODMOWA: DSTI ({dsti:.1f}%) przekracza limit {dsti_limit}%.")
        if cash_left < min_living_required:
            return render_template('index.html', error=f"⛔ ODMOWA: Zbyt mała kwota na życie ({cash_left:.0f} PLN).")

        # --- ML PREDICTION ---
        annual_inc = income_m * 12
        fico = 300 + (bik * 5.5)
        loan_to_inc = loan_amt / annual_inc if annual_inc > 0 else 0
        
        features = ['annual_inc', 'loan_amnt', 'term', 'fico_range_low', 'dti', 'loan_to_income']
        df_input = pd.DataFrame([[annual_inc, loan_amt, term, fico, dsti, loan_to_inc]], columns=features)
        
        base_prob = model.predict_proba(df_input)[0][1]
        adj = 0.10 if "nieokreslony" in emp else -0.15
        final_prob = max(0.0, min(1.0, base_prob + adj))

        # --- SHAP (XAI) ---
        plt.clf()
        shap_values = explainer.shap_values(df_input)
        
        # Захист від різних версій бібліотеки SHAP
        if isinstance(shap_values, list):
            v = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            v = shap_values[0, :, 1]
        else:
            v = shap_values[0]
            
        fig, ax = plt.subplots(figsize=(8, 4))
        idx = np.argsort(v)
        ax.barh(np.array(features)[idx], np.array(v)[idx], color=['#d9534f' if x < 0 else '#5cb85c' for x in np.array(v)[idx]])
        
        if not os.path.exists('static'): os.makedirs('static')
        plot_path = os.path.join('static', 'current_shap.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # time_stamp змусить браузер завантажити нову картинку
        return render_template('index.html', 
                               success=True,
                               prob=round(final_prob*100, 1),
                               rata=round(rata),
                               left=round(cash_left),
                               time_stamp=time.time())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)