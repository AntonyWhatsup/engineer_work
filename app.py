import os
import time
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import joblib

matplotlib.use('Agg')

app = Flask(__name__)

# static folder
if not os.path.exists('static'):
    os.makedirs('static')

# ✅ LOAD MODEL (fixed)
model = None
try:
    model = joblib.load('credit_model.pkl')
except Exception as e:
    print(f"Помилка завантаження моделі: {e}")

# --- CONSTANTS ---
RATA_FACTOR = 0.02
DSTI_LIMIT_HIGH = 0.65
DSTI_LIMIT_LOW = 0.50
DSTI_INCOME_THRESHOLD = 7500
MIN_CASH_APPLICANT = 2000
MIN_CASH_DEPENDENT = 1000


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # --- 1. ВАЛІДАЦІЯ ФОРМИ ---
        required_fields = ['loan_amnt', 'annual_inc', 'dti', 'fico']
        missing_fields = []

        for field in required_fields:
            val = request.form.get(field, '').strip()
            if not val:
                missing_fields.append(field)

        # Якщо є порожні обов'язкові поля - повертаємо форму з помилками
        if missing_fields:
            return render_template(
                'index.html', 
                success=False, 
                missing_fields=missing_fields, 
                data=request.form  # Зберігаємо те, що користувач вже ввів
            )

        # --- INPUT (безпечне зчитування після валідації) ---
        data_inputs = {
            'loan_amnt': float(request.form.get('loan_amnt')),
            'term': request.form.get('term', '36m'),
            'citizenship': request.form.get('citizenship', 'UA'),
            'residency': request.form.get('residency', 'Brak'),
            'employment_type': request.form.get('employment_type', 'UoP_Nd'),
            'annual_inc': float(request.form.get('annual_inc')),
            'dti': float(request.form.get('dti')),
            'fico': float(request.form.get('fico')),
            'num_dependents': int(request.form.get('num_dependents', 0) or 0)
        }

        # --- FEATURES ---
        term_num = 36 if '36' in data_inputs['term'] else 60

        loan_to_income = (
            data_inputs['loan_amnt'] / data_inputs['annual_inc']
            if data_inputs['annual_inc'] > 0 else 0
        )

        input_df = pd.DataFrame([{
            'annual_inc': data_inputs['annual_inc'],
            'loan_amnt': data_inputs['loan_amnt'],
            'term': term_num,
            'fico_range_low': data_inputs['fico'],
            'dti': data_inputs['dti'],
            'loan_to_income': loan_to_income
        }])

        # --- 2. ВИРІВНЮВАННЯ КОЛОНОК ДЛЯ МОДЕЛІ ---
        if model and hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
            # Додаємо відсутні колонки (заповнюємо 0) і ставимо їх у правильному порядку
            input_df = input_df.reindex(columns=expected_cols, fill_value=0)

        # --- MODEL ---
        base_prob = 0.85
        if model:
            try:
                base_prob = model.predict_proba(input_df)[0][1]
            except Exception as e:
                print(f"Помилка predict: {e}")

        # --- HYBRID ---
        adj = 0
        emp = data_inputs['employment_type']
        if emp == 'UoP_Nd': adj = 0.10
        elif emp == 'UoP_Cz': adj = -0.05
        elif emp == 'UZ_UdP': adj = -0.15
        elif emp == 'B2B_12m': adj = 0.05
        elif emp == 'B2B_lt12m': adj = -0.25

        final_prob = max(0.01, min(0.99, base_prob + adj))

        # --- CALCULATIONS ---
        monthly_inc = data_inputs['annual_inc'] / 12
        rata = data_inputs['loan_amnt'] * RATA_FACTOR
        total_debt_service = rata + (data_inputs['dti'] / 100 * monthly_inc)
        dsti = (total_debt_service / monthly_inc * 100) if monthly_inc > 0 else 100
        cash_left = monthly_inc - total_debt_service

        # --- RULES ---
        rejections = []

        if data_inputs['citizenship'] == 'UA' and data_inputs['residency'] in ['Brak', 'Krotki']:
            rejections.append("Odmowa: Status rezydencji.")

        limit = DSTI_LIMIT_HIGH if monthly_inc > DSTI_INCOME_THRESHOLD else DSTI_LIMIT_LOW
        if (dsti / 100) > limit:
            rejections.append("Odmowa: DSTI limit.")

        required_cash = MIN_CASH_APPLICANT + (data_inputs['num_dependents'] * MIN_CASH_DEPENDENT)
        if cash_left < required_cash:
            rejections.append("Odmowa: Za mały dochód.")

        final_decision = "NEGATYWNA" if rejections else "POZYTYWNA"

        # --- SHAP ---
        shap_img_path = 'static/current_shap.png'
        if model and hasattr(model, 'estimators_'):
            try:
                plt.clf()
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                    ev = explainer.expected_value[1]
                elif len(shap_values.shape) == 3:  # Новий SHAP (1, 6, 2)
                    sv = shap_values[0, :, 1]
                    ev = explainer.expected_value[1]
                else:
                    sv = shap_values[0]
                    ev = explainer.expected_value

                shap.force_plot(
                    ev,
                    sv,
                    input_df.iloc[0], # Беремо перший рядок (єдиний)
                    matplotlib=True,
                    show=False
                )

                plt.savefig(shap_img_path, bbox_inches='tight', dpi=100)

            except Exception as e:
                print(f"SHAP error: {e}")

        return render_template(
            'index.html',
            success=True,
            data=data_inputs,
            base=round(base_prob*100, 1),
            adj=round(adj*100, 1),
            prob=round(final_prob*100, 1),
            rata=round(rata),
            dsti=round(dsti, 1),
            left=round(cash_left),
            decision=final_decision,
            rejections=rejections,
            time_stamp=time.time()
        )

    return render_template('index.html', success=False)


if __name__ == '__main__':
    app.run(debug=True, port=5000)