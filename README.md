# Credit Default Risk DSS Prototype

```bash
git clone https://github.com/AntonyWhatsup/engineer_work.git
cd engineer_work
python run_app.py
```

На Linux також можна виконати `./start.sh`, а на Windows — `start.bat`. Обидві оболонки лише передають аргументи єдиному bootstrap-файлу `run_app.py`.

Це навчальна система підтримки рішень (DSS), а не банківська production-система. Вона оцінює `P(Default)` для ML-компонента, але не приймає юридично чинного кредитного рішення. Остаточне рішення та перевірка застосовних норм залишаються за людиною.

## Що робить автоматичний запуск

`run_app.py` визначає корінь репозиторію зі свого розташування, перевіряє Python 3.10+, створює локальну `.venv`, синхронізує `requirements.txt` і перезапускається всередині середовища. Далі він перевіряє policy YAML, model artifact, окремий `metadata.json`, schema, target semantics, versions, threshold, fitted `predict_proba` та однозначне відображення класу `1` на default.

Якщо валідної моделі немає, launcher шукає реальний CSV спочатку в:

```text
data/accepted_loans.csv
```

Для сумісності з поточним набором даних також розпізнається `data/accepted_2007_to_2018Q4.csv`. Якщо CSV є, запускається офіційний training module, artifact повторно валідується, і лише тоді стартує Flask. Пошкоджений або несумісний наявний artifact блокує запуск; його не замінює приховане автотренування. Для свідомого перенавчання використайте `--train`.

Якщо немає ні валідної моделі, ні CSV, launcher завершується з точними шляхами та командою training. Проєкт принципово не створює synthetic production model, `base_prob`, mock prediction чи вигадану `P(Default)`.

## Команди launcher

```bash
python run_app.py --help
python run_app.py --setup-only
python run_app.py --train
python run_app.py --skip-install
python run_app.py --data data/accepted_loans.csv --artifact-dir artifacts
python run_app.py --host 127.0.0.1 --port 5000
python run_app.py --debug
python run_app.py --verbose
```

`debug=False` за замовчуванням. Навіть із `--debug` Flask reloader вимкнений, щоб bootstrap, training і важка ініціалізація не виконувалися двічі. `--skip-install` пропускає автоматичний `pip install`; використовуйте його лише коли середовище вже підготовлене. `--setup-only` готує середовище та перевіряє policy config без training або запуску сервера.

Ручне навчання на Windows:

```powershell
.venv\Scripts\python.exe -m src.train --data data\accepted_loans.csv --output artifacts
```

Linux/macOS:

```bash
.venv/bin/python -m src.train --data data/accepted_loans.csv --output artifacts
```

Training зберігає `artifacts/model.joblib`, `artifacts/metadata.json`, метрики та calibration outputs. Повний LendingClub CSV і generated artifacts не комітяться.

## Семантика та архітектура

```text
target = 1 -> Charged Off / Default
target = 0 -> Fully Paid
predict_proba(...)[1] = P(Default)
```

- `run_app.py` — єдиний bootstrap/orchestrator.
- `src/train.py` і `src/training/` — data preparation, split validation, training, metrics і artifact I/O.
- `src/inference/` — тільки валідоване ML inference.
- `src/validation/`, `src/rules/`, `src/decision/` — input validation, policy rules і hybrid DSS logic.
- `src/explainability/` — lazy SHAP лише для ML default-class prediction.
- `web_app/` — Flask application factory, routes і UI.
- `config/policy_rules.yaml` — versioned demonstration rules, thresholds і risk bands.

`issue_d` перетворюється на datetime до chronological split. Train/validation/test та evaluation/calibration вимагають обидва target-класи. Deployment threshold у metadata мусить збігатися з YAML decision threshold і верхньою межею elevated-risk band.

SHAP імпортується ліниво та серіалізує native initialization/execution між паралельними запитами. Пояснення не пишуть спільний `current_shap.png`. Якщо SHAP недоступний або не підтримує конкретний artifact, UI чесно повертає статус недоступності без вигаданих contributions; inference при цьому залишається справжнім.

## Тести та перевірки

Після першого setup:

```bash
python -m pytest
ruff check .
ruff format --check .
```

Тести не встановлюють залежності з інтернету, не запускають справжній сервер і не потребують повного CSV. Малі synthetic fixtures використовуються лише для unit/smoke-перевірки training та inference, ніколи як production fallback.

Health endpoint: `GET /health` повертає `200 ready` лише для завантаженої сумісної моделі. Діагностичний not-ready factory mode, який не використовується launcher, повертає `503 not-ready`.

Вебінтерфейс має три робочі зони: форму заявки, dashboard із реальною `P(Default)`, risk bands, policy indicators і SHAP-графіком, а також згортаний довідник усіх полів. Кнопка «Заповнити приклад» підставляє документовані тестові значення; вони використовують справжню модель і не є mock prediction.

## Обмеження

LendingClub data може не репрезентувати сучасне польське кредитування; macroeconomic drift та історичні proxy bias можуть погіршувати валідність. Репозиторій не заявляє завершеної fairness validation. Реальне застосування потребувало б юридично погоджених правил, незалежної валідації, моніторингу drift/bias, захисту даних і процедури людського оскарження.
