"""Machine failure prediction web app (Flask).

Run:  python app.py   then open http://127.0.0.1:5001
The model is loaded from model.joblib, or trained from ai4i2020.csv if needed.
"""
import os
import pandas as pd
from flask import Flask, render_template_string, request

from train_model import load_or_train

app = Flask(__name__)

bundle = load_or_train()
model = bundle['model']
columns = bundle['columns']          # exact feature order used in training

# Allowed ranges for inputs (a little wider than the training data)
FIELDS = {
    'air_temp':     ('Air temperature (K)',      290, 310, '300'),
    'process_temp': ('Process temperature (K)',  300, 320, '310'),
    'rot_speed':    ('Rotational speed (rpm)',   1000, 3000, '1500'),
    'torque':       ('Torque (Nm)',              3, 80, '40'),
    'tool_wear':    ('Tool wear (min)',          0, 260, '100'),
}

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Machine failure check</title>
<style>
  :root { --ink:#1b2430; --muted:#5d6b7a; --line:#cfd8e1; --bg:#f3f6f9;
          --accent:#1f5f8b; --ok:#1d6b45; --bad:#a32828; --panel:#ffffff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
         background:var(--bg); color:var(--ink); line-height:1.5; }
  main { max-width: 34rem; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
  p.lead { margin: 0 0 1.5rem; color: var(--muted); }
  form { background: var(--panel); border:1px solid var(--line); border-radius: 6px;
         padding: 1.25rem; display:grid; gap: 1rem; }
  label { display:grid; gap:.3rem; font-weight:600; font-size:.95rem; }
  label span.range { font-weight:400; color:var(--muted); font-size:.85rem; }
  input, select { font: inherit; padding:.55rem .6rem; border:1px solid var(--line);
         border-radius:4px; background:#fff; color:var(--ink); }
  input:focus-visible, select:focus-visible, button:focus-visible
         { outline:3px solid #8fb9d6; outline-offset:1px; }
  button { font: inherit; font-weight:600; padding:.7rem 1rem; border:0; border-radius:4px;
         background:var(--accent); color:#fff; cursor:pointer; }
  button:hover { background:#184d70; }
  .result { margin-top:1.25rem; padding:1rem 1.25rem; border-radius:6px;
         border:1px solid var(--line); background:var(--panel); border-left-width:6px; }
  .result.fail { border-left-color: var(--bad); }
  .result.safe { border-left-color: var(--ok); }
  .result strong { font-size:1.15rem; }
  .error { margin-top:1rem; color: var(--bad); font-weight:600; }
  small { display:block; margin-top:1.5rem; color:var(--muted); }
</style>
</head>
<body>
<main>
  <h1>Machine failure check</h1>
  <p class="lead">Enter the operating conditions. The model estimates the chance of failure.</p>

  <form method="post" novalidate>
    <label>Product quality
      <select name="type">
        {% for t, name in [('L','Low (L)'),('M','Medium (M)'),('H','High (H)')] %}
        <option value="{{ t }}" {{ 'selected' if form.get('type', 'L') == t }}>{{ name }}</option>
        {% endfor %}
      </select>
    </label>
    {% for key, (label, lo, hi, default) in fields.items() %}
    <label>{{ label }} <span class="range">allowed {{ lo }} to {{ hi }}</span>
      <input name="{{ key }}" inputmode="decimal" required value="{{ form.get(key, default) }}">
    </label>
    {% endfor %}
    <button type="submit">Check machine</button>
  </form>

  {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}

  {% if result %}
  <section class="result {{ 'fail' if result.failure else 'safe' }}" aria-live="polite">
    <strong>{{ 'Failure likely' if result.failure else 'No failure expected' }}</strong>
    <p>Estimated failure probability: {{ '%.1f'|format(result.prob * 100) }}%.
       {% if result.failure %}Plan an inspection for this machine.{% else %}Keep normal monitoring.{% endif %}</p>
  </section>
  {% endif %}

  <small>Demo model trained on the synthetic AI4I 2020 dataset. Not for real maintenance decisions.</small>
</main>
</body>
</html>
"""


def build_row(form):
    """Validate the form and return a one-row DataFrame in training column order."""
    values = {}
    for key, (label, lo, hi, _) in FIELDS.items():
        raw = form.get(key, '').strip()
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f'{label}: enter a number.')
        if not lo <= value <= hi:
            raise ValueError(f'{label}: enter a value between {lo} and {hi}.')
        values[key] = value

    quality = form.get('type', 'L')
    if quality not in ('L', 'M', 'H'):
        raise ValueError('Product quality: choose L, M or H.')

    row = {
        'Air temperature [K]': values['air_temp'],
        'Process temperature [K]': values['process_temp'],
        'Rotational speed [rpm]': values['rot_speed'],
        'Torque [Nm]': values['torque'],
        'Tool wear [min]': values['tool_wear'],
        'Type_L': int(quality == 'L'),   # H is the dropped baseline: both 0
        'Type_M': int(quality == 'M'),
    }
    return pd.DataFrame([row], columns=columns)


@app.route('/', methods=['GET', 'POST'])
def home():
    result, error = None, None
    if request.method == 'POST':
        try:
            X = build_row(request.form)
            prob = float(model.predict_proba(X)[0][1])
            result = {'prob': prob, 'failure': prob >= 0.5}
        except ValueError as exc:
            error = str(exc)
    return render_template_string(PAGE, result=result, error=error,
                                  form=request.form, fields=FIELDS)


if __name__ == '__main__':
    # Port 5001: on macOS, port 5000 is often taken by AirPlay Receiver.
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5001)))
