import gradio as gr
import pandas as pd
import numpy as np
import joblib
from fastapi import Request
from fastapi.responses import JSONResponse
import plotly.graph_objects as go
import warnings
import os
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

print("🚀 Loading models and data...")

rf     = joblib.load('models/rf_model.pkl')
knn    = joblib.load('models/knn_model.pkl')
lr     = joblib.load('models/lr_model.pkl')
xgb    = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

df = pd.read_csv('data/indian_roads_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df['festival'] = df['festival'].fillna('none')

stats = {
    'total_accidents': len(df),
    'avg_risk': df['risk_score'].mean(),
    'fatal_count': (df['accident_severity'] == 'fatal').sum(),
    'fatal_rate': (df['accident_severity'] == 'fatal').mean() * 100,
    'hourly_risk': df.groupby('hour')['risk_score'].mean().to_dict(),
    'city_risk': df.groupby('city')['risk_score'].mean().sort_values(ascending=False).to_dict(),
    'weather_risk': df.groupby('weather')['risk_score'].mean().to_dict(),
}

if GEMINI_AVAILABLE:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_CONTEXT = """
You are an expert road safety analyst AI assistant for an Indian Road Accident Risk Analyzer.
The dataset contains 20,000 accident records from 8 Indian cities.
Key findings:
- Random Forest achieved 69% accuracy for severity prediction
- Fatal accidents are most common during fog and low visibility
- Peak hours (8-10 AM, 5-8 PM) and weekends show higher risk scores
- Festival periods (Diwali, Holi, Eid) show elevated accident rates
Answer questions about road safety, accident patterns, and this project.
Keep answers concise and helpful.
"""

def chat_with_gemini(message):
    if not GEMINI_AVAILABLE:
        return "Chat feature is currently unavailable."
    try:
        prompt = f"{SYSTEM_CONTEXT}\n\nUser: {message}\nAssistant:"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, could not process that. Error: {str(e)}"

city_map     = {'Mumbai':0,'Pune':1,'Delhi':2,'Chennai':3,'Bangalore':4,'Hyderabad':5,'Kolkata':6,'Chandigarh':7}
weather_map  = {'Clear':0,'Rain':1,'Fog':2}
road_map     = {'Highway':0,'Urban':1,'Rural':2}
vis_map      = {'Low':0,'Medium':1,'High':2}
traffic_map  = {'Low':0,'Medium':1,'High':2}
day_map      = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
festival_map = {'None':0,'Diwali':1,'Holi':2,'Eid':3,'New Year':4}
cause_map    = {'Weather':0,'Overspeeding':1,'Driver Error':2,'Road Condition':3}

def predict_risk(city, weather, road_type, visibility, traffic_density,
                 hour, is_weekend, is_peak_hour, temperature, lanes,
                 traffic_signal, vehicles_involved, casualties,
                 risk_score, day_of_week, festival, cause, model_choice):
    try:
        features = np.array([[
            hour, int(is_weekend), int(is_peak_hour), temperature,
            lanes, int(traffic_signal), vehicles_involved, casualties,
            risk_score, 6,
            weather_map.get(weather, 0), road_map.get(road_type, 0),
            vis_map.get(visibility, 0), traffic_map.get(traffic_density, 0),
            festival_map.get(festival, 0), cause_map.get(cause, 0),
            day_map.get(day_of_week, 0), city_map.get(city, 0), 0
        ]])

        if model_choice == 'Random Forest':
            pred = rf.predict(features)[0]; proba = rf.predict_proba(features)[0]
        elif model_choice == 'XGBoost':
            pred = xgb.predict(features)[0]; proba = xgb.predict_proba(features)[0]
        elif model_choice == 'KNN':
            fs = scaler.transform(features); pred = knn.predict(fs)[0]; proba = knn.predict_proba(fs)[0]
        else:
            fs = scaler.transform(features); pred = lr.predict(fs)[0]; proba = lr.predict_proba(fs)[0]

        risk_val = proba[0]*20 + proba[1]*50 + proba[2]*100

        if risk_val < 30:
            lvl, col, icon, advice = "LOW RISK", "#10b981", "🟢", "Normal conditions. Standard safety protocols sufficient."
        elif risk_val < 60:
            lvl, col, icon, advice = "MEDIUM RISK", "#f59e0b", "🟡", "Exercise caution. Reduce speed and stay alert."
        else:
            lvl, col, icon, advice = "HIGH RISK", "#ef4444", "🔴", "Critical risk! Avoid travel if possible."

        sev = {0:'Minor', 1:'Major', 2:'Fatal'}

        return f"""
<div style="font-family:'Segoe UI',sans-serif;padding:24px;background:#ffffff;
  border-radius:14px;border:1px solid #e5e7eb;margin-top:12px">

  <div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;
    padding:16px;background:#f9fafb;border-radius:10px;border:1px solid #e5e7eb">
    <div style="font-size:48px">{icon}</div>
    <div>
      <div style="font-size:26px;font-weight:700;color:{col}">{lvl}</div>
      <div style="font-size:14px;color:#6b7280;margin-top:4px">
        Risk Score: {risk_val:.1f}/100 &nbsp;|&nbsp; Predicted: {sev[int(pred)]} Accident
      </div>
    </div>
  </div>

  <div style="margin-bottom:20px">
    <div style="font-size:15px;font-weight:600;color:#111827;margin-bottom:14px">
      Severity Probability Breakdown
    </div>
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px">
        <span style="font-size:14px;color:#374151;font-weight:500">🟢 Minor</span>
        <span style="font-size:14px;font-weight:700;color:#10b981">{proba[0]*100:.1f}%</span>
      </div>
      <div style="background:#e5e7eb;height:10px;border-radius:5px">
        <div style="background:#10b981;width:{proba[0]*100:.1f}%;height:10px;border-radius:5px"></div>
      </div>
    </div>
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px">
        <span style="font-size:14px;color:#374151;font-weight:500">🟡 Major</span>
        <span style="font-size:14px;font-weight:700;color:#f59e0b">{proba[1]*100:.1f}%</span>
      </div>
      <div style="background:#e5e7eb;height:10px;border-radius:5px">
        <div style="background:#f59e0b;width:{proba[1]*100:.1f}%;height:10px;border-radius:5px"></div>
      </div>
    </div>
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:5px">
        <span style="font-size:14px;color:#374151;font-weight:500">🔴 Fatal</span>
        <span style="font-size:14px;font-weight:700;color:#ef4444">{proba[2]*100:.1f}%</span>
      </div>
      <div style="background:#e5e7eb;height:10px;border-radius:5px">
        <div style="background:#ef4444;width:{proba[2]*100:.1f}%;height:10px;border-radius:5px"></div>
      </div>
    </div>
  </div>

  <div style="padding:16px;border-radius:10px;background:#f0f9ff;
    border-left:5px solid {col}">
    <span style="font-size:14px;font-weight:600;color:#111827">📋 Recommendation: </span>
    <span style="font-size:14px;color:#374151">{advice}</span>
  </div>

  <div style="margin-top:12px;font-size:12px;color:#9ca3af;text-align:right">
    Model: {model_choice} &nbsp;|&nbsp; Confidence: {max(proba)*100:.1f}%
  </div>
</div>"""

    except Exception as e:
        return f"<div style='padding:20px;background:#fef2f2;border-radius:12px;color:#dc2626'><strong>Error:</strong> {str(e)}</div>"

def create_risk_by_hour_chart():
    hours = list(stats['hourly_risk'].keys())
    risks  = list(stats['hourly_risk'].values())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=risks, mode='lines+markers',
        line=dict(color='#ef4444', width=2.5),
        marker=dict(size=6, color='#ef4444'),
        fill='tozeroy', fillcolor='rgba(239,68,68,0.1)'))
    fig.update_layout(
        title=dict(text='Risk Level by Hour of Day', font=dict(color='#111827', size=16)),
        xaxis=dict(title='Hour', color='#374151', gridcolor='#f3f4f6'),
        yaxis=dict(title='Average Risk Score', color='#374151', gridcolor='#f3f4f6'),
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='white', plot_bgcolor='white', showlegend=False)
    return fig

def create_city_risk_chart():
    cities = list(stats['city_risk'].keys())[:10]
    risks  = list(stats['city_risk'].values())[:10]
    fig = go.Figure(data=[go.Bar(x=cities, y=risks,
        marker_color='#3b82f6',
        text=[f'{r:.3f}' for r in risks],
        textposition='outside',
        textfont=dict(color='#111827'))])
    fig.update_layout(
        title=dict(text='Cities by Average Risk Score', font=dict(color='#111827', size=16)),
        xaxis=dict(title='City', color='#374151'),
        yaxis=dict(title='Average Risk Score', color='#374151'),
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=50, b=80),
        paper_bgcolor='white', plot_bgcolor='white')
    return fig

def create_weather_chart():
    weathers = list(stats['weather_risk'].keys())
    risks    = list(stats['weather_risk'].values())
    colors   = {'Clear':'#10b981','Rain':'#3b82f6','Fog':'#6b7280'}
    fig = go.Figure(data=[go.Bar(x=weathers, y=risks,
        marker_color=[colors.get(w,'#ef4444') for w in weathers],
        text=[f'{r:.3f}' for r in risks],
        textposition='outside',
        textfont=dict(color='#111827'))])
    fig.update_layout(
        title=dict(text='Risk Level by Weather', font=dict(color='#111827', size=16)),
        xaxis=dict(title='Weather', color='#374151'),
        yaxis=dict(title='Average Risk Score', color='#374151'),
        template='plotly_white', height=350,
        margin=dict(l=40, r=20, t=50, b=60),
        paper_bgcolor='white', plot_bgcolor='white')
    return fig

def create_model_chart():
    models    = ['Random Forest','XGBoost','KNN','Logistic Regression']
    accuracy  = [0.692, 0.656, 0.527, 0.497]
    precision = [0.710, 0.670, 0.540, 0.510]
    recall    = [0.690, 0.660, 0.530, 0.500]
    fig = go.Figure(data=[
        go.Bar(name='Accuracy',  x=models, y=accuracy,
               text=[f'{v:.1%}' for v in accuracy],
               textposition='outside', marker_color='#3b82f6'),
        go.Bar(name='Precision', x=models, y=precision,
               text=[f'{v:.1%}' for v in precision],
               textposition='outside', marker_color='#ef4444'),
        go.Bar(name='Recall',    x=models, y=recall,
               text=[f'{v:.1%}' for v in recall],
               textposition='outside', marker_color='#10b981'),
    ])
    fig.update_layout(
        title=dict(text='Model Performance Comparison', font=dict(color='#111827', size=16)),
        barmode='group', template='plotly_white', height=420,
        yaxis=dict(range=[0,1.1], title='Score', color='#374151'),
        xaxis=dict(color='#374151'),
        paper_bgcolor='white', plot_bgcolor='white',
        legend=dict(font=dict(color='#374151')),
        margin=dict(l=40, r=20, t=60, b=60))
    return fig

def get_city_table():
    city_stats = df.groupby('city').agg(
        Avg_Risk=('risk_score','mean'),
        Total_Accidents=('risk_score','count'),
        Fatal_Accidents=('accident_severity', lambda x: (x=='fatal').sum())
    )
    city_stats['Fatal_Rate_%'] = (city_stats['Fatal_Accidents'] / city_stats['Total_Accidents'] * 100).round(1)
    city_stats['Avg_Risk'] = city_stats['Avg_Risk'].round(3)
    return city_stats.sort_values('Avg_Risk', ascending=False).reset_index()

hourly_chart  = create_risk_by_hour_chart()
city_chart    = create_city_risk_chart()
weather_chart = create_weather_chart()
model_chart   = create_model_chart()
city_table    = get_city_table()

try:
    with open('models/hotspot_map.html', 'r', encoding='utf-8') as f:
        map_html = f.read()
    map_encoded = map_html.replace('"','&quot;').replace("'","&#39;")
except:
    map_encoded = "<h3 style='padding:40px;text-align:center;color:#111827'>Map not found. Run map_generator.py first.</h3>"

# ── CSS ───────────────────────────────────────────────────────────
css = """
* { box-sizing: border-box; }

body, .gradio-container, .main, .wrap, .app {
    background: #f3f4f6 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    padding: 16px !important;
}

/* ── TABS ── */
.tab-nav {
    background: #1e293b !important;
    border-radius: 12px !important;
    padding: 8px !important;
    border: none !important;
    margin-bottom: 16px !important;
    display: flex !important;
    gap: 4px !important;
}

.tab-nav button {
    color: #cbd5e1 !important;
    background: transparent !important;
    border: none !important;
    padding: 12px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    min-width: 150px !important;
    letter-spacing: 0.01em !important;
}

.tab-nav button:hover {
    background: #334155 !important;
    color: #ffffff !important;
}

.tab-nav button.selected {
    background: #3b82f6 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.4) !important;
}

/* ── FORM ELEMENTS — force light theme ── */
.gr-form, .gr-box, .gr-panel, .gr-group,
[data-testid="block"], .block {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
}

/* Labels */
label, .label-wrap, .label-wrap span,
span.svelte-1gfkn6j, .gr-input-label {
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* All input types */
input[type="text"], input[type="number"],
input[type="email"], textarea, select,
.gr-dropdown, .gr-text-input,
.svelte-input, input.svelte-1oiin9d {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}

/* Dropdown specifically */
.wrap.svelte-w3tnzj, .secondary-wrap {
    background: #ffffff !important;
    color: #111827 !important;
}

/* Slider labels and numbers */
.gr-slider input[type="range"] + span,
.range-slider, input[type="number"] {
    color: #111827 !important;
    background: #ffffff !important;
}

/* Radio buttons */
.gr-radio, fieldset {
    background: #f9fafb !important;
    border-radius: 10px !important;
    padding: 12px !important;
    border: 1px solid #e5e7eb !important;
}

.gr-radio label span, fieldset label span,
.gr-radio span, fieldset span {
    color: #111827 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Checkboxes */
.gr-checkbox-group, .checkbox-group {
    background: #f9fafb !important;
}

.gr-checkbox label span, input[type="checkbox"] + span {
    color: #111827 !important;
    font-weight: 500 !important;
}

/* Column backgrounds */
.gr-column, .gr-row {
    background: transparent !important;
}

/* Force white background on all svelte components */
.svelte-1oiin9d, .svelte-9zdn1l,
.svelte-a1g9ya, .svelte-1ipelgc {
    background: #ffffff !important;
    color: #111827 !important;
}

/* Dataframe table */
.gr-dataframe, table {
    background: #ffffff !important;
    color: #111827 !important;
}

.gr-dataframe th, thead th {
    background: #3b82f6 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 12px !important;
}

.gr-dataframe td, tbody td {
    color: #374151 !important;
    padding: 10px !important;
    border-bottom: 1px solid #e5e7eb !important;
    background: #ffffff !important;
}

/* Primary button */
.gr-button-primary, button[variant="primary"],
button.primary {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
}

footer { display: none !important; }
"""

# ── CHATBOT (using Gradio's built-in ChatInterface in a drawer) ───
# Since floating JS chat has issues in Gradio's sandbox,
# we use a cleaner approach: Gradio's native chatbot in a collapsible row

def gradio_chat(message, history):
    response = chat_with_gemini(message)
    return response

# ── BUILD UI ──────────────────────────────────────────────────────
with gr.Blocks(title="Indian Road Accident Risk Analyzer", css=css) as demo:

    # ── Header ────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="background:white;border-radius:14px;padding:24px 28px;
      margin-bottom:20px;border:1px solid #e5e7eb;
      box-shadow:0 1px 4px rgba(0,0,0,0.07);
      display:flex;align-items:center;justify-content:space-between">
      <div>
        <h1 style="font-size:28px;font-weight:700;color:#111827;margin:0 0 6px">
          🚗 Indian Road Accident Risk Analyzer
        </h1>
        <p style="color:#6b7280;font-size:14px;margin:0">
          ML-Powered Predictive Analytics &nbsp;•&nbsp; Real-time Risk Assessment
        </p>
      </div>
      <div style="display:flex;gap:28px;text-align:right">
        <div>
          <div style="font-size:28px;font-weight:700;color:#111827">{len(df):,}</div>
          <div style="font-size:12px;color:#6b7280">Total Records</div>
        </div>
        <div>
          <div style="font-size:28px;font-weight:700;color:#ef4444">{stats['fatal_rate']:.1f}%</div>
          <div style="font-size:12px;color:#6b7280">Fatal Rate</div>
        </div>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Dashboard ──────────────────────────────────────
        with gr.Tab("📊 Dashboard"):
            gr.HTML(f"""
            <div style="display:flex;gap:16px;margin:8px 0 20px">
              <div style="flex:1;background:white;border-radius:12px;padding:20px;
                border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                <div style="font-size:13px;color:#6b7280;margin-bottom:6px;font-weight:500">
                  Average Risk Score
                </div>
                <div style="font-size:36px;font-weight:700;color:#f59e0b">
                  {stats['avg_risk']:.3f}
                </div>
              </div>
              <div style="flex:1;background:white;border-radius:12px;padding:20px;
                border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                <div style="font-size:13px;color:#6b7280;margin-bottom:6px;font-weight:500">
                  Fatal Accidents
                </div>
                <div style="font-size:36px;font-weight:700;color:#ef4444">
                  {stats['fatal_count']:,}
                </div>
              </div>
              <div style="flex:1;background:white;border-radius:12px;padding:20px;
                border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                <div style="font-size:13px;color:#6b7280;margin-bottom:6px;font-weight:500">
                  Cities Covered
                </div>
                <div style="font-size:36px;font-weight:700;color:#3b82f6">8</div>
              </div>
              <div style="flex:1;background:white;border-radius:12px;padding:20px;
                border:1px solid #e5e7eb;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                <div style="font-size:13px;color:#6b7280;margin-bottom:6px;font-weight:500">
                  Best Model Accuracy
                </div>
                <div style="font-size:36px;font-weight:700;color:#10b981">69.2%</div>
              </div>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.Plot(value=hourly_chart)
                with gr.Column():
                    gr.Plot(value=city_chart)

            with gr.Row():
                with gr.Column():
                    gr.Plot(value=weather_chart)
                with gr.Column():
                    gr.HTML("""
                    <div style="background:white;border-radius:12px;padding:0;
                      border:1px solid #e5e7eb;overflow:hidden">
                      <div style="background:#1e293b;padding:14px 18px">
                        <span style="color:#ffffff;font-size:15px;font-weight:600">
                          City Risk Ranking
                        </span>
                      </div>
                    """)
                    gr.Dataframe(
                        value=city_table,
                        label="",
                        headers=["City","Avg Risk","Total Accidents","Fatal Accidents","Fatal Rate %"],
                    )
                    gr.HTML("</div>")

        # ── Tab 2: Predict Risk ───────────────────────────────────
        with gr.Tab("🔮 Predict Risk"):
            gr.HTML("""
            <div style="background:white;border-radius:12px;padding:16px 20px;
              margin:8px 0 16px;border:1px solid #e5e7eb">
              <h3 style="margin:0 0 4px;color:#111827;font-size:17px;font-weight:600">
                Accident Risk Predictor
              </h3>
              <p style="margin:0;color:#6b7280;font-size:14px">
                Enter road and environmental conditions to assess accident risk level.
              </p>
            </div>
            """)

            with gr.Row():
                with gr.Column(elem_id="col-left"):
                    gr.HTML("<p style='font-size:13px;font-weight:700;color:#3b82f6;margin:0 0 8px;text-transform:uppercase;letter-spacing:0.05em'>📍 Location & Environment</p>")
                    city            = gr.Dropdown(list(city_map.keys()), label="City", value="Mumbai")
                    weather         = gr.Dropdown(list(weather_map.keys()), label="Weather", value="Clear")
                    visibility      = gr.Dropdown(list(vis_map.keys()), label="Visibility", value="High")
                    road_type       = gr.Dropdown(list(road_map.keys()), label="Road Type", value="Urban")
                    traffic_density = gr.Dropdown(list(traffic_map.keys()), label="Traffic Density", value="Medium")

                with gr.Column(elem_id="col-mid"):
                    gr.HTML("<p style='font-size:13px;font-weight:700;color:#3b82f6;margin:0 0 8px;text-transform:uppercase;letter-spacing:0.05em'>⏱️ Time & Incident</p>")
                    hour              = gr.Slider(0, 23, value=18, step=1, label="Hour of Day")
                    day_of_week       = gr.Dropdown(list(day_map.keys()), label="Day of Week", value="Friday")
                    temperature       = gr.Slider(5, 50, value=28, step=1, label="Temperature (°C)")
                    vehicles_involved = gr.Slider(1, 10, value=2, step=1, label="Vehicles Involved")
                    casualties        = gr.Slider(0, 10, value=1, step=1, label="Casualties")

                with gr.Column(elem_id="col-right"):
                    gr.HTML("<p style='font-size:13px;font-weight:700;color:#3b82f6;margin:0 0 8px;text-transform:uppercase;letter-spacing:0.05em'>🛣️ Road & Context</p>")
                    festival      = gr.Dropdown(list(festival_map.keys()), label="Festival", value="None")
                    cause         = gr.Dropdown(list(cause_map.keys()), label="Primary Cause", value="Overspeeding")
                    lanes         = gr.Slider(1, 6, value=2, step=1, label="Number of Lanes")
                    risk_score_in = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Initial Risk Score")
                    with gr.Row():
                        is_weekend     = gr.Checkbox(label="Weekend", value=False)
                        is_peak_hour   = gr.Checkbox(label="Peak Hour", value=True)
                        traffic_signal = gr.Checkbox(label="Traffic Signal", value=True)

            gr.HTML("<div style='background:white;border-radius:12px;padding:16px;margin:8px 0;border:1px solid #e5e7eb'>")
            model_choice = gr.Radio(
                ['Random Forest','XGBoost','KNN','Logistic Regression'],
                label="Select Model", value="Random Forest"
            )
            gr.HTML("</div>")

            predict_btn       = gr.Button("🔮 Analyze Risk", variant="primary", size="lg")
            prediction_output = gr.HTML()

            predict_btn.click(
                fn=predict_risk,
                inputs=[city, weather, road_type, visibility, traffic_density,
                        hour, is_weekend, is_peak_hour, temperature, lanes,
                        traffic_signal, vehicles_involved, casualties,
                        risk_score_in, day_of_week, festival, cause, model_choice],
                outputs=prediction_output
            )

        # ── Tab 3: Hotspot Map ────────────────────────────────────
        with gr.Tab("🗺️ Hotspot Map"):
            gr.HTML(f"""
            <div style="background:white;border-radius:14px;overflow:hidden;
              border:1px solid #e5e7eb;box-shadow:0 1px 4px rgba(0,0,0,0.07);
              margin-top:8px">
              <div style="padding:16px 20px;border-bottom:1px solid #f3f4f6;background:white">
                <h3 style="margin:0 0 4px;color:#111827;font-size:17px;font-weight:600">
                  Accident Hotspot Map
                </h3>
                <p style="margin:0;color:#6b7280;font-size:14px">
                  Interactive heatmap of accident-prone zones.
                  Click markers for details. Toggle layers top-right.
                </p>
              </div>
              <iframe srcdoc="{map_encoded}" width="100%" height="600"
                style="border:none;display:block"></iframe>
            </div>
            """)

        # ── Tab 4: Model Performance ──────────────────────────────
        with gr.Tab("📈 Model Performance"):
            gr.Plot(value=model_chart)
            gr.HTML("""
            <div style="background:white;padding:28px;border-radius:14px;
              border:1px solid #e5e7eb;margin-top:16px;
              box-shadow:0 1px 4px rgba(0,0,0,0.07)">

              <h3 style="color:#111827;font-size:18px;font-weight:700;margin:0 0 20px">
                Model Performance Summary
              </h3>

              <table style="width:100%;border-collapse:collapse;font-family:'Segoe UI',sans-serif">
                <thead>
                  <tr style="background:#3b82f6">
                    <th style="padding:13px 16px;text-align:left;color:white;font-weight:600;font-size:14px">Model</th>
                    <th style="padding:13px 16px;text-align:center;color:white;font-weight:600;font-size:14px">Accuracy</th>
                    <th style="padding:13px 16px;text-align:center;color:white;font-weight:600;font-size:14px">Precision</th>
                    <th style="padding:13px 16px;text-align:center;color:white;font-weight:600;font-size:14px">Recall</th>
                    <th style="padding:13px 16px;text-align:center;color:white;font-weight:600;font-size:14px">F1-Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style="background:white">
                    <td style="padding:13px 16px;color:#111827;font-weight:600;border-bottom:1px solid #e5e7eb">🌲 Random Forest</td>
                    <td style="padding:13px 16px;text-align:center;color:#10b981;font-weight:700;border-bottom:1px solid #e5e7eb">69.2%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">71.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">69.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">70.0%</td>
                  </tr>
                  <tr style="background:#f9fafb">
                    <td style="padding:13px 16px;color:#111827;font-weight:600;border-bottom:1px solid #e5e7eb">⚡ XGBoost</td>
                    <td style="padding:13px 16px;text-align:center;color:#3b82f6;font-weight:700;border-bottom:1px solid #e5e7eb">65.6%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">67.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">66.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">66.0%</td>
                  </tr>
                  <tr style="background:white">
                    <td style="padding:13px 16px;color:#111827;font-weight:600;border-bottom:1px solid #e5e7eb">📍 KNN</td>
                    <td style="padding:13px 16px;text-align:center;color:#f59e0b;font-weight:700;border-bottom:1px solid #e5e7eb">52.7%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">54.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">53.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151;border-bottom:1px solid #e5e7eb">53.0%</td>
                  </tr>
                  <tr style="background:#f9fafb">
                    <td style="padding:13px 16px;color:#111827;font-weight:600">📈 Logistic Regression</td>
                    <td style="padding:13px 16px;text-align:center;color:#ef4444;font-weight:700">49.7%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151">51.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151">50.0%</td>
                    <td style="padding:13px 16px;text-align:center;color:#374151">50.0%</td>
                  </tr>
                </tbody>
              </table>

              <div style="margin-top:28px">
                <h3 style="color:#111827;font-size:17px;font-weight:700;margin:0 0 16px">
                  Key Insights
                </h3>
                <div style="display:flex;flex-direction:column;gap:10px">
                  <div style="padding:13px 16px;background:#f0fdf4;border-radius:10px;
                    border-left:4px solid #10b981;color:#374151;font-size:14px">
                    🌲 <strong style="color:#111827">Random Forest</strong>
                    performs best overall with 69.2% accuracy
                  </div>
                  <div style="padding:13px 16px;background:#fef2f2;border-radius:10px;
                    border-left:4px solid #ef4444;color:#374151;font-size:14px">
                    🔴 <strong style="color:#111827">Fatal accidents</strong>
                    detected with very high recall (99%) in Random Forest
                  </div>
                  <div style="padding:13px 16px;background:#fffbeb;border-radius:10px;
                    border-left:4px solid #f59e0b;color:#374151;font-size:14px">
                    ⚠️ <strong style="color:#111827">Major severity</strong>
                    is the most challenging class due to class imbalance
                  </div>
                  <div style="padding:13px 16px;background:#eff6ff;border-radius:10px;
                    border-left:4px solid #3b82f6;color:#374151;font-size:14px">
                    📊 <strong style="color:#111827">Risk Score</strong>
                    is the single most important predictive feature across all models
                  </div>
                </div>
              </div>
            </div>
            """)

        # ── Tab 5: AI Assistant ───────────────────────────────────
        with gr.Tab("🤖 AI Assistant"):
            gr.HTML("""
            <div style="background:white;border-radius:12px;padding:16px 20px;
              margin:8px 0 16px;border:1px solid #e5e7eb">
              <h3 style="margin:0 0 4px;color:#111827;font-size:17px;font-weight:600">
                Road Safety AI Assistant
              </h3>
              <p style="margin:0;color:#6b7280;font-size:14px">
                Ask anything about accident patterns, risk factors, cities, or this project.
              </p>
            </div>
            """)
            gr.ChatInterface(
                fn=gradio_chat,
                chatbot=gr.Chatbot(height=420, label=""),
                textbox=gr.Textbox(
                    placeholder="e.g. Which city has highest accident risk? What causes fatal accidents?",
                    container=False
            ),
            examples=[
                "Which city has the highest accident risk?",
                "What weather conditions cause fatal accidents?",
                "What are the peak accident hours?",
                "How does festival season affect accident rates?",
                "What is the Random Forest model accuracy?",
            ],
        )

# ── Chat API ──────────────────────────────────────────────────────
app_api = demo.app

@app_api.post("/chat")
async def chat_endpoint(request: Request):
    data     = await request.json()
    message  = data.get("message", "")
    response = chat_with_gemini(message)
    return JSONResponse({"response": response})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("🚀 INDIAN ROAD ACCIDENT RISK ANALYZER")
    print("="*55)
    print(f"✅ Open at: http://localhost:7860")
    print("="*55)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)