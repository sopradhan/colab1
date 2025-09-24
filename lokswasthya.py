"""
LokSwasthya Agentic Food Safety Prototype
- Streamlit app that simulates an agentic microservice architecture for food safety monitoring
- Preserves LLM calling pattern via langchain_openai.ChatOpenAI and llm.invoke(prompt)
- Uses supervised models (RandomForest, SVM, MLP, optional XGBoost) for tabular risk prediction
- Simulates ingestion from: Government (FSSAI/ICAR), Market/Vendor, Lab/Sensor, Consumer surveys, Environmental feeds, and Image uploads (image processing stub)
- Provides Agent and Orchestrator classes to route readings, run harmonization, call models, and trigger LLM explanations and stakeholder notifications

Run: streamlit run lokswasthya_agentic_foodsafety.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import base64
import io
import json
import altair as alt

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# =============================
# LLM setup (preserve calling pattern)
# =============================
@st.cache_resource
def setup_llm():
    load_dotenv()
    http_client = httpx.Client(verify=False)
    base_url = os.getenv("api_endpoint")
    api_key = os.getenv("api_key")
    model_name = os.getenv("model")

    llm = ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        http_client=http_client,
        temperature=0.2
    )
    return llm

# =============================
# Domain definitions
# =============================
FOOD_GROUPS = ["Dairy", "Grains", "Vegetables", "Fruits", "Livestock"]
MINERALS = ['Ca','Fe','Mg','Zn','K','Na','Cu','Se','P','I','Mn']
TOXINS = ['Aflatoxin', 'PesticideResidue', 'Lead', 'Mercury']
PATHOGENS = ['Brucella', 'FMDV', 'BlueTongue', 'AvianInfluenza']
SENSOR_FEATURES = ['Moisture%', 'pH', 'Temp_C', 'CO2_ppm']
TABULAR_FEATURES = ['Moisture%', 'pH', 'Temp_C', 'Aflatoxin_ppb', 'Pesticide_ppm', 'Lead_ppm', 'Iron_mg']

np.random.seed(42)

# =============================
# Synthetic/harmonization helpers
# =============================

def synth_gov_database():
    # Mock FSSAI/ICAR thresholds and recall lists
    thresholds = {
        'Aflatoxin_ppb': 20,
        'Pesticide_ppm': 0.1,
        'Lead_ppm': 0.1
    }
    recalls = [
        {'product_code': 'WHT-2025-001', 'reason': 'Aflatoxin exceedance', 'regions': ['State1','State3']}
    ]
    return thresholds, recalls


def generate_market_record():
    return {
        'vendor_id': f"V{random.randint(100,999)}",
        'product_code': random.choice(['WHT-2025-001','MILK-2025-010','VEG-550']),
        'batch_id': f"B{random.randint(10000,99999)}",
        'quantity_kg': round(random.uniform(10,500),1),
        'price_per_kg': round(random.uniform(20,200),2),
        'region': random.choice(['State1','State2','State3'])
    }


def generate_lab_sensor_reading(food_group):
    # Simulate lab readings; ranges chosen plausibly but synthetic
    moisture = round(random.uniform(5,25),2)
    pH = round(random.uniform(4.0,8.5),2)
    temp = round(random.uniform(2,40),2)
    aflatoxin = round(max(0, np.random.normal(5,8)),2)
    pesticide = round(max(0, np.random.normal(0.05,0.05)),3)
    lead = round(max(0, np.random.normal(0.02,0.03)),4)
    iron = round(max(0, np.random.normal(2.5,1.0)),2)

    return {
        'Moisture%': moisture,
        'pH': pH,
        'Temp_C': temp,
        'Aflatoxin_ppb': aflatoxin,
        'Pesticide_ppm': pesticide,
        'Lead_ppm': lead,
        'Iron_mg': iron
    }


def generate_consumer_feedback():
    # Simplified survey -> symptom linking
    complaints = ['stomach pain', 'nausea', 'no_issue', 'fever', 'diarrhea']
    return {'complaint': random.choice(complaints), 'severity': random.choice([1,2,3])}


def generate_environmental_feed(region):
    return {'region': region, 'rain_mm': round(random.uniform(0,300),1), 'flood_risk': random.choice([0,1]), 'temp_C': round(random.uniform(15,40),1)}

# =============================
# Model training for tabular detection
# =============================

def train_tabular_model(n=200, model_type='RandomForest'):
    rows = []
    for _ in range(n):
        row = generate_lab_sensor_reading(None)
        # Label logic: unsafe if aflatoxin>20 or pesticide>0.15 or lead>0.1 or temp>35
        unsafe = 1 if (row['Aflatoxin_ppb']>20 or row['Pesticide_ppm']>0.15 or row['Lead_ppm']>0.1 or row['Temp_C']>35) else 0
        row['unsafe'] = unsafe
        rows.append(row)
    df = pd.DataFrame(rows)

    le = LabelEncoder()
    # 'unsafe' is already binary
    X = df[TABULAR_FEATURES].fillna(0).values
    y = df['unsafe'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if model_type == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
    elif model_type == 'SVM':
        clf = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_type == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400, random_state=42)
    elif model_type == 'XGBoost' and HAS_XGBOOST:
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(Xs, y)
    return {'model': clf, 'scaler': scaler}


def predict_tabular_risk(reading, model_dict):
    df = pd.DataFrame([reading])
    X = df[TABULAR_FEATURES].fillna(0).values
    Xs = model_dict['scaler'].transform(X)
    clf = model_dict['model']
    try:
        prob = clf.predict_proba(Xs)[0][1]
    except Exception:
        prob = None
    pred = clf.predict(Xs)[0]
    return int(pred), float(prob) if prob is not None else None

# =============================
# Agent & Orchestrator
# =============================
class Agent:
    def __init__(self, agent_id, food_group, region, model_dict, llm=None):
        self.agent_id = agent_id
        self.food_group = food_group
        self.region = region
        self.model = model_dict
        self.llm = llm
        self.log = []

    def harmonize(self, market, lab, env, survey, image_meta=None, gov_thresholds=None):
        # Normalize and merge readings into a single record
        record = {}
        record.update(market)
        record.update(lab)
        record.update({'region_env_temp': env.get('temp_C') if env else None})
        record.update({'complaint': survey.get('complaint') if survey else None, 'complaint_severity': survey.get('severity') if survey else None})
        if image_meta:
            record['image_flag'] = image_meta.get('flag', False)
        else:
            record['image_flag'] = False
        record['food_group'] = self.food_group
        return record

    def analyze(self, record, gov_thresholds):
        # Run tabular risk model
        risk_label, risk_prob = predict_tabular_risk(record, self.model)

        # Rule-based checks against govt thresholds
        issues = []
        if gov_thresholds:
            for k,v in gov_thresholds.items():
                if record.get(k) is not None and record.get(k) > v:
                    issues.append({'parameter': k, 'value': record.get(k), 'threshold': v})

        # Image flag or survey complaint raise priority
        if record.get('image_flag'):
            issues.append({'parameter': 'image', 'value': True})
        if record.get('complaint') and record.get('complaint') != 'no_issue':
            issues.append({'parameter': 'consumer_complaint', 'value': record.get('complaint')})

        # Compose summary
        severity = 'LOW'
        if risk_label == 1 or len(issues) > 0:
            severity = 'HIGH' if risk_prob and risk_prob>0.6 else 'MEDIUM'

        return {'risk_label': risk_label, 'risk_prob': risk_prob, 'issues': issues, 'severity': severity}

    def notify(self, analysis, record):
        # Build prompt for LLM; preserve llm.invoke interface
        if self.llm:
            prompt = f"""You are a public health food-safety analyst.
Agent: {self.agent_id} ({self.food_group} in {self.region})
Summary: severity={analysis['severity']}, risk_prob={analysis['risk_prob']}
Detected issues: {json.dumps(analysis['issues'])}
Market record: {json.dumps({'vendor_id':record.get('vendor_id'),'product_code':record.get('product_code'),'batch_id':record.get('batch_id')})}
Provide: concise advisory for vendor, action for state health authority, and consumer advisory in 3 short bullet points each."""
            try:
                llm_resp = self.llm.invoke(prompt)
                text = llm_resp.content if hasattr(llm_resp, 'content') else str(llm_resp)
            except Exception as e:
                text = f"LLM invocation failed: {str(e)}"
        else:
            text = "LLM not available"

        event = {
            'time': pd.Timestamp.now(),
            'agent_id': self.agent_id,
            'food_group': self.food_group,
            'region': self.region,
            'severity': analysis['severity'],
            'risk_prob': analysis['risk_prob'],
            'issues': analysis['issues'],
            'advisory': text
        }
        self.log.append(event)
        return event

class Orchestrator:
    def __init__(self):
        self.agents = {}

    def register(self, agent: Agent):
        self.agents[agent.agent_id] = agent

    def route(self, reading, target_agent_id=None):
        if target_agent_id and target_agent_id in self.agents:
            return self.agents[target_agent_id], self.agents[target_agent_id]
        # pick agent by food_group/region matching
        candidates = [a for a in self.agents.values() if a.food_group==reading.get('food_group')]
        if not candidates:
            candidates = list(self.agents.values())
        chosen = random.choice(candidates)
        return chosen

# =============================
# Image stub (no Detectron2) — marks images with simple heuristics
# =============================

def analyze_image_stub(uploaded_file):
    # Very small heuristic: if filesize > X, mark as 'high_detail'
    try:
        b = uploaded_file.read()
        size_kb = len(b)/1024
        uploaded_file.seek(0)
        flag = True if size_kb>50 else False
        # return metadata
        return {'size_kb': size_kb, 'flag': flag, 'notes': 'image_stub_processed'}
    except Exception:
        return {'size_kb':0, 'flag': False, 'notes':'failed'}

# =============================
# UI and Simulation
# =============================

def run_simulation(model_choice='RandomForest'):
    st.header('LokSwasthya — Agentic Food Safety Simulator')

    # Train tabular model
    with st.spinner('Training tabular risk model...'):
        model_dict = train_tabular_model(model_type=model_choice)
    st.success('Tabular model ready')

    # LLM
    try:
        llm = setup_llm()
        st.success('LLM connected')
    except Exception as e:
        st.warning(f'LLM not available: {str(e)}')
        llm = None

    # Setup Orchestrator & Agents (one per food group/region combo)
    orch = Orchestrator()
    regions = ['State1','State2','State3']
    for fg in FOOD_GROUPS:
        for r in regions:
            aid = f"agent_{fg}_{r}"
            ag = Agent(aid, fg, r, model_dict, llm=llm)
            orch.register(ag)

    thresholds, recalls = synth_gov_database()

    # Controls
    iterations = st.slider('Simulated events', 10, 500, 100)
    interval = st.slider('Interval (s)', 0.1, 3.0, 0.5)

    # Placeholders
    events_df = pd.DataFrame(columns=['time','agent_id','food_group','region','severity','risk_prob'])
    table_ph = st.empty()
    raw_ph = st.empty()

    for i in range(iterations):
        market = generate_market_record()
        lab = generate_lab_sensor_reading(market.get('product_code'))
        survey = generate_consumer_feedback() if random.random()<0.2 else None
        env = generate_environmental_feed(market.get('region'))
        # small chance of image upload
        image_meta = None
        if random.random()<0.15:
            # fake an uploaded file via bytesIO
            fake_image = io.BytesIO(b"PNG

" + os.urandom(5000))
            image_meta = analyze_image_stub(fake_image)

        record = {
            'food_group': random.choice(FOOD_GROUPS),
            **market,
            **lab
        }

        # route
        agent = orch.route(record)
        harmonized = agent.harmonize(market, lab, env, survey, image_meta=image_meta, gov_thresholds=thresholds)
        analysis = agent.analyze(harmonized, thresholds)
        event = agent.notify(analysis, harmonized)

        # append event
        events_df = pd.concat([events_df, pd.DataFrame([{'time':event['time'],'agent_id':event['agent_id'],'food_group':event['food_group'],'region':event['region'],'severity':event['severity'],'risk_prob':event['risk_prob']}])], ignore_index=True)

        # UI updates
        table_ph.dataframe(events_df.tail(10))
        raw_ph.json({'last_event': event}, expanded=False)

        time.sleep(interval)

    st.success('Simulation finished')
    st.markdown('### Example Agent Logs')
    sample_agent = random.choice(list(orch.agents.values()))
    st.write(pd.DataFrame(sample_agent.log).tail(5))

# =============================
# CSV + Image Upload Mode
# =============================

def run_upload_mode(model_choice='RandomForest'):
    st.header('Upload real data (CSV + images)')
    uploaded = st.file_uploader('Upload CSV with lab readings (cols: ' + ','.join(TABULAR_FEATURES) + ' )', type=['csv'])
    uploaded_img = st.file_uploader('Optional: upload image(s)', type=['png','jpg','jpeg'], accept_multiple_files=True)

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        model = train_tabular_model(model_type=model_choice)
        try:
            llm = setup_llm()
        except Exception:
            llm = None

        orch = Orchestrator()
        # register a single agent for uploaded processing
        agent = Agent('uploaded_agent_1','Dairy','State1',model,llm=llm)
        orch.register(agent)

        results = []
        image_meta = None
        if uploaded_img:
            # analyze first image just as a stub
            image_meta = analyze_image_stub(uploaded_img[0])

        thresholds, recalls = synth_gov_database()

        for _, row in df.iterrows():
            market = generate_market_record()
            lab = {k: row[k] if k in row else None for k in TABULAR_FEATURES}
            survey = None
            env = generate_environmental_feed(market.get('region'))
            harmonized = agent.harmonize(market, lab, env, survey, image_meta=image_meta, gov_thresholds=thresholds)
            analysis = agent.analyze(harmonized, thresholds)
            ev = agent.notify(analysis, harmonized)
            results.append(ev)

        st.write(pd.DataFrame(results))

# =============================
# Streamlit app
# =============================

def main():
    st.set_page_config(page_title='LokSwasthya Agentic Food Safety', layout='wide')
    st.title('LokSwasthya Agentic Food Safety — Prototype')
    st.markdown('This prototype simulates an agentic microservice architecture for food safety monitoring in India. It preserves the LLM calling pattern and environment config you requested.')

    mode = st.radio('Mode:', ['Simulate', 'Upload CSV'])
    model_choice = st.selectbox('Choose tabular model:', ['RandomForest','SVM','MLP'] + (['XGBoost'] if HAS_XGBOOST else []))

    if mode == 'Simulate':
        run_simulation(model_choice=model_choice)
    else:
        run_upload_mode(model_choice=model_choice)

    st.markdown('---')
    st.markdown('**Next steps (suggested):**')
    st.markdown('- Replace synthetic connectors with actual FSSAI/ICAR API ingestors and secure webhook endpoints.
- Add Kafka/RabbitMQ for agent messaging and scale agents as containerized microservices.
- Implement Detectron2-based image microservice and a fusion service to combine image + tabular + sensor features.
- Add audit logs, RBAC, encryption, and data retention policies to meet compliance.
- Build dashboards & APIs for State Health Departments, vendors and consumer apps.')

if __name__ == '__main__':
    main()
