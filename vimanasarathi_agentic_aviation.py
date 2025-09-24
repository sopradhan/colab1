"""
VimanaSarathi AI — Full runnable agentic skeleton (single-file)

Features:
- FastAPI-based microservice skeleton for Orchestrator and Agents (Maintenance, Vision, NLP).
- MessageBus abstraction with InMemoryBus (default) and optional KafkaBus (if kafka-python + env USE_KAFKA=1).
- Agents expose /predict endpoints; orchestrator ingests events and dynamically selects agents and models.
- LLM Reasoner integration preserves calling pattern via langchain_openai.ChatOpenAI and llm.invoke(prompt).
- CLI modes: run_orchestrator, run_agent --agent-type {maintenance,vision,nlp}, run_all_sim (starts apps in threads and simulates events).

Run examples (development):
1) Single-process simulation: python vimanasarathi_full_agentic_system.py run_all_sim

2) Run orchestrator only (on port 8000):
   python vimanasarathi_full_agentic_system.py run_orchestrator --host 0.0.0.0 --port 8000

3) Run agent (maintenance) on port 8101:
   python vimanasarathi_full_agentic_system.py run_agent --agent-type maintenance --port 8101

Notes:
- Provide .env with api_endpoint, api_key, model for LLM or the LLM calls will be skipped (graceful fallback).
- This is a developer skeleton to demonstrate agentic flows; replace stubs (Detectron2 service, real models, Kafka) for production.

"""

import argparse
import asyncio
import os
import threading
import time
import json
import random
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

# ML imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Optional packages
try:
    from kafka import KafkaProducer, KafkaConsumer
    HAS_KAFKA = True
except Exception:
    HAS_KAFKA = False

# LLM (preserve calling pattern)
try:
    from dotenv import load_dotenv
    import httpx
    from langchain_openai import ChatOpenAI
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# -------------------------
# Utility: LLM setup
# -------------------------

def setup_llm():
    if not LLM_AVAILABLE:
        return None
    load_dotenv()
    http_client = httpx.Client(verify=False)
    base_url = os.getenv("api_endpoint")
    api_key = os.getenv("api_key")
    model_name = os.getenv("model")
    try:
        llm = ChatOpenAI(base_url=base_url, model=model_name, api_key=api_key, http_client=http_client, temperature=0.2)
        return llm
    except Exception as e:
        print("LLM setup failed:", e)
        return None

# -------------------------
# Message Bus abstraction
# -------------------------
class InMemoryBus:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def publish(self, topic: str, message: Dict[str, Any]):
        await self.queue.put((topic, message))

    async def subscribe(self):
        while True:
            item = await self.queue.get()
            yield item

class KafkaBus:
    def __init__(self, bootstrap_servers='localhost:9092'):
        if not HAS_KAFKA:
            raise RuntimeError('kafka-python not installed')
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    async def publish(self, topic: str, message: Dict[str, Any]):
        # kafka-python is blocking; wrap in thread
        def send():
            self.producer.send(topic, message)
            self.producer.flush()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send)

    async def subscribe(self, topic='events'):
        if not HAS_KAFKA:
            raise RuntimeError('kafka-python not installed')
        consumer = KafkaConsumer(topic, bootstrap_servers=self.bootstrap_servers, value_deserializer=lambda v: json.loads(v.decode('utf-8')))
        for msg in consumer:
            yield (topic, msg.value)

# -------------------------
# Simple model helpers (tabular)
# -------------------------

def train_small_tabular_model(seed=42, n=300, model_type='RandomForest'):
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        engine_temp = np.random.normal(700, 80)
        vibration = np.abs(np.random.normal(3.0, 1.5))
        oil_pressure = np.random.normal(45, 10)
        hours = np.abs(np.random.normal(200, 150))
        risk = 1 if (vibration>6.0 or engine_temp>900 or hours>500) else 0
        rows.append([engine_temp, vibration, oil_pressure, hours, risk])
    df = pd.DataFrame(rows, columns=['engine_temp','vibration','oil_pressure','hours_since_maint','risk'])
    X = df[['engine_temp','vibration','oil_pressure','hours_since_maint']].values
    y = df['risk'].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    if model_type=='RandomForest':
        clf = RandomForestClassifier(n_estimators=150, random_state=seed)
    elif model_type=='SVM':
        clf = SVC(probability=True, random_state=seed)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400, random_state=seed)
    clf.fit(Xs, y)
    return {'clf': clf, 'scaler': scaler}

# -------------------------
# Agent microservices (FastAPI)
# Each agent exposes /predict and /health
# -------------------------

class PredictRequest(BaseModel):
    event_id: str
    event_type: str
    payload: Dict[str, Any]

class PredictResponse(BaseModel):
    event_id: str
    agent_id: str
    prediction: Dict[str, Any]

# Maintenance Agent
def create_maintenance_agent(app_name='maintenance_agent', host='0.0.0.0', port=8101, model_type='RandomForest'):
    app = FastAPI(title=app_name)
    model = train_small_tabular_model(model_type=model_type)

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    @app.post('/predict', response_model=PredictResponse)
    async def predict(req: PredictRequest):
        payload = req.payload
        # expect sensors: engine_temp, vibration, oil_pressure, hours_since_maint
        features = [payload.get('engine_temp'), payload.get('vibration'), payload.get('oil_pressure'), payload.get('hours_since_maint')]
        import numpy as _np
        X = _np.array([features], dtype=float)
        Xs = model['scaler'].transform(X)
        clf = model['clf']
        try:
            prob = float(clf.predict_proba(Xs)[0][1])
        except Exception:
            prob = None
        pred = int(clf.predict(Xs)[0])
        return PredictResponse(event_id=req.event_id, agent_id=app_name, prediction={'risk':pred, 'prob':prob})

    return app

# Vision Agent (image analysis stub)
from fastapi import File, UploadFile

def create_vision_agent(app_name='vision_agent', host='0.0.0.0', port=8102):
    app = FastAPI(title=app_name)

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    @app.post('/predict', response_model=PredictResponse)
    async def predict(req: PredictRequest):
        payload = req.payload
        # in this prototype we accept optional image_meta or base64 image key
        # simulate detection
        defects = []
        if random.random() < 0.25:
            defects = random.choices(['crack','corrosion','dent','leak'], k=random.randint(1,2))
        score = min(1.0, 0.1 + 0.4*len(defects) + random.random()*0.4)
        return PredictResponse(event_id=req.event_id, agent_id=app_name, prediction={'defects':defects, 'score':round(score,3)})

    return app

# NLP Agent (crew reports)

def create_nlp_agent(app_name='nlp_agent', host='0.0.0.0', port=8103):
    app = FastAPI(title=app_name)

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    @app.post('/predict', response_model=PredictResponse)
    async def predict(req: PredictRequest):
        payload = req.payload
        text = payload.get('text','')
        findings = []
        text_lower = text.lower() if text else ''
        if 'engine' in text_lower:
            findings.append('engine_anomaly')
        if 'vibration' in text_lower or 'shake' in text_lower:
            findings.append('vibration')
        if 'oil' in text_lower or 'leak' in text_lower:
            findings.append('oil_leak')
        return PredictResponse(event_id=req.event_id, agent_id=app_name, prediction={'findings':findings})

    return app

# -------------------------
# Orchestrator app
# -------------------------

def create_orchestrator(app_name='orchestrator', host='0.0.0.0', port=8000, agent_endpoints=None, use_inproc=False):
    app = FastAPI(title=app_name)
    bus = InMemoryBus()
    llm = setup_llm()

    # agent_endpoints: dict mapping domain->http_url or 'inproc' mapping to callables
    agent_endpoints = agent_endpoints or {
        'maintenance': 'http://localhost:8101/predict',
        'vision': 'http://localhost:8102/predict',
        'nlp': 'http://localhost:8103/predict'
    }

    logs = []

    @app.get('/health')
    def health():
        return {'status':'ok','role':'orchestrator'}

    @app.post('/ingest')
    async def ingest(event: Dict[str,Any]):
        # event should contain {event_id, type, payload}
        topic = 'events'
        await bus.publish(topic, event)
        return {'status':'accepted', 'event_id': event.get('event_id')}

    async def dispatcher_loop():
        print('Orchestrator dispatcher started')
        sub = bus.subscribe()
        async for topic, event in sub:
            try:
                await handle_event(event)
            except Exception as e:
                print('Error handling event', e)

    async def call_agent_http(url, event):
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {'event_id': event.get('event_id'), 'event_type': event.get('type'), 'payload': event.get('payload')}
            r = await client.post(url, json=payload)
            return r.json()

    async def handle_event(event):
        etype = event.get('type')
        # Simple routing rules
        if etype in ['sensor','maintenance_check']:
            agent_url = agent_endpoints.get('maintenance')
            agent_key = 'maintenance'
        elif etype in ['image','drone_image']:
            agent_url = agent_endpoints.get('vision')
            agent_key = 'vision'
        elif etype in ['crew_report','log_entry']:
            agent_url = agent_endpoints.get('nlp')
            agent_key = 'nlp'
        else:
            # fallback to maintenance agent
            agent_url = agent_endpoints.get('maintenance')
            agent_key = 'maintenance'

        # call agent (http)
        try:
            if use_inproc and callable(agent_url):
                # inproc call: agent_url should be a callable that accepts event
                resp = agent_url(event)
            else:
                resp = await call_agent_http(agent_url, event)
        except Exception as e:
            print('Agent call failed', e)
            resp = {'error': str(e)}

        # Collect agent outputs and call LLM reasoner
        reasoning_input = {
            'event': event,
            'agent': resp
        }
        advisory = await call_reasoner(reasoning_input)

        log_entry = {'time': time.time(), 'event_id': event.get('event_id'), 'agent': agent_key, 'agent_resp': resp, 'advisory': advisory}
        logs.append(log_entry)
        print('Logged:', log_entry)

    async def call_reasoner(reasoning_input):
        # Build structured prompt and call LLM if available
        if not llm:
            return {'text': 'LLM not available', 'structured': reasoning_input}
        prompt = f"""
You are VimanaSarathi LLM Reasoner. Given an incoming aviation event and an agent prediction, propose an action plan.
Event: {json.dumps(reasoning_input['event'])}
Agent output: {json.dumps(reasoning_input['agent'])}
Provide:
- Maintenance checklist (3 bullets)
- Flight Ops recommendation (1-2 bullets)
- Regulatory note (1 bullet referencing possible advisories)
Respond in JSON with keys: maintenance, flight_ops, regulatory, summary.
"""
        try:
            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, 'content') else str(resp)
            return {'text': text}
        except Exception as e:
            print('LLM call failed', e)
            return {'text': f'LLM call failed: {e}'}

    # background task
    @app.on_event('startup')
    async def startup_event():
        loop = asyncio.get_event_loop()
        loop.create_task(dispatcher_loop())

    @app.get('/logs')
    def get_logs():
        return {'logs': logs[-50:]}

    return app

# -------------------------
# Simple simulation client to produce events to orchestrator
# -------------------------
async def simulate_events(orchestrator_url='http://localhost:8000/ingest', n=50, interval=0.5):
    import httpx
    async with httpx.AsyncClient() as client:
        for i in range(n):
            ev_type = random.choices(['sensor','image','crew_report'], weights=[0.6,0.2,0.2])[0]
            event = {'event_id': f'EV{i}_{int(time.time())}', 'type': ev_type}
            if ev_type=='sensor':
                event['payload'] = {'engine_temp': round(np.random.normal(700,100),2),'vibration': round(abs(np.random.normal(3.0,2.0)),3),'oil_pressure': round(np.random.normal(45,10),2),'hours_since_maint': int(abs(np.random.normal(200,150)))}
            elif ev_type=='image':
                event['payload'] = {'image_id': f'IMG{i}', 'meta': {'resolution': 'high'}}
            else:
                event['payload'] = {'text': random.choice(['engine anomaly observed','vibration increasing','no issue','oil leak smell'])}
            try:
                r = await client.post(orchestrator_url, json=event)
                print('Produced', event['event_id'], '->', r.status_code)
            except Exception as e:
                print('Produce failed', e)
            await asyncio.sleep(interval)

# -------------------------
# CLI Entrypoint
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('run_all_sim', help='Run orchestrator + agents locally and simulate events')

    run_orch = sub.add_parser('run_orchestrator', help='Run orchestrator only')
    run_orch.add_argument('--host', default='0.0.0.0')
    run_orch.add_argument('--port', default=8000, type=int)

    run_agent = sub.add_parser('run_agent', help='Run an agent service')
    run_agent.add_argument('--agent-type', choices=['maintenance','vision','nlp'], required=True)
    run_agent.add_argument('--port', type=int, required=True)

    args = parser.parse_args()

    if args.cmd=='run_all_sim':
        # Start agents in threads
        def start_app(app, port):
            uvicorn.run(app, host='0.0.0.0', port=port)

        maint_app = create_maintenance_agent(port=8101)
        vis_app = create_vision_agent(port=8102)
        nlp_app = create_nlp_agent(port=8103)
        orch_app = create_orchestrator()

        t1 = threading.Thread(target=start_app, args=(maint_app,8101), daemon=True)
        t2 = threading.Thread(target=start_app, args=(vis_app,8102), daemon=True)
        t3 = threading.Thread(target=start_app, args=(nlp_app,8103), daemon=True)
        t4 = threading.Thread(target=start_app, args=(orch_app,8000), daemon=True)

        t1.start(); t2.start(); t3.start(); t4.start()
        print('All services started (orch@8000, maint@8101, vis@8102, nlp@8103)')

        # kick off async simulator
        asyncio.run(simulate_events(orchestrator_url='http://localhost:8000/ingest', n=200, interval=0.2))

        print('Simulation finished — services still running. Ctrl+C to exit.')
        while True:
            time.sleep(1)

    elif args.cmd=='run_orchestrator':
        app = create_orchestrator()
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.cmd=='run_agent':
        if args.agent_type=='maintenance':
            app = create_maintenance_agent(port=args.port)
        elif args.agent_type=='vision':
            app = create_vision_agent(port=args.port)
        else:
            app = create_nlp_agent(port=args.port)
        uvicorn.run(app, host='0.0.0.0', port=args.port)

    else:
        parser.print_help()
