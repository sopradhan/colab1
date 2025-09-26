"""
SmartDairy — Enhanced AI-Driven Dairy Growth Engine
Single-file developer skeleton (FastAPI) showing:
- IoT ingestion (edge simulation)
- Edge anomaly detection (RandomForest) + neural network upgrades
- Vision agent with simple CNN stub and OpenCV-compatible interfaces
- Predictive forecaster using Keras LSTM (train-on-synthetic stub)
- Agent AI for inventory recommendations
- RAG-style retriever (VectorStoreStub) + LLM Bot with dynamic Chain-of-Thought prompting
- Orchestrator accepts bot-issued IoT action signals and routes them back to farm agents
- Message bus (InMemory) and orchestrator

Run examples:
  python smartdairy_enhanced.py run_all_sim
  python smartdairy_enhanced.py run_orchestrator --port 8000
  python smartdairy_enhanced.py run_agent --agent-type iot --port 8201

Notes:
- This skeleton is for development and demo. Replace stubs (vector DB, production LLM, persistent storage, actual device auth) for production use.
- If tensorflow/keras or opencv are unavailable the code falls back to lightweight alternatives and graceful messages.
"""

import argparse
import asyncio
import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Optional deep learning (Keras)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional CV (OpenCV)
try:
    import cv2
    CV_AVAILABLE = True
except Exception:
    CV_AVAILABLE = False

# Optional LLM client
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
    try:
        load_dotenv()
    except Exception:
        pass
    try:
        http_client = httpx.Client(verify=False)
        base_url = os.getenv("API_ENDPOINT") or os.getenv("api_endpoint")
        api_key = os.getenv("API_KEY") or os.getenv("api_key")
        model_name = os.getenv("MODEL") or os.getenv("model")
        llm = ChatOpenAI(base_url=base_url, model=model_name, api_key=api_key, http_client=http_client, temperature=0.0)
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

# -------------------------
# Data Model helpers
# -------------------------

def train_dairy_anomaly_detector(seed: int = 42, n: int = 800):
    """Train a RandomForest on synthetic dairy sensor data."""
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        milk = np.random.normal(18, 6)
        cow_temp = np.random.normal(38.6, 0.6)
        feed = np.random.normal(24, 7)
        amb_temp = np.random.normal(22, 8)
        anomaly = 1 if (milk < 10 or milk > 32 or cow_temp > 39.5 or feed < 8) else 0
        rows.append([milk, cow_temp, feed, amb_temp, anomaly])
    df = pd.DataFrame(rows, columns=["milk","cow_temp","feed","amb_temp","anomaly"])
    X = df[["milk","cow_temp","feed","amb_temp"]].values
    y = df["anomaly"].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(Xs, y)
    return {"clf": clf, "scaler": scaler}

# -------------------------
# Neural forecasting (LSTM) - training stub
# -------------------------

def build_lstm_forecaster(seq_len: int = 30):
    class LSTMForecaster:
        def __init__(self):
            self.seq_len = seq_len
            self.model = None
            self.trained = False

        def _make_dataset(self, series: List[float]):
            X, y = [], []
            for i in range(len(series) - self.seq_len):
                X.append(series[i:i + self.seq_len])
                y.append(series[i + self.seq_len])
            if not X:
                return np.zeros((0, self.seq_len, 1)), np.zeros((0,))
            X = np.array(X).reshape(-1, self.seq_len, 1)
            y = np.array(y)
            return X, y

        def train(self, series: List[float], epochs: int = 5):
            if not TF_AVAILABLE:
                print("TF not available, skipping LSTM training")
                return
            X, y = self._make_dataset(series)
            if X.shape[0] < 10:
                print("Not enough data to train LSTM")
                return
            model = keras.Sequential([
                keras.layers.Input(shape=(self.seq_len, 1)),
                keras.layers.LSTM(64, return_sequences=False),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
            self.model = model
            self.trained = True

        def forecast(self, history: List[float], horizon: int = 7):
            if TF_AVAILABLE and self.trained and self.model is not None and len(history) >= self.seq_len:
                seq = np.array(history[-self.seq_len:]).reshape(1, self.seq_len, 1)
                preds = []
                cur = seq
                for _ in range(horizon):
                    p = float(self.model.predict(cur, verbose=0)[0][0])
                    preds.append(p)
                    cur = np.roll(cur, -1)
                    cur[0, -1, 0] = p
                return [float(p) for p in preds]
            # fallback: seasonal naive or mean
            if len(history) >= 7:
                return [float(history[-7 + (i % 7)]) for i in range(horizon)]
            return [float(np.mean(history)) if history else 0.0 for _ in range(horizon)]

    return LSTMForecaster()

# -------------------------
# Vision model stub (CNN) + simple image analysis
# -------------------------

def build_vision_model():
    class VisionModel:
        def __init__(self):
            self.trained = False
            self.model = None
            if TF_AVAILABLE:
                self.model = keras.Sequential([
                    keras.layers.Input(shape=(64,64,3)),
                    keras.layers.Conv2D(16, 3, activation='relu'),
                    keras.layers.MaxPool2D(),
                    keras.layers.Conv2D(32,3,activation='relu'),
                    keras.layers.Flatten(),
                    keras.layers.Dense(32, activation='relu'),
                    keras.layers.Dense(2, activation='softmax')
                ])
                self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        def analyze_image_bytes(self, image_bytes: bytes):
            # convert bytes to numpy image if cv2 available
            if CV_AVAILABLE:
                arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return {"error": "invalid_image"}
                small = cv2.resize(img, (64,64))
                # naive heuristic: bright spots or stains detection via mean intensity
                mean_brightness = float(np.mean(small))
                abnormal = mean_brightness < 40 or mean_brightness > 210
                # simulate defect detection
                defects = []
                if abnormal:
                    defects.append('suspect_stain_or_overexposure')
                # if TF model exists, run predict (stub)
                score = random.random()
                return {"defects": defects, "score": round(score,3), "brightness": round(mean_brightness,2)}
            # fallback: random simulation
            return {"defects": random.choices(['none','crack','stain'], weights=[0.8,0.1,0.1], k=1), "score": round(random.random(),3)}

    return VisionModel()

# -------------------------
# Vector store (RAG) stub
# -------------------------
class VectorStoreStub:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    def add(self, doc_id: str, text: str):
        emb = None
        self.docs.append({"id": doc_id, "text": text, "embedding": emb})

    def search(self, query: str, k: int = 3):
        scores = []
        q = query.lower()
        for d in self.docs:
            t = d['text'].lower()
            score = t.count(q) + (1 if q in t else 0)
            scores.append((score, d))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [d for s,d in scores[:k] if s>0]

# -------------------------
# FastAPI Schemas
# -------------------------
class IngestEvent(BaseModel):
    event_id: str
    type: str
    timestamp: Optional[float] = None
    payload: Dict[str, Any]

class PredictResponse(BaseModel):
    event_id: str
    agent_id: str
    prediction: Dict[str, Any]

# -------------------------
# Agent: IoT (Edge) - anomaly detection + actuator receiver
# -------------------------

def create_iot_agent(app_name: str = "iot_agent", port: int = 8201):
    app = FastAPI(title=app_name)
    model = train_dairy_anomaly_detector()

    # simple in-memory device state
    devices: Dict[str, Dict[str, Any]] = {}

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    @app.post('/predict', response_model=PredictResponse)
    async def predict(ev: IngestEvent):
        p = ev.payload
        try:
            feats = [p.get('milk_yield'), p.get('cow_temp'), p.get('feed_intake'), p.get('ambient_temp')]
            X = np.array([feats], dtype=float)
            Xs = model['scaler'].transform(X)
            prob = float(model['clf'].predict_proba(Xs)[0][1])
            pred = int(model['clf'].predict(Xs)[0])
            fid = p.get('farm_id', 'unknown')
            devices.setdefault(fid, {})
            devices[fid]['last_seen'] = time.time()
            devices[fid]['last_payload'] = p
            return PredictResponse(event_id=ev.event_id, agent_id=app_name, prediction={'anomaly': pred, 'prob': round(prob,3)})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Bad payload or model error: {e}')

    @app.post('/actuate')
    async def actuate(cmd: Dict[str, Any]):
        # cmd: {farm_id, action, params}
        farm = cmd.get('farm_id')
        action = cmd.get('action')
        devices.setdefault(farm, {})
        devices[farm]['last_action'] = {'action': action, 'params': cmd.get('params', {}), 'time': time.time()}
        print(f'[IOT_ACTUATE] farm={farm} action={action}')
        return {'status': 'ok', 'farm': farm, 'action': action}

    @app.get('/devices')
    async def get_devices():
        return {'devices': devices}

    return app

# -------------------------
# Predictive Agent (Forecasting) using LSTM forecaster
# -------------------------

def create_predictive_agent(app_name: str = 'predictive_agent', port: int = 8301):
    app = FastAPI(title=app_name)
    forecaster = build_lstm_forecaster()
    history_store: Dict[str, List[float]] = {}

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    @app.post('/ingest')
    async def ingest(ts: IngestEvent):
        fid = ts.payload.get('farm_id', 'default')
        milk = ts.payload.get('milk_yield')
        if milk is None:
            raise HTTPException(status_code=400, detail='missing milk_yield')
        history_store.setdefault(fid, []).append(float(milk))
        # optional: train LSTM incrementally when enough data
        if TF_AVAILABLE and len(history_store[fid]) >= 60:
            # train asynchronously to avoid blocking
            def train_task(series):
                try:
                    forecaster.train(series, epochs=3)
                except Exception as e:
                    print('LSTM train failed', e)
            threading.Thread(target=train_task, args=(history_store[fid][-500:],), daemon=True).start()
        return {'status':'stored','farm_id':fid,'len':len(history_store[fid])}

    @app.post('/forecast')
    async def forecast(req: Dict[str, Any]):
        fid = req.get('farm_id','default')
        horizon = int(req.get('horizon',7))
        hist = history_store.get(fid, [])
        preds = forecaster.forecast(hist, horizon=horizon)
        return {'farm_id': fid, 'horizon': horizon, 'predictions': preds}

    return app

# -------------------------
# Vision Agent - accepts image bytes and analyzes
# -------------------------

def create_vision_agent(app_name: str = 'vision_agent', port: int = 8402):
    app = FastAPI(title=app_name)
    vision = build_vision_model()

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    @app.post('/analyze')
    async def analyze(payload: Dict[str, Any]):
        # expects {image_bytes_base64, meta}
        img_b64 = payload.get('image_base64')
        if not img_b64:
            raise HTTPException(status_code=400, detail='missing image')
        import base64
        try:
            b = base64.b64decode(img_b64)
            out = vision.analyze_image_bytes(b)
            return {'agent': app_name, 'result': out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app

# -------------------------
# Agent: Inventory & Supply Agent
# -------------------------

def create_agent_ai(app_name: str = 'agent_ai', port: int = 8401):
    app = FastAPI(title=app_name)
    inventory: Dict[str, Dict[str, Any]] = {}

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    @app.post('/update_inventory')
    async def update(inv: Dict[str, Any]):
        pid = inv.get('product_id')
        if not pid:
            raise HTTPException(status_code=400, detail='missing product_id')
        inventory[pid] = {**inventory.get(pid, {}), **inv}
        return {'status':'ok','product':pid,'inventory':inventory[pid]}

    @app.post('/recommend')
    async def recommend(req: Dict[str, Any]):
        preds = req.get('predictions', [])
        product = req.get('product_id','milk')
        current = inventory.get(product, {}).get('qty', 0)
        avg = float(np.mean(preds)) if preds else 0.0
        reorder_threshold = avg * 1.2
        if current < reorder_threshold:
            qty = max(1, int(round(reorder_threshold - current)))
            return {'action':'reorder','product':product,'qty':qty,'reason':f'current<{reorder_threshold:.2f}'}
        return {'action':'hold','product':product,'current':current,'reason':'sufficient stock'}

    return app

# -------------------------
# Bot Interface (LLM + dynamic CoT) - sends IoT signals when needed
# -------------------------

def create_bot(app_name: str = 'bot', port: int = 8501, vectorstore: Optional[VectorStoreStub] = None, orchestrator_url: str = 'http://localhost:8000'):
    app = FastAPI(title=app_name)
    llm = setup_llm()
    vs = vectorstore or VectorStoreStub()
    vs.add('mastitis','Early mastitis signs: elevated cow_temp, reduced milk yield, change in milk conductivity, swollen udder.')
    vs.add('cheese','Cheese optimization: monitor fermentation temp, pH, and starter cultures')

    @app.get('/health')
    def health():
        return {'status':'ok','agent':app_name}

    def build_cot_prompt(question: str, context: str, detailed: bool = False):
        # dynamic CoT toggles deeper reasoning steps
        cot_instructions = """
You are SmartDairy reasoning assistant. Follow step-by-step chain-of-thought if requested.
Steps:
1) Identify key signals from context
2) List possible root causes (2-3)
3) Propose immediate actions (3 bullets)
4) Suggest follow-up checks and data to collect
Respond in JSON with keys: cot_steps, action_plan, follow_up
""" if detailed else "Provide concise actionable recommendations (3 bullets) in JSON with key 'action_plan'."
        prompt = f"Context:
{context}

Question:
{question}

Instructions:
{cot_instructions}"
        return prompt

    async def call_llm(prompt: str):
        if not llm:
            return {'text': 'LLM not available, fallback used.'}
        try:
            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, 'content') else str(resp)
            return {'text': text}
        except Exception as e:
            return {'text': f'LLM call failed: {e}'}

    async def notify_orchestrator_action(action_payload: Dict[str, Any]):
        # sends a POST to orchestrator to create an action event which orchestrator will route to IoT
        import httpx
        url = orchestrator_url.rstrip('/') + '/bot_action'
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                r = await client.post(url, json=action_payload)
                return r.json()
            except Exception as e:
                return {'error': str(e)}

    @app.post('/query')
    async def query(req: Dict[str, Any]):
        q = req.get('query','')
        farm_id = req.get('farm_id')
        detailed = bool(req.get('detailed', False))
        if not q:
            raise HTTPException(status_code=400, detail='missing query')
        retrieved = vs.search(q, k=4)
        context_text = '

'.join([d['text'] for d in retrieved]) if retrieved else ''
        prompt = build_cot_prompt(q, context_text, detailed=detailed)
        llm_resp = await call_llm(prompt)

        # If detailed CoT indicates immediate risk, create actuate signal (heuristic)
        text = llm_resp.get('text','')
        lower = text.lower()
        should_act = any(tok in lower for tok in ['mastitis','urgent','immediate','stop delivery','hold milk'])
        action_result = None
        if should_act and farm_id:
            act = {'farm_id': farm_id, 'action': 'investigate', 'params': {'reason_excerpt': text[:300]}}
            action_result = await notify_orchestrator_action({'source':'bot','action': act, 'timestamp': time.time()})
        return {'query': q, 'context': [d for d in retrieved], 'llm': llm_resp, 'action_result': action_result}

    return app

# -------------------------
# Orchestrator - routes events, handles bot actions and relays to IoT
# -------------------------

def create_orchestrator(port: int = 8000, iot_base: str = 'http://localhost:8201'):
    app = FastAPI(title='orchestrator')
    bus = InMemoryBus()
    vectorstore = VectorStoreStub()
    logs: List[Dict[str, Any]] = []

    agent_endpoints = {
        'iot': iot_base.rstrip('/') + '/predict',
        'predictive': 'http://localhost:8301/forecast',
        'agent_ai': 'http://localhost:8401/recommend',
        'vision': 'http://localhost:8402/analyze',
        'bot': 'http://localhost:8501/query'
    }

    @app.get('/health')
    def health():
        return {'status':'ok','role':'orchestrator'}

    @app.post('/ingest')
    async def ingest(event: Dict[str, Any]):
        event.setdefault('timestamp', time.time())
        await bus.publish('events', event)
        return {'status':'accepted','event_id': event.get('event_id')}

    async def call_agent_http(url: str, payload: Dict[str, Any], timeout: float = 8.0):
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
            return r.json()

    async def handle_event(event: Dict[str, Any]):
        etype = event.get('type')
        if etype == 'iot_event':
            agent_url = agent_endpoints.get('iot')
            payload = {'event_id': event.get('event_id'), 'event_type': etype, 'payload': event.get('payload')}
            resp = await call_agent_http(agent_url, payload)
            logs.append({'time': time.time(), 'event': event, 'agent_resp': resp})
            # if anomaly detected, call bot for CoT detailed reasoning
            if resp.get('prediction', {}).get('anomaly') == 1:
                q = f"Farm {event.get('payload',{}).get('farm_id','?')} anomaly: milk={event.get('payload',{}).get('milk_yield')}, temp={event.get('payload',{}).get('cow_temp')}"
                bot_url = agent_endpoints.get('bot')
                bot_resp = await call_agent_http(bot_url, {'query': q, 'farm_id': event.get('payload',{}).get('farm_id'), 'detailed': True})
                logs.append({'time': time.time(), 'bot': bot_resp})
                # if bot requested action_result, relay to IoT actuate endpoint
                act = bot_resp.get('action_result')
                if act and isinstance(act, dict) and not act.get('error'):
                    # orchestrator will also push immediate actuate to iot
                    iot_act_url = iot_base.rstrip('/') + '/actuate'
                    await call_agent_http(iot_act_url, act.get('action'))
        elif etype == 'vision_check':
            agent_url = agent_endpoints.get('vision')
            payload = event.get('payload', {})
            resp = await call_agent_http(agent_url, payload)
            logs.append({'time': time.time(), 'vision_resp': resp})
        else:
            logs.append({'time': time.time(), 'event': event})

    async def dispatcher_loop():
        print('Orchestrator dispatcher started')
        sub = bus.subscribe()
        async for topic, event in sub:
            try:
                await handle_event(event)
            except Exception as e:
                print('Error handling event:', e)

    @app.post('/bot_action')
    async def bot_action(payload: Dict[str, Any]):
        # Bot posts here to ask for orchestrator to route actions
        action = payload.get('action')
        if not action:
            return {'error': 'missing action'}
        # forward to IoT actuate endpoint
        iot_act = iot_base.rstrip('/') + '/actuate'
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(iot_act, json=action)
                return {'forwarded': True, 'iot_resp': r.json()}
        except Exception as e:
            return {'error': str(e)}

    @app.get('/logs')
    def get_logs():
        return {'logs': logs[-200:]}

    @app.on_event('startup')
    async def startup_event():
        loop = asyncio.get_event_loop()
        loop.create_task(dispatcher_loop())

    return app

# -------------------------
# Simulation generators (IoT + Images)
# -------------------------
async def simulate_iot_events(orchestrator_url: str = 'http://localhost:8000/ingest', n: int = 100, interval: float = 0.5):
    import httpx
    async with httpx.AsyncClient() as client:
        for i in range(n):
            ev = {
                'event_id': f'IOT_{i}_{int(time.time())}',
                'type': 'iot_event',
                'timestamp': time.time(),
                'payload': {
                    'farm_id': f'farm_{random.randint(1,5)}',
                    'milk_yield': round(np.random.normal(18, 8), 2),
                    'cow_temp': round(np.random.normal(38.6, 1.0), 2),
                    'feed_intake': round(np.random.normal(24, 8), 2),
                    'ambient_temp': round(np.random.normal(22, 9), 2),
                }
            }
            try:
                r = await client.post(orchestrator_url, json=ev)
                print('Produced', ev['event_id'], '->', r.status_code)
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
    run_orch.add_argument('--port', default=8000, type=int)

    run_agent = sub.add_parser('run_agent', help='Run an agent service')
    run_agent.add_argument('--agent-type', choices=['iot','predictive','agent_ai','vision','bot'], required=True)
    run_agent.add_argument('--port', type=int, required=True)

    args = parser.parse_args()

    if args.cmd == 'run_all_sim':
        def start_app(app, port):
            uvicorn.run(app, host='0.0.0.0', port=port)

        orch_app = create_orchestrator()
        iot_app = create_iot_agent()
        pred_app = create_predictive_agent()
        agent_app = create_agent_ai()
        vision_app = create_vision_agent()
        bot_app = create_bot()

        threads = []
        threads.append(threading.Thread(target=start_app, args=(orch_app,8000), daemon=True))
        threads.append(threading.Thread(target=start_app, args=(iot_app,8201), daemon=True))
        threads.append(threading.Thread(target=start_app, args=(pred_app,8301), daemon=True))
        threads.append(threading.Thread(target=start_app, args=(agent_app,8401), daemon=True))
        threads.append(threading.Thread(target=start_app, args=(vision_app,8402), daemon=True))
        threads.append(threading.Thread(target=start_app, args=(bot_app,8501), daemon=True))

        for t in threads:
            t.start()
        print('All services started (orch@8000, iot@8201, pred@8301, agent@8401, vision@8402, bot@8501)')

        asyncio.run(simulate_iot_events(n=300, interval=0.25))
        print('Simulation finished — services still running. Ctrl+C to exit.')
        while True:
            time.sleep(1)

    elif args.cmd == 'run_orchestrator':
        uvicorn.run(create_orchestrator(port=args.port), host='0.0.0.0', port=args.port)

    elif args.cmd == 'run_agent':
        if args.agent_type == 'iot':
            uvicorn.run(create_iot_agent(port=args.port), host='0.0.0.0', port=args.port)
        elif args.agent_type == 'predictive':
            uvicorn.run(create_predictive_agent(port=args.port), host='0.0.0.0', port=args.port)
        elif args.agent_type == 'agent_ai':
            uvicorn.run(create_agent_ai(port=args.port), host='0.0.0.0', port=args.port)
        elif args.agent_type == 'vision':
            uvicorn.run(create_vision_agent(port=args.port), host='0.0.0.0', port=args.port)
        else:
            uvicorn.run(create_bot(port=args.port), host='0.0.0.0', port=args.port)
    else:
        parser.print_help()
