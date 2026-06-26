# SentinelSite — ML Master Reference (Senior Engineer Brain Dump)
> Use this as the single source of truth before implementing anything. No need to re-read all 5 docs.

---

## 0. WHAT THIS PRODUCT IS (30-second version)

**Passive first-person near-miss detection system for construction workers wearing Meta Ray-Ban glasses.**

- Worker does NOTHING. System witnesses for them.
- Glasses → Bluetooth → Android Phone (edge compute) → FastAPI Cloud → React Dashboard (supervisor)
- Two independent signals → AND gate → event logged → supervisor reviews → confirmed events → retrain model

---

## 1. THE FOUR-LAYER DECISION MAP

```
WHEN ASKED TO IMPLEMENT SOMETHING, LOCATE IT HERE FIRST:

┌────────────────────────────────────────────────────────┐
│  LAYER 1: GLASSES (Meta Ray-Ban)                       │
│  Role: Sensor ONLY. No compute, no storage, no net.    │
│  Outputs: audio stream, IMU readings, on-demand JPEG   │
│  SDK: Meta Wearables DAT                               │
│  Code location: android/.../imu/, audio/, vision/      │
└───────────────────────┬────────────────────────────────┘
                        │ Bluetooth
┌───────────────────────▼────────────────────────────────┐
│  LAYER 2: ANDROID PHONE (Edge Compute) — Kotlin        │
│  Role: All real-time ML inference                      │
│  Models live here as TFLite assets                     │
│  4 threads running constantly:                         │
│    T1: AudioBuffer → YAMNet → AnomalyScorer            │
│    T2: IMU → JerkDetector                              │
│    T3: EventHandler (fires on trigger only)            │
│    T4: VoiceCopilot (on-demand)                        │
│  Code: android/app/src/main/java/com/sentinelsite/     │
└───────────────────────┬────────────────────────────────┘
                        │ HTTPS/LTE/WiFi (event-driven)
┌───────────────────────▼────────────────────────────────┐
│  LAYER 3: CLOUD BACKEND — FastAPI + Python             │
│  Role: Storage, RAG, training orchestration            │
│  Stack: FastAPI + Postgres + Qdrant + Redis + Celery   │
│  Code: backend/app/                                    │
│    api/ → routers                                      │
│    ml/ → server-side inference (YAMNet recheck, GPT4V) │
│    rag/ → ingestion, retrieval, llm_chain              │
│    training/ → replay_buffer, schedulers, trainers     │
│    db/ → SQLAlchemy ORM                                │
└───────────────────────┬────────────────────────────────┘
                        │ HTTPS + WebSocket
┌───────────────────────▼────────────────────────────────┐
│  LAYER 4: REACT DASHBOARD — Vite + Tailwind + Zustand  │
│  Role: Supervisor review, admin training, analytics    │
│  Code: dashboard/src/                                  │
│    components/alerts/, heatmap/, analytics/, admin/    │
└────────────────────────────────────────────────────────┘
```

---

## 2. ML COMPONENTS — DECISION TREE

```
WHAT ARE YOU IMPLEMENTING?
│
├─► ACOUSTIC DETECTION
│   ├─ Model: YAMNet (MobileNet-v1 backbone), TFLite, INT8 quantized
│   ├─ Input: 0.96s audio window @ 16kHz mono
│   ├─ Output: 521-class probability (AudioSet)
│   ├─ Key classes: Crash(373), Bang(374), Thud(376), Shout(44),
│   │              Screaming(45), Breaking(378), Alarm(388), Impact(375)
│   ├─ EXCLUDE from anomaly score: Drill(474), Jackhammer(476), Sawing(479),
│   │              Engine(355), Compressor(472) — these are site baseline
│   ├─ Threshold: θ₁ — calibrated per site via 60s baseline recording
│   │   Algorithm: RMS Z-score → sigmoid → anomaly_score (0–1)
│   │   Calibration: 3σ above baseline RMS = θ₁ default
│   ├─ File: ml/acoustic/acoustic_baseline_calibrator.py
│   ├─ Android: YAMNetInferenceEngine.kt, AnomalyScorer.kt
│   └─ Training: ml/acoustic/acoustic_fine_tuning.py
│                backend/app/training/acoustic_trainer.py
│
├─► IMU / STARTLE DETECTION
│   ├─ Sensor: Gyroscope @ 50Hz (glasses or phone fallback)
│   ├─ Signal: Angular velocity [x, y, z] → jerk = |dω/dt| magnitude
│   ├─ Threshold: θ₂ — configured in ThresholdConfig.kt
│   ├─ Android: IMUManager.kt, JerkDetector.kt, MotionBaseline.kt
│   └─ No ML here — pure signal processing / threshold comparison
│
├─► FUSION GATE (AND gate — most critical for FP reduction)
│   ├─ Logic: BOTH acoustic anomaly (>θ₁) AND IMU jerk (>θ₂)
│   │         must occur within ±2000ms of each other
│   ├─ On trigger: 30s cooldown (no re-triggering)
│   ├─ Android: FusionGate.kt, EventTriggerController.kt
│   └─ No ML — pure stateful logic
│
├─► VISUAL CLASSIFICATION
│   ├─ Model: MobileNet-v3-Small, backbone FROZEN (ImageNet weights)
│   ├─ Head only trainable: Linear(576→256) → Hardswish → Dropout(0.3) → Linear(256→N)
│   ├─ Input: 224×224 RGB frame from glasses camera
│   ├─ Augmentation: flip, brightness±30%, rotation±15°, GaussianBlur
│   │   (simulates construction dust/lighting — critical for <20 images)
│   ├─ Loss: CrossEntropyLoss(label_smoothing=0.1) — prevents overconfidence
│   ├─ Optimizer: AdamW, lr=1e-3, CosineAnnealingLR scheduler
│   ├─ File: ml/visual/mobilenet_fine_tuning.py
│   ├─ Android: MobileNetInferenceEngine.kt, FramePreprocessor.kt
│   └─ Admin training: backend/app/training/admin_trainer.py
│
├─► VOICE COPILOT / RAG PIPELINE
│   ├─ Wake word: "Hey Sentinel" → Porcupine (on-device, <5mA)
│   ├─ STT: Whisper Small TFLite → ~150ms on-device, no cloud needed
│   ├─ Intent Router: DistilBERT classifier → 8 doc types
│   │   Categories: STRUCTURAL, SAFETY, SCHEDULE, MATERIAL,
│   │               ELECTRICAL, PLUMBING, INSPECTION, GENERAL
│   │   Training data: GPT-4o generated synthetic Q&A from site docs
│   │   File: backend/app/rag/intent_classifier.py
│   ├─ Retrieval: Qdrant vector store, hybrid search + reranker
│   │   File: backend/app/rag/retriever.py
│   ├─ LLM: Claude claude-sonnet-4-6 or GPT-4o-mini, max_tokens=256
│   │   SYSTEM RULE: Answer ONLY from context. Always cite source.
│   │   If no context: "I couldn't find that in the site documents."
│   │   File: backend/app/rag/llm_chain.py
│   ├─ Ingestion: PDF → unstructured → chunks → text-embedding-3-small → Qdrant
│   │   Namespace: per site_id (strict isolation)
│   │   File: backend/app/rag/ingestion.py
│   └─ Android: WhisperSTTEngine.kt, IntentRouter.kt, RAGQueryDispatcher.kt
│
├─► SELF-LEARNING PIPELINE (continual learning)
│   ├─ Trigger conditions (ALL must be true):
│   │   a) ≥ 20 new confirmed samples since last training
│   │   b) ≥ 7 days since last training run
│   │   c) GPU utilization < 50%
│   │   Check interval: every 6 hours via APScheduler
│   ├─ Data flow: supervisor Confirm → training_samples table → replay buffer
│   ├─ REPLAY BUFFER — critical, prevents catastrophic forgetting
│   │   Ratio: 70% historical + 30% new samples ALWAYS
│   │   Strategies: random | class_balanced (default) | uncertainty_weighted
│   │   class_balanced: equal samples per class → rare events preserved
│   │   uncertainty_weighted: prioritize samples with low model confidence
│   │   File: backend/app/training/replay_buffer.py (ExperienceReplayBuffer)
│   ├─ Promotion gate: new model val_accuracy ≥ previous model accuracy
│   ├─ Rollback: if post-deploy accuracy drops >5% → auto rollback <10min
│   ├─ OTA delivery: S3 presigned URL → device downloads on WiFi+charging
│   │   Applied on next cold start ONLY (never hot-swap during inference)
│   └─ Files: training/scheduler.py, acoustic_trainer.py, quantizer.py
│
└─► ADMIN TRAINING (few-shot fine-tuning)
    ├─ Minimum: 15 images per new class
    ├─ Backbone: FROZEN. Head only trained.
    ├─ Epochs: 20, lr=1e-3
    ├─ Promotion: val_acc(new class) > 70% AND existing classes within 5%
    ├─ File: backend/app/training/admin_trainer.py
    └─ Dashboard: admin/ImageUploader.tsx, TrainingStatus.tsx
```

---

## 3. DATA FLOW — NEAR MISS EVENT (end-to-end)

```
[Glasses Mic] → BT → [Phone: AudioRecord 16kHz]
                           │
                    [CircularBuffer 30s]
                           │ every 1s
                    [YAMNet TFLite] → anomaly_score
                           │ if score > θ₁
                    [AudioAnomalyEvent → EventBus]
                                              ↓
[Glasses IMU] → BT → [JerkDetector 50Hz]           ← [EventBus]
                           │ if jerk > θ₂
                    [IMUJerkEvent → EventBus]
                                              ↓
                                      [FusionGate]
                                      Δt ≤ 2000ms?
                                           YES
                                            │
                                    [NEAR_MISS_TRIGGER]
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                   Freeze 30s audio   FrameCapture        GPS fix
                          │           via BT glasses           │
                          └─────────────────┼──────────────────┘
                                            │
                                  [MobileNet classify frame]
                                            │
                                  [NearMissPayloadBuilder]
                                  {timestamp, GPS, audio_clip,
                                   frame, yamnet_class, yamnet_score,
                                   visual_class, visual_score,
                                   worker_id, device_id}
                                            │
                                  [SQLite UploadQueue]
                                            │ on LTE/WiFi
                                  [POST /api/v1/events]
                                            │
                          ┌─────────────────┴──────────────────┐
                          ▼                                     ▼
                   [S3: audio + frame]              [Postgres: event record]
                          │                                     │
                   [Celery: GPT-4o Vision]          [WebSocket → Dashboard]
                          │                                     │
                   frame description                   [Supervisor Alert Card]
                                                              │
                                              [Confirm → training_samples]
                                              [Dismiss → false positive log]
```

---

## 4. KEY THRESHOLDS & NUMBERS (memorize these)

| Parameter | Value | Note |
|-----------|-------|------|
| Audio window | 0.96s @ 16kHz | YAMNet fixed input |
| Circular buffer | 30s × 16000 = 480k samples | Pre-event context |
| IMU sampling | 50Hz | Gyroscope |
| Fusion window | ±2000ms | AND gate tolerance |
| Cooldown after trigger | 30s | Prevents re-triggering |
| θ₁ default formula | baseline_rms + 3σ | Per-site calibration |
| Calibration duration | 60 seconds | Shift-start baseline |
| TFLite YAMNet inference | ~100ms CPU, ~15ms NPU | On-device |
| Frame capture deadline | <500ms after trigger | FR-D06 |
| Voice response target | <3s end-to-end | STT+RAG+LLM+TTS |
| Training trigger: min samples | 20 confirmed events | FR-L02a |
| Training trigger: min interval | 7 days | FR-L02b |
| Training trigger: max GPU | <50% utilization | FR-L02c |
| Replay ratio | 70% historical : 30% new | FR-L03 |
| Admin training min images | 15 per class | FR-A01 |
| Admin training epochs | 20 | Visual head |
| Admin val accuracy target | >70% new class, <5% drop existing | Promotion gate |
| Self-learning val accuracy | ≥ previous model | Promotion gate |
| YAMNet accuracy target | >80% TPR @ <5% FPR | NFR-AC01 |
| Visual accuracy target | >75% site-specific | NFR-AC02 |
| RAG hit rate target | >85% in top-3 | NFR-AC03 |
| Intent routing target | >90% on 8 classes | NFR-AC04 |

---

## 5. FILE → PURPOSE MAP (quick navigation)

### Backend (Python)
```
backend/app/
├── api/events.py          POST /api/v1/events — near-miss ingest
├── api/voice.py           POST /api/v1/voice/query — RAG endpoint
├── api/training.py        POST /api/v1/admin/train — trigger job
├── api/models.py          GET /api/v1/models/latest — OTA check
├── api/ws.py              WS /ws/alerts — supervisor real-time
├── ml/yamnet_server.py    Server-side YAMNet recheck (full quality)
├── ml/vision_describer.py GPT-4o Vision API for frame descriptions
├── ml/risk_scorer.py      Zone × time risk computation
├── rag/ingestion.py       PDF → chunks → Qdrant
├── rag/retriever.py       Hybrid search + reranker
├── rag/llm_chain.py       LangChain LCEL: context → Claude/GPT answer
├── rag/intent_classifier.py DistilBERT → doc type
├── training/replay_buffer.py ExperienceReplayBuffer (class_balanced)
├── training/scheduler.py  APScheduler: check every 6h, trigger if conditions
├── training/acoustic_trainer.py YAMNet head fine-tuning
├── training/visual_trainer.py  MobileNet head fine-tuning
├── training/admin_trainer.py   Admin image upload → fine-tune → push
├── training/quantizer.py  PyTorch → ONNX → TFLite INT8
└── training/model_pusher.py    Push model OTA
```

### Android (Kotlin)
```
audio/    YAMNetInferenceEngine.kt, AnomalyScorer.kt, AudioBufferManager.kt
imu/      IMUManager.kt, JerkDetector.kt
fusion/   FusionGate.kt, EventTriggerController.kt, NearMissPayloadBuilder.kt
vision/   MobileNetInferenceEngine.kt, FrameCaptureManager.kt
voice/    WakeWordDetector.kt, WhisperSTTEngine.kt, IntentRouter.kt
upload/   UploadQueueManager.kt, SyncWorker.kt
model/    ModelUpdateManager.kt, ModelValidator.kt
```

### ML Workspace (Python, dev only)
```
ml/acoustic/     yamnet fine-tuning, threshold sweep, noise robustness
ml/visual/       mobilenet fine-tuning, augmentation, few-shot eval
ml/rag/          ingestion pipeline, intent classifier training, retrieval eval
ml/training/     replay buffer simulation, forgetting benchmark, TFLite export
ml/evaluation/   end-to-end latency, FPR tests, accuracy tracker
```

---

## 6. TECH STACK QUICK REFERENCE

| Layer | Key Tech |
|-------|----------|
| Android | Kotlin, TFLite 2.14, Meta Wearables SDK, WorkManager, Room SQLite, Ktor |
| Backend | FastAPI 0.111, SQLAlchemy 2.0, Celery 5.3, APScheduler, boto3 |
| ML Training | PyTorch 2.3, torchvision 0.18, TF 2.16, ONNX 1.16, onnx2tf |
| RAG | LangChain 0.2, Qdrant 1.9, unstructured[pdf], OpenAI embeddings |
| LLM | Claude claude-sonnet-4-6 (primary), gpt-4o-mini (fallback), GPT-4o Vision |
| Vector DB | Qdrant (namespaced by site_id) |
| Storage | S3 (AES-256), Postgres 16 |
| Cache/Queue | Redis 7 (Celery broker + WS bus) |
| Dashboard | React 18, Vite, Zustand, Recharts, Leaflet, Tailwind, WaveSurfer |
| Infra | Docker Compose: api, worker, beat, postgres, qdrant, redis, nginx |

---

## 7. CRITICAL IMPLEMENTATION RULES (never violate)

1. **Training NEVER runs on-device.** Cloud GPU only. On-device = inference only.
2. **Replay buffer ratio ALWAYS 70% historical / 30% new.** Non-negotiable.
3. **New model NEVER hot-swapped.** Applied only on next cold start.
4. **Rollback is automated.** If post-deploy accuracy drops >5%, rollback in <10 min.
5. **MobileNet backbone always FROZEN.** Only classification head trains.
6. **RAG must cite source.** "Per [Document], Section [X]..." — no hallucination.
7. **Audio clips stored ONLY on trigger.** Never continuous upload (privacy).
8. **Fusion gate requires BOTH signals.** Single signal alone = ignore.
9. **Site isolation is mandatory.** Every Qdrant collection, S3 prefix namespaced by site_id.
10. **Voice queries NOT logged** unless worker opts in (NFR-S01).

---

## 8. THE HONEST RISKS (what to watch for)

| Risk | Level | Mitigation |
|------|-------|-----------|
| YAMNet accuracy at 90dB ambient noise | HIGH | Threshold sweep pre-deployment; noise-robust fine-tuning path |
| TFLite export pipeline (PyTorch→ONNX→TFLite) | HIGH | Test Day 0, not Day 2; known compat issues with some ops |
| FPR > 2/hr during normal construction activity | MEDIUM | θ₁ tuning; fusion gate reduces vs single signal |
| Voice response >3s on real LTE | MEDIUM | Measure on real LTE, not office WiFi |
| Replay buffer only as good as supervisor labels | MEDIUM | Dashboard UX: confirm/dismiss <10s per event |
| Catastrophic forgetting without replay proof | LOW | Run forgetting_benchmark.py, save plot — show to judges |

---

## 9. WHEN ASKED TO IMPLEMENT X — USE THIS DECISION TREE

```
"Implement X" →

Is X about audio/sound processing?
  YES → YAMNet path. Check ml/acoustic/. On-device = TFLite.
        Cloud = yamnet_server.py (full quality recheck)

Is X about visual/camera/frame?
  YES → MobileNet path. ml/visual/ for training.
        On-device = MobileNetInferenceEngine.kt
        Admin training = admin_trainer.py

Is X about detecting/logging events?
  YES → FusionGate.kt + EventTriggerController.kt (Android)
        POST /api/v1/events (cloud ingest)

Is X about voice / questions / documents?
  YES → Voice stack: WakeWord → Whisper STT → IntentRouter → RAG
        RAG stack: ingestion.py → retriever.py → llm_chain.py

Is X about training / improving models?
  YES → Self-learning path:
        Trigger: scheduler.py (check conditions)
        Data: replay_buffer.py (70/30 split, class_balanced)
        Train: acoustic_trainer.py or visual_trainer.py
        Export: quantizer.py (PyTorch → TFLite)
        Deploy: model_pusher.py → OTA → cold start swap

Is X about the dashboard / UI?
  YES → React components in dashboard/src/components/
        Alerts: AlertFeed.tsx (WebSocket), AlertCard.tsx
        Admin: ImageUploader.tsx, TrainingStatus.tsx
        Analytics: TrendChart.tsx, SiteHeatmap.tsx

Is X a new API endpoint?
  YES → Add to backend/app/api/, register in main.py
        Auth: JWT + RBAC (worker/supervisor/admin/system)

Is X about storage?
  YES → Audio/frames/models → S3 (via storage.py)
        Events/training data → Postgres (via db/models.py)
        Vectors → Qdrant (namespaced by site_id)
        Upload queue (phone offline) → SQLite Room
```

---

## 10. MVP vs V2 SCOPE

### MVP (must work for demo)
- End-to-end: acoustic event → fusion → payload → supervisor alert card
- Audio playback on dashboard (30s clip)
- Frame viewer + GPT-4o scene description
- Confirm/Dismiss interaction
- Site heatmap with GPS pins
- OSHA PDF export (one event)
- Voice Q&A with intent routing (demo scenario)
- RAG document ingestion (PDF upload)

### V2 (post-hackathon)
- Multilingual voice (Whisper auto-detects)
- 2-turn conversation context
- IMU motion baseline (per-worker)
- Document replacement flow
- Retrieval testing UI
- Trend analytics / time-of-day heatmap
- Worker risk ranking (anonymized)
- Weekly digest email
- Date-range CSV export
