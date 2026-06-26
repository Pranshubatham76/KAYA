# SentinelSite — System Architecture
**Version:** 1.0 | **Last Updated:** June 2026

---

## 1. Architecture Overview

SentinelSite is a four-layer distributed system. Each layer has a single, clear responsibility. No layer does what another layer should do.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — SENSOR LAYER          Meta Ray-Ban Glasses                   │
│  Responsibility: Raw signal collection only                              │
│  Processes: Nothing. Passes everything via BT.                          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ Bluetooth (Meta Wearables SDK)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — EDGE COMPUTE LAYER     Android Phone                         │
│  Responsibility: All on-device ML inference + event packaging           │
│  Processes: YAMNet, IMU jerk, fusion gate, Whisper STT, intent router   │
│  Stores: SQLite upload queue, model files, 30s audio ring buffer        │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ HTTPS / LTE / WiFi (event-driven)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — CLOUD BACKEND          FastAPI + Postgres + Qdrant + Redis   │
│  Responsibility: Event storage, analysis, RAG, training orchestration    │
│  Processes: YAMNet recheck, GPT-4o Vision, RAG retrieval, training jobs │
│  Stores: All events, models, documents, training samples, analytics     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ HTTPS REST + WebSocket
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — PRESENTATION LAYER     React Web Dashboard                   │
│  Responsibility: Supervisor review, admin training, analytics display    │
│  Consumes: WebSocket alerts, REST analytics, training status            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Layer Architecture

### Layer 1 — Glasses (Sensor Only)

```
Meta Ray-Ban Gen 2 / Display Model
├── Microphone          → captures ambient audio
│   └── Streams via Bluetooth BT audio profile to phone
├── IMU (Accel + Gyro) → captures head motion
│   └── Exposed via Meta Wearables SDK to phone
└── Camera             → standby; trigger-only capture
    └── SDK: requestCapture() → JPEG delivered to phone callback
```

**What the glasses do NOT do:**
- No inference
- No processing
- No storage
- No network calls

The glasses are a sensor array. All intelligence is elsewhere.

---

### Layer 2 — Android Phone (Edge Compute)

This is the most architecturally important layer. All real-time processing happens here.

```
┌──────────────────────────────────────────────────────────────────┐
│              ANDROID PHONE — DETAILED ARCHITECTURE              │
└──────────────────────────────────────────────────────────────────┘

THREAD 1 — AudioBufferThread (runs continuously, low priority)
┌─────────────────────────────────────────────────────────────┐
│  AudioRecord (16kHz, mono, 16-bit PCM)                      │
│    └── writes to CircularBuffer<Float>[30s × 16000 = 480k]  │
│                                                             │
│  Every 1 second → extract 0.96s window from buffer         │
│    └── convert to mel spectrogram                           │
│    └── YAMNetInferenceEngine.classify(window)               │
│         └── returns: class_id, class_name, confidence       │
│    └── AnomalyScorer.score(class_id, confidence, baseline)  │
│         └── returns: anomaly_score (0.0–1.0)                │
│    └── if anomaly_score > θ₁ → post to EventBus            │
└─────────────────────────────────────────────────────────────┘

THREAD 2 — IMUReaderThread (runs continuously, low priority)
┌─────────────────────────────────────────────────────────────┐
│  SensorManager (SENSOR_TYPE_GYROSCOPE, 50Hz)                │
│    └── reads angular velocity vector [x, y, z]             │
│    └── JerkDetector.compute(ω_current, ω_previous, Δt)      │
│         └── jerk = |dω/dt| magnitude                        │
│    └── if jerk > θ₂ → post to EventBus                     │
└─────────────────────────────────────────────────────────────┘

EVENT BUS (in-memory, lightweight)
┌─────────────────────────────────────────────────────────────┐
│  FusionGate subscribes to both AudioAnomalyEvent            │
│  and IMUJerkEvent                                           │
│                                                             │
│  FusionGate logic:                                          │
│    pendingAudioEvent = null                                 │
│    pendingIMUEvent = null                                   │
│                                                             │
│    on AudioAnomalyEvent(t_audio):                           │
│      pendingAudioEvent = t_audio                            │
│      checkFusion()                                          │
│                                                             │
│    on IMUJerkEvent(t_imu):                                  │
│      pendingIMUEvent = t_imu                                │
│      checkFusion()                                          │
│                                                             │
│    checkFusion():                                           │
│      if pendingAudio AND pendingIMU:                        │
│        if |t_audio - t_imu| <= 2000ms:                     │
│          → NEAR_MISS_TRIGGER                                │
│          reset both pending events                          │
│          start cooldown (30s — don't re-trigger)           │
└─────────────────────────────────────────────────────────────┘

THREAD 3 — EventHandlerThread (fires on trigger only)
┌─────────────────────────────────────────────────────────────┐
│  on NEAR_MISS_TRIGGER:                                      │
│    1. freezeAudioBuffer()         → copy 30s to file        │
│    2. FrameCaptureManager.capture() → BT request to glasses │
│    3. LocationManager.getLastKnown() → GPS lat/lon          │
│    4. MobileNetInferenceEngine.classify(frame)              │
│         → visual_class, visual_confidence                   │
│    5. NearMissPayloadBuilder.build(                         │
│         timestamp, gps, audio_file, frame_file,             │
│         yamnet_class, yamnet_score,                         │
│         visual_class, visual_score,                         │
│         worker_id, device_id)                               │
│    6. UploadQueueManager.enqueue(payload)                   │
└─────────────────────────────────────────────────────────────┘

THREAD 4 — VoiceCopilotThread (on-demand, idle otherwise)
┌─────────────────────────────────────────────────────────────┐
│  WakeWordDetector (Porcupine, continuous, <5mA)             │
│    └── on "Hey Sentinel":                                   │
│         └── WhisperSTT.transcribe(audio_segment)           │
│         └── IntentRouter.classify(text)                     │
│              → document_type (8 categories)                 │
│         └── RAGQueryDispatcher.query(text, document_type)   │
│              → POST /api/v1/voice/query → answer            │
│         └── TTSPlayer.speak(answer)                         │
└─────────────────────────────────────────────────────────────┘

BACKGROUND SERVICES (WorkManager)
┌─────────────────────────────────────────────────────────────┐
│  SyncWorker — triggers on: network available                │
│    └── processes UploadQueue → POST events to server        │
│    └── exponential backoff on failure                       │
│                                                             │
│  ModelUpdateWorker — triggers on: charging + WiFi           │
│    └── GET /api/v1/models/latest                            │
│    └── if newer version: download to temp file              │
│    └── ModelValidator.smokeTest(new_model)                  │
│    └── if passes: stage for next cold start                 │
└─────────────────────────────────────────────────────────────┘
```

---

### Layer 3 — Cloud Backend

```
┌──────────────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND — DETAILED ARCHITECTURE            │
└──────────────────────────────────────────────────────────────────┘

API GATEWAY (FastAPI routers)
├── POST /api/v1/events          → EventsRouter
├── POST /api/v1/voice/query     → VoiceRouter
├── GET  /api/v1/models/latest   → ModelsRouter
├── POST /api/v1/admin/train     → TrainingRouter
├── POST /api/v1/admin/upload    → AdminRouter
├── GET  /api/v1/dashboard/*     → DashboardRouter
└── WS   /ws/alerts              → WebSocketRouter

EVENT INGEST PIPELINE (async)
┌─────────────────────────────────────────────────────────────┐
│  POST /api/v1/events                                        │
│    1. Validate payload (Pydantic model)                     │
│    2. Store audio_clip → S3 (presigned URL returned)        │
│    3. Store frame → S3                                      │
│    4. Write event record → Postgres                         │
│    5. Enqueue analysis task → Celery queue                  │
│    6. Return 202 Accepted immediately                       │
│                                                             │
│  Celery Worker — AnalysisTask:                              │
│    1. yamnet_server.reanalyze(audio_clip)                   │
│         → higher quality server-side classification         │
│    2. vision_describer.describe(frame)                      │
│         → GPT-4o Vision: natural language scene context     │
│    3. risk_scorer.update(event)                             │
│         → update zone risk scores                           │
│    4. Update event record with analysis results             │
│    5. ws_server.broadcast(event_id) → supervisor dashboard  │
└─────────────────────────────────────────────────────────────┘

RAG ENGINE
┌─────────────────────────────────────────────────────────────┐
│  Document Ingestion (triggered on PDF upload):              │
│    1. unstructured.io: PDF → structured chunks              │
│         (layout-aware: handles tables, annotations)         │
│    2. RecursiveCharacterTextSplitter                        │
│         chunk_size=800, chunk_overlap=150                   │
│    3. metadata tagging: {source, doc_type, page, site_id}  │
│    4. OpenAI text-embedding-3-small: chunks → vectors       │
│    5. Qdrant upsert: collection per site + doc type         │
│                                                             │
│  Query Pipeline:                                            │
│    POST /api/v1/voice/query {text, site_id, worker_id}     │
│    1. intent_classifier.classify(text)                      │
│         → doc_type (Structural/Safety/Schedule/...)        │
│    2. embed query → vector                                  │
│    3. Qdrant hybrid search: filter by {site_id, doc_type}  │
│         → top-5 chunks (semantic + keyword combined)        │
│    4. Cohere Rerank: reorder by relevance                  │
│    5. LLM chain (Claude claude-sonnet-4-6):                        │
│         system: "Answer only from context. Always cite."   │
│         → grounded answer + source reference               │
│    6. Return {answer, source, confidence}                   │
└─────────────────────────────────────────────────────────────┘

TRAINING ORCHESTRATION (the self-learning core)
┌─────────────────────────────────────────────────────────────┐
│  TrainingQueue (Postgres table: training_samples)           │
│    Populated when: supervisor clicks Confirm on alert       │
│    Fields: event_id, audio_s3_url, frame_s3_url,           │
│            label (YAMNet class), osha_category,             │
│            is_used_in_training (bool), site_id              │
│                                                             │
│  APScheduler — checks every 6 hours:                       │
│    trigger_conditions:                                      │
│      new_samples = COUNT(*) WHERE is_used = false           │
│      if new_samples >= 20                                   │
│      AND days_since_last_training >= 7                      │
│      AND gpu_util < 0.5:                                    │
│        → dispatch TrainingJob to Celery                     │
│                                                             │
│  TrainingJob — AcousticTrainer:                             │
│    1. ReplayBuffer.sample(                                  │
│         n_new=new_samples,                                  │
│         n_historical=int(new_samples * 70/30))              │
│    2. Load YAMNet backbone (frozen)                         │
│    3. Initialize/load existing classification head          │
│    4. Train head only:                                      │
│         epochs=30, lr=1e-3, batch=32                        │
│         loss=CrossEntropy, optimizer=AdamW                  │
│         scheduler=CosineAnnealingLR                         │
│    5. Evaluate on held-out validation split (80/20)         │
│    6. if val_accuracy >= previous_accuracy:                 │
│         → quantize to INT8 TFLite (quantizer.py)           │
│         → store in S3 with version tag                      │
│         → update ModelVersionStore (Postgres)               │
│         → mark samples as is_used = true                    │
│       else:                                                 │
│         → reject model, keep previous                       │
│         → notify admin via dashboard                        │
│                                                             │
│  Same flow for VisualTrainer (admin images)                 │
└─────────────────────────────────────────────────────────────┘

DATABASE SCHEMA (Postgres)
┌─────────────────────────────────────────────────────────────┐
│  near_miss_events                                           │
│    id, worker_id, site_id, timestamp_utc, gps_lat,         │
│    gps_lon, yamnet_class, yamnet_score, visual_class,       │
│    visual_score, audio_s3_url, frame_s3_url,               │
│    gpt4v_description, review_status, osha_category,         │
│    confirmed_by, confirmed_at, severity                     │
│                                                             │
│  training_samples                                           │
│    id, event_id, audio_s3_url, frame_s3_url,               │
│    label_acoustic, label_visual, osha_category,             │
│    site_id, is_used_in_training, created_at                 │
│                                                             │
│  model_versions                                             │
│    id, model_type (acoustic/visual), version_tag,           │
│    s3_url, val_accuracy, training_sample_count,             │
│    site_id, status (active/retired/failed), created_at      │
│                                                             │
│  site_documents                                             │
│    id, site_id, filename, doc_type, s3_url,                │
│    ingestion_status, chunk_count, created_at               │
│                                                             │
│  admin_training_images                                      │
│    id, site_id, s3_url, label, is_used, created_at         │
│                                                             │
│  sites, workers, devices (standard entity tables)           │
└─────────────────────────────────────────────────────────────┘
```

---

### Layer 4 — Dashboard

```
React SPA (Vite + Tailwind + Zustand)

Pages:
├── /dashboard     → SupervisorDashboard
│   ├── left panel:  AlertFeed (WebSocket, live)
│   └── right panel: SiteHeatmap (Leaflet)
│
├── /analytics     → Analytics
│   ├── TrendChart (7/30 days)
│   ├── ZoneBreakdown bar chart
│   └── TimeOfDayHeatmap
│
├── /admin         → AdminPanel
│   ├── Tab 1: Document Upload + Ingestion Status
│   ├── Tab 2: Image Upload + Labeling + Training
│   └── Tab 3: Model Version History
│
└── /login         → Login

Data Flows:
├── WebSocket: ws://backend/ws/alerts → AlertFeed (real-time push)
├── REST: GET /dashboard/events?site_id=X → heatmap pins
├── REST: GET /dashboard/analytics → charts data
├── REST: POST /admin/train → trigger training job
└── REST: GET /models/latest → training status polling
```

---

## 3. Data Flow Diagrams

### 3.1 Near-Miss Event Flow (Primary Feature)

```
[Worker on site]
    |
    | (event occurs — e.g., object falls near worker)
    ▼
[Glasses Microphone] → [30s Audio Buffer] → [YAMNet @ 1s windows]
[Glasses IMU]        → [Angular velocity]  → [Jerk Detector]
    |                                              |
    | acoustic_score > θ₁                         | jerk > θ₂
    └──────────────────┬────────────────────────── ┘
                       │ BOTH within ±2s
                       ▼
                  [FUSION TRIGGER]
                       │
         ┌─────────────┼──────────────┐
         │             │              │
         ▼             ▼              ▼
   [Freeze audio] [Capture frame] [Read GPS]
         │             │              │
         └─────────────┼──────────────┘
                       │
                       ▼
              [Build NearMissPayload]
                       │
                  [SQLite Queue]
                       │ (when network available)
                       ▼
              [POST /api/v1/events]
                       │
               [Postgres — store]
                       │
              [Celery — AnalysisTask]
              ┌────────┴──────────┐
              │                   │
              ▼                   ▼
     [YAMNet recheck]    [GPT-4o Vision]
              │                   │
              └────────┬──────────┘
                       │
              [Update event record]
                       │
         [WebSocket broadcast to dashboard]
                       │
              [Supervisor sees alert card]
                       │
              [Reviews in < 10 seconds]
                       │
            ┌──────────┴──────────┐
            │                     │
         [CONFIRM]             [DISMISS]
            │                     │
   [Add to training_samples]  [Mark as FP]
   [Update heatmap]           [Optionally tune θ]
   [Update zone risk]
```

### 3.2 Self-Learning Flow (The Core Innovation)

```
[Supervisor confirms near-miss]
    │
    ▼
[training_samples record created]
{event_id, audio_s3_url, frame_s3_url, label, osha_category}
    │
    │ (accumulates over days/weeks)
    ▼
[APScheduler check every 6 hours]
    │
    ├─ new_samples >= 20?  ──NO──► wait
    ├─ days_since_train >= 7? ─NO─► wait
    └─ gpu_util < 0.5?   ──NO──► wait
    │
    ALL YES
    │
    ▼
[TrainingJob dispatched via Celery]
    │
    ▼
[ReplayBuffer.sample()]
    │
    ├── 70% → historical confirmed samples (all sites, anonymized)
    └── 30% → new unprocessed samples (this trigger's new_samples)
    │
    ▼
[AcousticTrainer]
│  Load YAMNet backbone (weights FROZEN)
│  Load existing classification head
│  Fine-tune head: 30 epochs, AdamW, CosineAnnealingLR
│  80/20 train/val split on sampled data
    │
    ▼
[Evaluate on validation set]
    │
    ├─ accuracy >= previous? ──YES──► promote
    │                                     │
    │                                     ▼
    │                          [Quantize to INT8 TFLite]
    │                          [Upload to S3]
    │                          [Update ModelVersionStore]
    │                          [Mark samples as is_used=true]
    │                          [Devices download at next check]
    │
    └─ accuracy < previous? ──YES──► REJECT
                                         │
                                         ▼
                                [Keep previous model]
                                [Notify admin dashboard]
                                [Log rejection reason]
    │
    ▼
[ModelUpdateWorker on phone]
    │ (triggers on: charging + WiFi)
    ▼
[GET /api/v1/models/latest]
    │
    ├─ newer version? ──YES──► download to temp
    │                              │
    │                              ▼
    │                     [ModelValidator.smokeTest()]
    │                     (run 5 synthetic audio clips,
    │                      verify expected classes detected)
    │                              │
    │                    passes ───► stage for next restart
    │                    fails  ───► discard, keep current
    │
    └─ same version? ──YES──► do nothing
```

### 3.3 Voice Copilot Flow (Secondary Feature)

```
[Worker says "Hey Sentinel, what is the rebar spacing on grid D?"]
    │
    ▼
[WakeWordDetector fires] (~5ms, always on)
    │
    ▼
[WhisperSTT.transcribe()] (~150ms, on-device)
→ "what is the rebar spacing on grid D"
    │
    ▼
[IntentRouter.classify()] (~30ms, on-device DistilBERT)
→ doc_type: "STRUCTURAL"
    │
    ▼
[POST /api/v1/voice/query {text, doc_type, site_id}]
    │
    ▼
[embed query → vector] (~50ms)
    │
    ▼
[Qdrant hybrid search: filter={site_id, doc_type=STRUCTURAL}]
→ top-5 relevant chunks (~80ms)
    │
    ▼
[Cohere Rerank: reorder chunks] (~100ms)
    │
    ▼
[LLM chain (Claude claude-sonnet-4-6)]
system: "Answer only from the provided context.
         Always cite the source document and section.
         If context is insufficient, say so explicitly.
         Be concise — answer will be spoken aloud."
→ "According to Section 3.4 of the Structural Drawings,
   rebar spacing on grid D columns is 150mm center to center
   for vertical bars and 200mm for horizontal ties."
    │
    ▼
[TTSPlayer.speak(answer)] (~100ms)
    │
    ▼
[Worker hears answer in ~2–2.5 seconds]
    (hands never left the rebar)
```

### 3.4 Admin Training Flow

```
[Admin uploads 20 images of "Confined Space Entrance"]
    │
    ▼
[S3 storage: site-isolated bucket]
    │
    ▼
[Admin labels all images: "confined_space_entrance"]
    │
    ▼
[Admin clicks "Train Model"]
    │
    ▼
[Celery: AdminTrainingJob]
│
│  Load MobileNet-v3-Small (backbone FROZEN, ImageNet weights)
│  Replace classification head:
│    old_classes + ["confined_space_entrance"]
│  Apply augmentations:
│    HorizontalFlip, Brightness±20%, Rotation±10°
│    (critical for <20 images per class)
│  Train head only: 20 epochs, lr=1e-3
│  Evaluate: held-out 20% per class
│  If val_accuracy (new class) > 70% AND
│     val_accuracy (existing classes) within 5% of previous:
│       → promote
│     else:
│       → reject, notify admin: "Need more images for X class"
    │
    ▼
[Admin sees accuracy report in dashboard]
    │
    ├─ Accept → model pushed to all devices on site
    └─ Reject → back to upload more images
```

---

## 4. User Workflows (Complete)

### 4.1 Worker — Start of Shift

```
1. Worker puts on Meta glasses and starts the SentinelSite app
2. App connects to glasses via Bluetooth
3. CalibrationActivity prompts: "Record 60 seconds of ambient site sound"
4. Worker walks around their work zone normally for 60 seconds
5. AcousticBaselineCalibrator computes PSD baseline, sets θ₁
6. Status overlay shows: "Sentinel Active — Monitoring"
7. Worker forgets the app exists and does their job
```

### 4.2 Worker — Near-Miss Event

```
1. Worker is installing formwork at height. A metal rod falls nearby.
2. Loud impact + worker flinches (head snap)
3. YAMNet detects "Impact, heavy objects" (score: 0.87) > θ₁ → AudioAnomalyEvent
4. IMU detects jerk magnitude 4.2 rad/s² > θ₂ → IMUJerkEvent
5. FusionGate: Δt = 0.6s < 2s → NEAR_MISS_TRIGGER
6. System: freezes 30s audio, captures frame, reads GPS
7. Payload built and queued in < 500ms
8. Worker does not hear or feel anything. Work continues uninterrupted.
9. Payload uploads when next LTE window available (< 5 minutes)
10. Supervisor sees alert card on dashboard in < 5s from upload
```

### 4.3 Worker — Voice Query

```
1. Worker (hands on rebar) needs to confirm beam dimensions
2. Says: "Hey Sentinel, what is the beam depth for the B-2 frame?"
3. App activates voice mode (no button press needed)
4. Whisper transcribes in ~150ms
5. IntentRouter → "STRUCTURAL"
6. Query sent to backend
7. RAG retrieves Section 6.1 from Structural Drawings
8. Claude answers: "Per Section 6.1, the B-2 frame beam depth is 600mm"
9. Worker hears answer in ~2.5 seconds via glasses speaker
10. Worker confirms the dimension without putting down tools
```

### 4.4 Supervisor — Alert Review

```
1. Alert card appears on dashboard (real-time WebSocket push)
2. Card shows:
   - Timestamp: 09:34:21 UTC
   - Worker: Worker #14 (Zone B - North Scaffolding)
   - YAMNet: "Impact, heavy objects" — 87% confidence
   - Frame: [480p photo of scaffolding area]
   - GPT-4o: "Worker at elevation ~4m. Metal rod visible falling in frame."
3. Supervisor clicks Play → hears 30s audio clip
   (hears ambient sound, metal impact, brief exclamation)
4. Supervisor clicks Confirm
5. OSHA category selector: chooses "Struck-by"
6. Severity auto-calculated: High
7. Near-miss record confirmed in < 10 seconds
8. Heatmap updates with new pin at GPS location
9. Zone B risk score increases → turns yellow in heatmap
10. Record flagged for OSHA report and added to training queue
```

### 4.5 Admin — Model Training Session

```
1. Admin logs into dashboard → Admin Panel → Train Model tab
2. Clicks "Upload Images" → drag-drops 25 photos of
   "Unguarded Excavation Edge" (specific to this project's terrain)
3. Labels all 25 images: "unguarded_excavation"
4. Dashboard warns: "Minimum 15 images: ✓ You have 25"
5. Admin clicks "Start Training"
6. Training status: Queued → Downloading base model → Training...
   (estimated 4 minutes for 25-image dataset)
7. Accuracy report:
   - "unguarded_excavation": 84% validation accuracy
   - All existing classes: within 3% of previous accuracy ✓
8. Admin clicks Accept
9. Model pushed OTA to all 12 devices on this site
10. Devices download silently. Next shift start loads new model.
11. Workers can now see "unguarded_excavation" zones detected
    in visual context during near-miss events
```

### 4.6 Self-Learning — Week 3 Training Cycle

```
Conditions by Day 21 of deployment:
  - 34 new confirmed near-miss samples in training_samples table
  - Last training was 8 days ago
  - Server GPU at 23% utilization
  → APScheduler triggers TrainingJob

Training executes:
  - Samples: 34 new + 79 historical (70/30 replay split)
  - Fine-tunes acoustic head: 30 epochs
  - Validation accuracy: 84.3% (previous: 81.2%)
  → Promoted

Result:
  - Detection threshold auto-adjusted for this site's noise profile
  - "Concrete mixer shutdown spike" no longer triggers false positives
    (model learned this site's specific acoustic signature)
  - False positive rate dropped from 3.2/hour to 1.1/hour
  - Workers notice they're no longer seeing spurious events
    (which they wouldn't notice anyway since they're passive)
```

---

## 5. Infrastructure Deployment

```
docker-compose.yml services:

sentinelsite-api      → FastAPI (uvicorn, 4 workers)
sentinelsite-worker   → Celery worker (GPU access for training)
sentinelsite-beat     → Celery beat (APScheduler for training triggers)
postgres              → Postgres 16 (events, models, training data)
qdrant                → Qdrant 1.9 (vector store for RAG)
redis                 → Redis 7 (Celery broker + WebSocket message bus)
nginx                 → Reverse proxy + SSL termination

Cloud resources:
S3 (or MinIO self-hosted) → audio clips, frames, model files, admin images

Minimum server spec for training:
  CPU: 8 vCPU (for Celery workers + API)
  RAM: 32GB
  GPU: 1× T4 or A10G (for training jobs — not needed for inference)
  Storage: 500GB SSD (audio clips grow over time)
```

---

## 6. Security Architecture

```
Authentication:    JWT tokens, RS256, 24h expiry
Authorization:     RBAC — roles: worker / supervisor / admin / system
Data at rest:      S3 AES-256 encryption, Postgres encrypted volume
Data in transit:   TLS 1.3 for all API calls and WebSocket connections
Audio data:        Stored only on fusion trigger — never continuous upload
Worker IDs:        SHA-256 hashed in aggregate reports (anonymizable)
Site isolation:    Every Qdrant collection and S3 prefix namespaced by site_id
Model delivery:    Signed S3 presigned URLs (1-hour expiry) for OTA downloads
```