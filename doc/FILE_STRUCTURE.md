# SentinelSite — Complete File & Folder Structure
> Every file listed has a reason. No placeholder directories.

```
sentinelsite/
│
├── android/                          # Android mobile app (Kotlin)
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── java/com/sentinelsite/
│   │   │   │   │
│   │   │   │   ├── audio/
│   │   │   │   │   ├── AudioBufferManager.kt       # 30s circular ring buffer, AudioRecord API
│   │   │   │   │   ├── YAMNetInferenceEngine.kt    # TFLite interpreter, 1s window inference
│   │   │   │   │   ├── AcousticBaselineCalibrator.kt # Site ambient PSD baseline computation
│   │   │   │   │   ├── AnomalyScorer.kt            # Compares YAMNet output vs baseline
│   │   │   │   │   └── AudioConstants.kt           # Sample rate, window size, class indices
│   │   │   │   │
│   │   │   │   ├── vision/
│   │   │   │   │   ├── FrameCaptureManager.kt      # BT camera trigger, frame receipt
│   │   │   │   │   ├── MobileNetInferenceEngine.kt # TFLite visual classifier (on-device)
│   │   │   │   │   └── FramePreprocessor.kt        # Resize to 224x224, normalize
│   │   │   │   │
│   │   │   │   ├── imu/
│   │   │   │   │   ├── IMUManager.kt               # Reads glasses/phone IMU via SDK
│   │   │   │   │   ├── JerkDetector.kt             # dω/dt threshold — startle detection
│   │   │   │   │   └── MotionBaseline.kt           # Per-worker normal motion profile
│   │   │   │   │
│   │   │   │   ├── fusion/
│   │   │   │   │   ├── FusionGate.kt               # AND gate: audio + IMU within ±2s window
│   │   │   │   │   ├── NearMissPayloadBuilder.kt   # Assembles full event JSON
│   │   │   │   │   ├── EventTriggerController.kt   # Orchestrates detection → payload → queue
│   │   │   │   │   └── ThresholdConfig.kt          # θ₁ (audio) and θ₂ (IMU) config store
│   │   │   │   │
│   │   │   │   ├── voice/
│   │   │   │   │   ├── WakeWordDetector.kt         # "Hey Sentinel" — Porcupine or Picovoice
│   │   │   │   │   ├── WhisperSTTEngine.kt         # On-device Whisper Small TFLite
│   │   │   │   │   ├── IntentRouter.kt             # DistilBERT classifier → doc type
│   │   │   │   │   ├── RAGQueryDispatcher.kt       # Sends text query to backend RAG endpoint
│   │   │   │   │   └── TTSPlayer.kt                # EdgeTTS / Android TTS playback
│   │   │   │   │
│   │   │   │   ├── upload/
│   │   │   │   │   ├── UploadQueueManager.kt       # SQLite-backed queue, retry logic
│   │   │   │   │   ├── SyncWorker.kt               # WorkManager worker: upload on LTE/WiFi
│   │   │   │   │   └── ConnectivityMonitor.kt      # Watches for network availability
│   │   │   │   │
│   │   │   │   ├── model/
│   │   │   │   │   ├── ModelUpdateManager.kt       # OTA model download, staged swap
│   │   │   │   │   ├── ModelVersionStore.kt        # Tracks current model version per type
│   │   │   │   │   └── ModelValidator.kt           # Runs smoke test on new model before swap
│   │   │   │   │
│   │   │   │   └── ui/
│   │   │   │       ├── MainActivity.kt             # Entry point, service binder
│   │   │   │       ├── StatusOverlay.kt            # Minimal HUD: recording / event triggered
│   │   │   │       ├── CalibrationActivity.kt      # 60s ambient calibration UI flow
│   │   │   │       └── VoiceCopilotFragment.kt     # Voice Q&A UI panel
│   │   │   │
│   │   │   ├── assets/
│   │   │   │   ├── yamnet.tflite                   # YAMNet INT8 quantized (from TF Hub)
│   │   │   │   ├── mobilenet_v3_head.tflite        # Visual classifier (updated via OTA)
│   │   │   │   ├── whisper_small.tflite            # STT model
│   │   │   │   ├── intent_router.tflite            # DistilBERT intent classifier
│   │   │   │   └── yamnet_class_map.csv            # 521 AudioSet class names
│   │   │   │
│   │   │   └── res/
│   │   │       └── xml/network_security_config.xml # Allow cleartext for local dev only
│   │   │
│   │   ├── build.gradle
│   │   └── AndroidManifest.xml                     # RECORD_AUDIO, CAMERA, ACCESS_FINE_LOCATION
│   │
│   └── gradle/
│       └── libs.versions.toml                      # TFLite, Meta SDK, WorkManager, Ktor
│
├── backend/                          # Python FastAPI server
│   ├── app/
│   │   ├── main.py                               # FastAPI app init, router registration
│   │   ├── config.py                             # Env vars, DB URLs, API keys
│   │   │
│   │   ├── api/
│   │   │   ├── events.py                         # POST /api/v1/events (near-miss ingest)
│   │   │   ├── voice.py                          # POST /api/v1/voice/query (RAG endpoint)
│   │   │   ├── training.py                       # POST /api/v1/admin/train (trigger job)
│   │   │   ├── models.py                         # GET /api/v1/models/latest (OTA check)
│   │   │   ├── dashboard.py                      # GET /api/v1/dashboard/* (analytics)
│   │   │   └── ws.py                             # WS /ws/alerts (supervisor real-time feed)
│   │   │
│   │   ├── core/
│   │   │   ├── auth.py                           # JWT auth, role-based (worker/admin/supervisor)
│   │   │   ├── storage.py                        # S3 upload/download (audio clips, frames)
│   │   │   ├── gps.py                            # Zone mapping: GPS → site zone label
│   │   │   └── osha_report.py                    # Generate OSHA-300 compatible PDF report
│   │   │
│   │   ├── ml/
│   │   │   ├── yamnet_server.py                  # Server-side YAMNet (full quality recheck)
│   │   │   ├── vision_describer.py               # GPT-4o Vision API call for frame description
│   │   │   ├── risk_scorer.py                    # Zone/time risk score computation
│   │   │   └── model_manager.py                  # Track versions, promote/rollback
│   │   │
│   │   ├── rag/
│   │   │   ├── ingestion.py                      # PDF → chunks → embeddings → Qdrant
│   │   │   ├── intent_classifier.py              # Query → document type (10 classes)
│   │   │   ├── retriever.py                      # Semantic search, hybrid search, reranker
│   │   │   ├── llm_chain.py                      # LangChain LCEL chain: context → answer
│   │   │   └── document_types.py                 # Enum: STRUCTURAL, SAFETY, SCHEDULE, etc.
│   │   │
│   │   ├── training/
│   │   │   ├── training_queue.py                 # DB-backed queue of training-eligible events
│   │   │   ├── replay_buffer.py                  # Experience replay: sample historical + new
│   │   │   ├── acoustic_trainer.py               # YAMNet head fine-tuning (PyTorch)
│   │   │   ├── visual_trainer.py                 # MobileNet-v3 head fine-tuning (PyTorch)
│   │   │   ├── admin_trainer.py                  # Admin image upload → fine-tune → push
│   │   │   ├── scheduler.py                      # APScheduler: trigger training when conditions met
│   │   │   ├── quantizer.py                      # PyTorch INT8 → TFLite export pipeline
│   │   │   └── model_pusher.py                   # Push new model to device OTA endpoint
│   │   │
│   │   └── db/
│   │       ├── models.py                         # SQLAlchemy ORM models
│   │       ├── session.py                        # DB connection pool
│   │       └── migrations/                       # Alembic migration files
│   │           └── versions/
│   │
│   ├── tests/
│   │   ├── test_events.py
│   │   ├── test_rag.py
│   │   ├── test_training_pipeline.py
│   │   └── test_replay_buffer.py
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml                        # FastAPI + Postgres + Qdrant + Redis
│
├── dashboard/                        # React web app (supervisor + admin)
│   ├── src/
│   │   ├── components/
│   │   │   ├── alerts/
│   │   │   │   ├── AlertFeed.tsx                 # Real-time WebSocket near-miss card list
│   │   │   │   ├── AlertCard.tsx                 # Audio player + frame + description + actions
│   │   │   │   ├── ReviewModal.tsx               # Confirm/Dismiss + OSHA category selector
│   │   │   │   └── AudioPlayer.tsx               # 30s clip player with waveform
│   │   │   │
│   │   │   ├── heatmap/
│   │   │   │   ├── SiteHeatmap.tsx               # Leaflet map + site plan image overlay
│   │   │   │   ├── ZoneRiskBadge.tsx             # Color-coded risk level per zone
│   │   │   │   └── EventPin.tsx                  # Clickable GPS pin → event detail
│   │   │   │
│   │   │   ├── analytics/
│   │   │   │   ├── TrendChart.tsx                # Near-miss rate over time (Recharts)
│   │   │   │   ├── ZoneBreakdown.tsx             # Events by zone bar chart
│   │   │   │   ├── TimeOfDayHeatmap.tsx          # Hour × day risk matrix
│   │   │   │   └── OshaExportButton.tsx          # Triggers PDF report generation
│   │   │   │
│   │   │   ├── admin/
│   │   │   │   ├── ImageUploader.tsx             # Drag-drop image upload with preview
│   │   │   │   ├── LabelingGrid.tsx              # Image grid + class name input per image
│   │   │   │   ├── TrainingStatus.tsx            # Real-time training job progress
│   │   │   │   ├── ModelVersionTable.tsx         # Current vs previous model accuracy
│   │   │   │   └── DocumentIngestion.tsx         # Upload site PDFs for RAG
│   │   │   │
│   │   │   └── shared/
│   │   │       ├── Navbar.tsx
│   │   │       ├── RoleBadge.tsx
│   │   │       └── ConnectionStatus.tsx          # WS connection indicator
│   │   │
│   │   ├── pages/
│   │   │   ├── SupervisorDashboard.tsx           # Main page: AlertFeed + SiteHeatmap
│   │   │   ├── Analytics.tsx                     # Full analytics view
│   │   │   ├── AdminPanel.tsx                    # Training + document management
│   │   │   └── Login.tsx
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts                   # WS connection management + reconnect
│   │   │   ├── useAlertFeed.ts                   # Alert state management
│   │   │   └── useTrainingStatus.ts              # Polling training job status
│   │   │
│   │   ├── store/
│   │   │   └── alertStore.ts                     # Zustand store for near-miss events
│   │   │
│   │   └── utils/
│   │       ├── api.ts                            # Axios client with auth headers
│   │       └── formatters.ts                     # Date, GPS, confidence formatting
│   │
│   ├── public/
│   │   └── site_plan_placeholder.png             # Replace with actual site plan image
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── ml/                               # ML development workspace (not deployed)
│   ├── acoustic/
│   │   ├── yamnet_baseline_eval.ipynb            # Evaluate YAMNet on construction sounds
│   │   ├── acoustic_fine_tuning.py               # Training script: YAMNet head fine-tune
│   │   ├── noise_robustness_test.py              # Test accuracy at different dB noise levels
│   │   ├── threshold_sweep.py                    # Grid search θ₁ threshold on labeled data
│   │   └── construction_sound_classes.json       # Mapped AudioSet classes for construction
│   │
│   ├── visual/
│   │   ├── mobilenet_fine_tuning.py              # Admin images → head fine-tune script
│   │   ├── augmentation_pipeline.py              # Albumentations augmentation for small datasets
│   │   ├── few_shot_eval.py                      # Accuracy vs. number of training images
│   │   └── ppe_class_definitions.json            # Standard PPE class taxonomy
│   │
│   ├── rag/
│   │   ├── ingestion_pipeline.py                 # PDF → chunks → embeddings → Qdrant
│   │   ├── intent_classifier_train.py            # DistilBERT intent router fine-tuning
│   │   ├── retrieval_eval.py                     # Hit rate, MRR on construction Q&A pairs
│   │   ├── synthetic_qa_generator.py             # GPT-4o generates Q&A from site docs
│   │   └── rag_eval.ipynb                        # End-to-end RAG accuracy evaluation
│   │
│   ├── training/
│   │   ├── replay_buffer_simulation.py           # Simulate replay buffer over N training cycles
│   │   ├── forgetting_benchmark.py               # Measure accuracy drop without replay
│   │   ├── continual_learning_experiment.ipynb   # Full CL experiment with replay vs without
│   │   └── export_to_tflite.py                   # PyTorch → ONNX → TFLite INT8 conversion
│   │
│   └── evaluation/
│       ├── end_to_end_latency_test.py            # Measure full pipeline latency on device
│       ├── false_positive_rate_test.py           # FPR in controlled construction environments
│       └── model_accuracy_tracker.py             # Log accuracy per version per site
│
├── docs/
│   ├── FILE_STRUCTURE.md             # This file
│   ├── PRD.md                        # Product Requirements Document
│   ├── SYSTEM_ARCHITECTURE.md       # Full system + user workflow
│   └── ML_SETUP.md                  # End-to-end ML pipeline guide
│
├── .env.example                      # All required environment variables
├── .gitignore
└── README.md
```

---

## Key Dependency Decisions

### Android
```toml
# gradle/libs.versions.toml
tensorflow-lite = "2.14.0"
tensorflow-lite-task-audio = "0.4.4"
tensorflow-lite-gpu = "2.14.0"       # GPU delegate for NPU acceleration
meta-wearables-sdk = "1.x"          # Meta Wearables DAT
work-manager = "2.9.0"              # Background sync worker
room = "2.6.0"                      # SQLite upload queue
ktor-client = "2.3.0"               # HTTP client for API calls
accompanist = "0.32.0"              # Compose utilities
```

### Backend
```txt
# requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.29.0
sqlalchemy==2.0.29
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.4
celery==5.3.6                   # Async training job queue
boto3==1.34.0                   # S3 storage
torch==2.3.0                    # Training pipeline
torchvision==0.18.0
tensorflow==2.16.1              # TFLite export
onnx==1.16.0                    # PyTorch → ONNX conversion
onnx2tf==1.22.3                 # ONNX → TFLite
langchain==0.2.0
langchain-openai==0.1.6
langchain-qdrant==0.1.1
qdrant-client==1.9.0
unstructured[pdf]==0.13.0       # Layout-aware PDF parsing
openai==1.30.0
apscheduler==3.10.4             # Training scheduler
websockets==12.0
reportlab==4.2.0                # OSHA PDF generation
```

### Dashboard
```json
{
  "dependencies": {
    "react": "^18.3.0",
    "vite": "^5.2.0",
    "zustand": "^4.5.0",
    "recharts": "^2.12.0",
    "leaflet": "^1.9.0",
    "react-leaflet": "^4.2.0",
    "axios": "^1.6.0",
    "tailwindcss": "^3.4.0",
    "react-dropzone": "^14.2.0",
    "wavesurfer.js": "^7.7.0"
  }
}
```