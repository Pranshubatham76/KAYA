# SentinelSite — Product Requirements Document (PRD)
**Version:** 1.0  
**Status:** Pre-development  
**Scope:** Hackathon MVP + V2 roadmap  
**Last Updated:** June 2026

---

## 1. Executive Summary

SentinelSite is a first-person passive intelligence system for construction sites, deployed via Meta smart glasses. It detects near-miss events automatically through multimodal sensing (acoustic + IMU), captures and logs them with contextual metadata, and surfaces structured reports to supervisors — without requiring any action from the worker. A secondary voice copilot provides on-demand access to site documents. A self-learning ML pipeline improves detection accuracy per-site over time from confirmed events and admin-curated training data.

**Problem it solves:** 49% of construction injuries go unreported, and near-misses are almost never captured. Every existing system requires the worker to voluntarily act. SentinelSite makes the worker's glasses do the witnessing.

---

## 2. User Roles

| Role | Who They Are | Primary Interface | Key Goal |
|---|---|---|---|
| **Worker** | Ironworker, concrete crew, electrician, plumber — anyone doing physical site work | Android app on phone + glasses | Zero interruption to their work; voice Q&A when needed |
| **Supervisor** | Foreman or safety officer on site | Web dashboard (mobile-first) | Review near-miss alerts in real-time, confirm/dismiss, track zone risk |
| **Admin / Site Manager** | Project manager, safety director | Web dashboard (admin panel) | Upload training images, manage site documents, review model accuracy |
| **System (ML Pipeline)** | Automated background process | Backend + cloud GPU | Collect confirmed events, retrain models on schedule, push updates |

---

## 3. Features by Role

### 3.1 Worker Features

#### PASSIVE SENSING (zero worker action required)

| ID | Feature | Description | Priority |
|---|---|---|---|
| W-P01 | Acoustic near-miss detection | YAMNet continuously classifies ambient audio every 1 second. Anomalous events (crash, shout, impact) above site-calibrated threshold trigger event candidate | MVP |
| W-P02 | IMU startle detection | Glasses IMU monitors head angular velocity. Sudden jerk (flinch/startle) above threshold contributes to fusion trigger | MVP |
| W-P03 | Multimodal fusion gate | Near-miss candidate logged only when BOTH acoustic anomaly AND IMU startle occur within ±2 second window. Prevents single-signal false positives | MVP |
| W-P04 | 30-second pre-event audio buffer | Circular buffer always records. Frozen on trigger. Preserves 30s of context before the event — not just the moment of impact | MVP |
| W-P05 | Event-triggered frame capture | Single photo taken at moment of fusion trigger. Not continuous streaming. Captures worker's visual field at event time | MVP |
| W-P06 | Offline queue | Near-miss payloads stored in local SQLite when no connectivity. Auto-sync when LTE/WiFi available. No data loss on poor-connectivity sites | MVP |
| W-P07 | Battery-conscious operation | YAMNet runs on CPU with 1s inference window. No continuous video streaming. Background sync only on charging. Inference loop consumes ~80–100mA extra on modern phone | MVP |

#### ACTIVE VOICE COPILOT (worker-initiated, secondary feature)

| ID | Feature | Description | Priority |
|---|---|---|---|
| W-V01 | Wake-word activation | "Hey Sentinel" activates voice mode. No manual button press needed. Uses Porcupine on-device wake-word detection | MVP |
| W-V02 | Speech-to-text | On-device Whisper Small TFLite. ~150ms latency. Works without internet | MVP |
| W-V03 | Intent routing to correct document | Query classified into document type (structural, safety, schedule, material, electrical, etc.) before retrieval. Fetches from relevant documents only, not all site docs | MVP |
| W-V04 | RAG-grounded answer | LLM answers only from retrieved site document chunks. Always includes source citation ("Per Section 4.2 of Structural Drawings..."). Refuses to answer if no relevant context found | MVP |
| W-V05 | Text-to-speech response | Answer played via glasses speaker. Edge TTS / Android TTS. ~100ms latency | MVP |
| W-V06 | Multilingual support | Whisper detects language automatically. LLM responds in detected language. Covers Hindi, Spanish, Tamil, Mandarin | V2 |
| W-V07 | Conversation context (2-turn) | Remembers previous question for follow-up queries within same work session | V2 |

#### SITE CALIBRATION

| ID | Feature | Description | Priority |
|---|---|---|---|
| W-C01 | Ambient baseline calibration | 60-second ambient audio recording at shift start. Computes mean power spectral density per frequency band. Sets site-specific detection threshold θ₁ automatically | MVP |
| W-C02 | IMU motion baseline | 2-minute normal motion recording. Computes worker's typical motion range. Sets site-specific IMU threshold θ₂ | V2 |

---

### 3.2 Supervisor Features

#### REAL-TIME ALERT MANAGEMENT

| ID | Feature | Description | Priority |
|---|---|---|---|
| S-A01 | Live near-miss alert feed | WebSocket-powered real-time card feed. Each card: timestamp, zone, worker ID, YAMNet class + confidence, audio player, event frame, GPT-4o scene description | MVP |
| S-A02 | Audio clip playback | 30-second pre-event audio clip playable inline on alert card. Allows supervisor to hear context of event | MVP |
| S-A03 | Frame viewer | Event frame displayed on card. Low-res (480p) — labelled as "forensic context, not conclusive evidence" | MVP |
| S-A04 | GPT-4o scene description | Natural language description of frame content (e.g., "Worker appears to be on scaffolding at ~4m elevation. Debris visible in upper right of frame") | MVP |
| S-A05 | One-click confirm / dismiss | Confirm: supervisor marks event as real near-miss. Dismiss: false positive. Time to review target < 10 seconds per event | MVP |
| S-A06 | OSHA category classification | On confirm, supervisor selects: Fall / Struck-by / Caught-in / Electrocution / Other. Used for report generation and model training label | MVP |
| S-A07 | Alert severity badge | Auto-assigned severity (Low / Medium / High) based on YAMNet confidence + OSHA category historical risk weight | MVP |

#### SITE INTELLIGENCE

| ID | Feature | Description | Priority |
|---|---|---|---|
| S-I01 | Site heatmap | GPS-plotted confirmed near-miss pins on site plan overlay. Color gradient: green → yellow → red by event density | MVP |
| S-I02 | Zone risk indicator | Per-zone near-miss rate vs site average. Flags zones at 2× or higher site average rate | MVP |
| S-I03 | Event detail drill-down | Click any pin → event detail modal: full audio, full frame, description, worker, timestamp, review history | MVP |
| S-I04 | Trend chart | Near-miss frequency over last 7/30 days. Line chart. Highlights anomalous spikes | V2 |
| S-I05 | Time-of-day heatmap | Hour × day matrix of near-miss density. Identifies high-risk time windows | V2 |
| S-I06 | Worker risk ranking | Anonymized per-worker near-miss frequency. Flag for safety coaching. Opt-in, privacy-compliant | V2 |

#### REPORTING

| ID | Feature | Description | Priority |
|---|---|---|---|
| S-R01 | OSHA-format PDF export | Generate OSHA 300 Log compatible near-miss report from confirmed events. Includes all metadata fields | MVP |
| S-R02 | Date-range export | Export all confirmed events in a date range to CSV. For insurance and audit purposes | V2 |
| S-R03 | Weekly summary digest | Auto-generated weekly email: top risk zones, event count, model accuracy, recommended actions | V2 |

---

### 3.3 Admin / Site Manager Features

#### TRAINING DATA MANAGEMENT

| ID | Feature | Description | Priority |
|---|---|---|---|
| A-T01 | Image upload interface | Drag-drop image uploader. Supports JPG/PNG/HEIC. Batch upload. Preview grid. No size limit (server resizes) | V2 (demo in hackathon) |
| A-T02 | Image labeling UI | Per-image class label input. Dropdown with standard construction categories + custom text entry. Warning if < 15 images per class | V2 |
| A-T03 | Training job trigger | "Train Model" button. Shows progress indicator: Queued → Training → Evaluating → Deployed / Failed | V2 |
| A-T04 | Model accuracy report | Post-training: per-class validation accuracy. Comparison with previous model. Accept / Reject before deployment | V2 |
| A-T05 | Model version history | Table of all model versions: date, accuracy, training sample count, status (active/retired/failed) | V2 |

#### DOCUMENT MANAGEMENT

| ID | Feature | Description | Priority |
|---|---|---|---|
| A-D01 | Site document upload | Upload PDFs: structural drawings, safety plans, SOPs, material data sheets, RFI logs | MVP |
| A-D02 | Document type tagging | Assign document type on upload (Structural / Safety / Schedule / Material / Electrical / Other). Used by intent router | MVP |
| A-D03 | Ingestion status | Show ingestion status per document: Queued / Processing / Indexed / Failed with error message | MVP |
| A-D04 | Document replacement | Replace existing document (e.g., updated drawings). System re-indexes and removes old chunks | V2 |
| A-D05 | Retrieval testing UI | Input a test question, see which document chunks were retrieved. For QA of RAG accuracy | V2 |

#### SITE CONFIGURATION

| ID | Feature | Description | Priority |
|---|---|---|---|
| A-C01 | Site plan upload | Upload site plan image (JPG/PNG). Defines GPS boundary polygon for heatmap overlay | MVP |
| A-C02 | Zone definition | Draw zones on site plan (polygon tool). Name each zone. Sets basis for zone risk reporting | V2 |
| A-C03 | Worker device management | Assign device IDs to workers. Set worker roles. Enable/disable devices | MVP |
| A-C04 | Detection sensitivity config | Manual override of θ₁ and θ₂ thresholds if auto-calibration is insufficient | V2 |

---

## 4. Functional Requirements (FR)

### Detection Pipeline

| ID | Requirement |
|---|---|
| FR-D01 | System SHALL classify audio in 1-second windows continuously using YAMNet TFLite on the paired phone |
| FR-D02 | System SHALL maintain a 30-second circular audio buffer in memory at all times during active sensing |
| FR-D03 | System SHALL detect IMU angular velocity from the glasses or phone IMU at minimum 50Hz sampling rate |
| FR-D04 | System SHALL compute jerk (dω/dt) and compare against a configurable threshold θ₂ |
| FR-D05 | System SHALL trigger a near-miss event ONLY when acoustic anomaly score > θ₁ AND IMU jerk > θ₂ within ±2 seconds |
| FR-D06 | System SHALL capture a single frame from the glasses camera within 500ms of fusion trigger |
| FR-D07 | System SHALL package event payload with: timestamp (UTC ms), GPS lat/lon, 30s audio clip, event frame, YAMNet class, YAMNet confidence score, worker ID, device ID |
| FR-D08 | System SHALL store payloads in SQLite queue if network unavailable and retry upload with exponential backoff |
| FR-D09 | Acoustic baseline calibration SHALL record minimum 60 seconds of ambient audio and compute per-band PSD baseline |

### Voice Copilot

| ID | Requirement |
|---|---|
| FR-V01 | System SHALL respond to voice queries within 3 seconds under normal LTE conditions |
| FR-V02 | System SHALL classify query intent into minimum 8 document type categories before retrieval |
| FR-V03 | Every LLM answer SHALL include a source citation referencing the specific document and section retrieved |
| FR-V04 | System SHALL return "I could not find information about this in the site documents" rather than hallucinate when no relevant chunks are retrieved |
| FR-V05 | STT SHALL operate on-device without cloud dependency |

### Self-Learning Pipeline

| ID | Requirement |
|---|---|
| FR-L01 | System SHALL add confirmed near-miss events to training queue when supervisor clicks Confirm |
| FR-L02 | Training job SHALL be triggered ONLY when: (a) ≥ 20 new confirmed samples in queue, (b) ≥ 7 days since last training, (c) server GPU utilization < 50% |
| FR-L03 | Every training batch SHALL use experience replay: minimum 70% historical samples + maximum 30% new samples |
| FR-L04 | System SHALL evaluate new model on held-out validation set before promotion. Promotion requires accuracy ≥ previous model accuracy |
| FR-L05 | System SHALL rollback to previous model version if post-deployment accuracy drops more than 5% on validation set |
| FR-L06 | Model updates SHALL be delivered to devices as background downloads without interrupting the running inference loop |
| FR-L07 | New model SHALL be applied only on next application cold start, never via hot-swap during active sensing |

### Admin Training

| ID | Requirement |
|---|---|
| FR-A01 | Admin image upload SHALL accept minimum 15 images per class before enabling training |
| FR-A02 | Training SHALL freeze backbone weights and fine-tune only the classification head |
| FR-A03 | Admin SHALL be able to accept or reject a new model before it is deployed to worker devices |
| FR-A04 | Data augmentation (horizontal flip, ±20% brightness, ±10° rotation) SHALL be applied to all training images |

---

## 5. Non-Functional Requirements (NFR)

### Performance

| ID | Requirement | Target |
|---|---|---|
| NFR-P01 | YAMNet inference latency on phone CPU | < 100ms per 1-second window |
| NFR-P02 | End-to-end event detection to supervisor alert | < 5 seconds under LTE conditions |
| NFR-P03 | Voice query response (STT + RAG + TTS) | < 3 seconds under LTE conditions |
| NFR-P04 | Frame capture from trigger to receipt | < 500ms |
| NFR-P05 | Dashboard WebSocket alert delivery after server receipt | < 200ms |
| NFR-P06 | Inference loop additional battery draw | < 120mA on mid-range Android phone |

### Reliability

| ID | Requirement | Target |
|---|---|---|
| NFR-R01 | Event payload delivery guarantee | Zero data loss — SQLite queue + retry until delivered |
| NFR-R02 | Offline operation capability | Full passive sensing functional with no network |
| NFR-R03 | Fusion gate false positive rate | < 2 events per hour during normal construction activity (tunable via θ) |
| NFR-R04 | Model rollback time on accuracy degradation | < 10 minutes automated rollback |

### Privacy & Security

| ID | Requirement |
|---|---|
| NFR-S01 | Worker voice queries SHALL NOT be logged beyond session unless worker explicitly opts in |
| NFR-S02 | Audio clips SHALL be stored encrypted at rest in S3 with AES-256 |
| NFR-S03 | Worker IDs in near-miss records SHALL be anonymizable for aggregate reporting |
| NFR-S04 | Admin training images SHALL be stored in site-isolated S3 buckets with no cross-site access |
| NFR-S05 | All API endpoints SHALL require JWT authentication with role-based access control |
| NFR-S06 | Audio is NOT continuously uploaded — only the 30s clip captured on trigger event |

### Scalability

| ID | Requirement |
|---|---|
| NFR-SC01 | Backend SHALL support minimum 50 concurrent worker devices per site |
| NFR-SC02 | Qdrant vector store SHALL support minimum 10 site document collections with 100k chunks each |
| NFR-SC03 | Training pipeline SHALL handle concurrent training jobs for multiple sites without interference |

### Accuracy (Targets, Not Guarantees)

| ID | Requirement | Target |
|---|---|---|
| NFR-AC01 | YAMNet acoustic anomaly detection (controlled environment) | > 80% TPR at < 5% FPR |
| NFR-AC02 | Visual PPE classification (on-device MobileNet) | > 75% accuracy on site-specific classes after admin training |
| NFR-AC03 | RAG retrieval hit rate (relevant chunk in top-3) | > 85% on site document Q&A pairs |
| NFR-AC04 | Intent routing accuracy | > 90% on 8 document type categories |

---

## 6. Out of Scope (Explicitly)

| What | Why Excluded |
|---|---|
| Real-time sub-1-second hazard alerts to workers | Hardware latency makes this unsafe — a 2s delayed warning creates false confidence |
| BIM / AR spatial overlay | Requires LIDAR/depth sensors not present in Ray-Ban hardware |
| On-device model training | Computationally infeasible without killing inference loop and battery |
| Continuous video streaming | Bluetooth bandwidth insufficient; defeats purpose of passive detection |
| Biometric sensors (HR, GSR) | Not available on Ray-Ban hardware |
| Predictive accident prevention | Requires longitudinal dataset not yet available; V3+ roadmap |

---

## 7. Definition of Done (Hackathon MVP)

A feature is done when:
1. It works end-to-end in a controlled demo scenario
2. Its failure modes are documented and acknowledged to judges
3. It produces visible output in the supervisor dashboard
4. Latency meets the NFR target in local testing

android studio extension
expo go


