# SentinelSite — Complete Project Brief
### Physical AI Track | Hackathon Research & Architecture Document
**Version:** 3.0 — Final Pre-Code Brief  
**Status:** Pre-prototype | Evidence-driven | No sugar coating

---

## Table of Contents

1. [The Problem — Deep Understanding](#1-the-problem--deep-understanding)
2. [Why Every Obvious Solution Fails](#2-why-every-obvious-solution-fails)
3. [The Insight — What Nobody Is Building](#3-the-insight--what-nobody-is-building)
4. [Your New Ideas — Honest Evaluation](#4-your-new-ideas--honest-evaluation)
5. [The Full ML Architecture](#5-the-full-ml-architecture)
6. [How the Self-Training Works Without Killing Latency](#6-how-the-self-training-works-without-killing-latency)
7. [Admin Training Pipeline](#7-admin-training-pipeline)
8. [Feature List — Complete](#8-feature-list--complete)
9. [System Architecture — End to End](#9-system-architecture--end-to-end)
10. [What Is Real vs What Is a Bet](#10-what-is-real-vs-what-is-a-bet)
11. [Hackathon Demo Plan](#11-hackathon-demo-plan)
12. [Why You Win](#12-why-you-win)

---

## 1. The Problem — Deep Understanding

### Surface-Level Reading (What Everyone Sees)

Construction workers operate in dangerous, information-dense environments without real-time AI support. Documents live in trailers. Hands are full. Workers can't access specs while standing on a beam.

This is true. It is also what **every other hackathon team will say in their first slide.** If your problem statement ends here, you've already lost.

### The Deeper Layer (What the Data Actually Says)

**Layer 1 — The information gap is not the root cause of accidents.**

Research on construction accident causation identifies four cognitive failure types in sequence:
- Failure to *obtain* relevant information
- Failure to *understand* that information
- Choosing an *incorrect response* despite correct understanding
- Failure of *execution* even with the correct intention

A voice chatbot solves Layer 1. It does nothing for Layers 2, 3, and 4 — which together account for the majority of incidents. The worker who understood the spec but misjudged the structural load, or who knew the procedure but skipped a step under time pressure — no RAG system helps them.

**Layer 2 — The near-miss data problem is the actual unsolved problem.**

49% of all workplace injuries go unreported. Near-misses — which are more frequent and more predictive than injuries — are almost *never* captured. The reason is not culture or indifference. It is friction:
- Too many steps to file a report
- Fear of blame and peer judgment
- Disrupts workflow
- No immediate personal benefit to the reporter

Heinrich's Triangle establishes that for every fatal accident, approximately 300 near-misses occurred first. Those 300 events *contain all the information needed to prevent the fatality.* The construction industry has never had a way to capture them systematically because it always required a human to voluntarily act after a stressful event.

**Layer 3 — The cognitive load problem means adding a voice assistant makes things worse, not better.**

Research is explicit: double-tasking — performing a physical task while formulating and verbalizing a cognitive query — measurably increases error rates. A worker on scaffold with tools in both hands, under time pressure, with 90dB ambient noise, cannot safely and reliably operate a voice query interface. The moments when information is most needed are the moments when the cognitive bandwidth to request it is most scarce.

**The synthesis: The real problem is not information access. It is that the construction site generates no first-person, real-time, passive intelligence about what is actually happening at the worker level — and every system that could produce that intelligence requires the worker to do something.**

### What This Means for the Solution

The solution cannot be "give workers a better way to ask for information." It must be: **build a system that generates intelligence from the worker's perceptual field without requiring the worker to do anything at all.**

---

## 2. Why Every Obvious Solution Fails

### Approach A: Real-Time Hazard Detection via Streaming Video

**What it promises:** Glasses stream video → YOLO detects missing PPE, proximity violations → instant voice alert.

**Why it fails:**  
Bluetooth bandwidth caps at 720p/30fps theoretical — expect 480p or lower in metal/concrete RF environments. Round-trip latency from glasses → phone → cloud inference → alert is 800ms–3 seconds on LTE, 3–5 seconds in poor site connectivity. An alert that arrives 3 seconds after a near-miss event is not safety infrastructure. It is a liability.

Additionally: YOLO on degraded 480p video will generate false positives. A worker who gets three false alarms on their first shift removes the glasses and never puts them back on. Alert fatigue is a documented failure mode in industrial safety — not a theoretical concern.

Fixed cameras (Visionify, Detect Technologies) already do this with direct Ethernet connections, better optics, and stable mounting. You cannot beat them at their own game through Bluetooth.

**Verdict: REJECT.**

### Approach B: BIM / AR Overlay on Live View

**What it promises:** Worker sees a digital overlay of building plans, conduit paths, structural elements superimposed on physical reality.

**Why it fails:**  
This requires spatial anchoring — the device must know exactly where it is in 3D space relative to a coordinate system tied to the BIM model. This requires LIDAR, depth sensors, or visual-inertial odometry at centimeter accuracy. Ray-Ban Meta glasses have none of this. The Display model HUD is a small flat overlay — it cannot align a 3D BIM model to a physical wall in real time.

This is a HoloLens problem. HoloLens costs $3,500, weighs 566g, requires dedicated site surveying to set up spatial anchors, and has 2–3 hour battery life. Any team presenting BIM AR on Ray-Ban glasses is either faking it or lying about the hardware.

**Verdict: REJECT.**

### Approach C: Standard Voice Copilot + RAG on Site Documents

**What it promises:** Worker asks a question → LLM retrieves from embedded site documents → voice answer in 2 seconds.

**Why it is not enough:**  
It is buildable. It provides real value. It is also what every ML engineer in the room will build because it is the most obvious application of current LLM tooling to the problem statement.

More critically: it requires the worker to formulate and verbalize a query. See Layer 3 above. The cognitive load argument alone is enough to disqualify this as a primary solution. As a secondary feature — an on-demand assistant when the worker is in a safe moment and has a specific question — it has value. As the core thesis of your product, it loses.

**Verdict: SECONDARY FEATURE ONLY.**

---

## 3. The Insight — What Nobody Is Building

Smart glasses on a construction worker's face are the world's first scalable **first-person near-miss sensor.**

Not a chatbot. Not a document retriever. A sensor. The glasses see what the worker sees, hear what the worker hears, and feel what the worker feels (via IMU). This combination has never been available before at scale on a construction site. Fixed cameras have none of it. Tablets and phones require the worker to hold them. Smartwatches see physiological signals but not the perceptual field.

**The core thesis:**  
> When a near-miss event occurs — something falls, someone shouts a warning, the worker flinches — there is a specific, detectable, multimodal signal signature. Audio anomaly + sudden head motion occurring within a 2-second window is a near-miss candidate. The system captures the 30 seconds of context before the event, packages it with GPS and visual context, and surfaces it to a supervisor for review — without the worker doing anything at all.

This turns the near-miss data problem from a behavioral problem (convincing workers to report) into a technical problem (detecting signals reliably). Behavioral problems are hard. Technical problems are solvable.

The output is not a warning. It is a **structured near-miss record** — timestamped, geotagged, acoustically classified, and visually contextualized — that feeds a site-level risk intelligence layer. After one month of deployment, the system knows which zones, which times of day, and which activity types generate the highest near-miss density. This has never been known before because the data has never existed before.

---

## 4. Your New Ideas — Honest Evaluation

You proposed two additions:
1. Admin / site manager can upload images to train the model
2. The model silently trains itself from field observations without affecting latency

These are genuinely good ideas. They are also more complex than they sound. Here is the honest breakdown.

### Idea 1: Admin-Uploaded Image Training

**What this actually enables:**  
The base model (YAMNet for audio, MobileNet-v3 for vision) is trained on general data — AudioSet, ImageNet, COCO. It knows what a "crash" sounds like in a YouTube video. It does not know what a crash sounds like in *this specific concrete plant in Pune with three industrial mixers running simultaneously.* Admin-uploaded site images allow **domain adaptation**: fine-tuning the visual model on site-specific objects, conditions, and hazards.

**Concrete examples of what admins would upload:**
- Images of this site's specific PPE requirements (some sites require full-face respirators; base model doesn't know this)
- Images of site-specific machinery and equipment for identification queries
- Images of specific hazardous zones (live excavation edges, high-voltage panels labeled with site signage)
- Images of materials specific to this project (unusual composite panels, non-standard formwork systems)
- Images of the site at different times of day / weather conditions so the model handles site-specific lighting

**The ML mechanism:**  
This is transfer learning + few-shot fine-tuning. The backbone (MobileNet-v3 or EfficientNet-B0) is frozen. Only the classification head — the last 1–2 layers — is retrained on the admin-uploaded images. This is safe, fast (minutes on server GPU), and does not cause catastrophic forgetting because the backbone weights are untouched.

**What admin uploads do NOT do:**  
They do not train the acoustic model. Audio domain adaptation is harder and requires labeled audio samples, not images. Keep admin uploads image-only for V1. Do not oversell audio adaptation.

**Honest constraints:**
- Each class (each new type of object/hazard) needs minimum 10–20 images from varied angles, lighting, partial occlusion. Five images of one hard hat is not enough.
- Admin needs a labeling UI — they upload images AND label them ("this is a confined space entrance", "this is an unguarded edge"). Without labels, the images are useless.
- Model retraining happens server-side, not on the phone. The updated model then gets pushed back to the phone as a new TFLite file. This adds an upload-retrain-deploy cycle of 5–30 minutes depending on dataset size.

**Implementation:**
```
Admin uploads images via web dashboard
  → Labeled with class name (text input)
  → Server queues fine-tuning job
  → Load pre-trained MobileNet-v3 backbone (frozen)
  → Retrain classification head on new + existing site classes
  → Quantize to INT8 TFLite
  → Push updated model to all phones on this site (background download)
  → Phone swaps model on next app restart (zero runtime disruption)
```

### Idea 2: Silent Self-Training from Field Observations

**What this actually means and what it does not mean:**

This idea has two very different interpretations, and most people conflate them:

**Interpretation A (What you probably mean):**  
The model learns from confirmed near-miss events — when a supervisor confirms a near-miss candidate, that event's audio + visual data becomes a labeled training sample. Over time, the model gets better at detecting the acoustic and visual signatures of near-misses on *this specific site.*

**Interpretation B (What sounds cooler but is dangerous):**  
The model continuously updates its weights in real-time from every observation it makes, autonomously deciding what to learn from.

**Interpretation A is the right answer.** Here is why Interpretation B fails:

- **Catastrophic forgetting**: Naive sequential fine-tuning destroys previously learned capabilities. A model that fine-tunes on 10 new samples will forget what it learned from the first 10,000 samples. This is a well-documented, hard problem in continual learning. Research shows accuracy on prior tasks can drop from 85% to 40% after training on just 3–4 new subjects.
- **Label quality**: The model cannot label its own outputs reliably enough to train on them. Pseudo-labeling (using the model's own predictions as labels) amplifies errors — if the model is 80% accurate and you train on its predictions, you're training on 20% incorrect labels. Over many cycles this degrades.
- **Latency impact**: Any on-device weight update is computationally expensive. Doing this at inference time will freeze the app for seconds. This cannot happen.

**The correct architecture for self-training:**

```
Phase 1 (Collection — zero latency impact):
  Every confirmed near-miss event → audio clip + frame + label
  stored in a local "training queue" on phone
  (just JSON + file references, no computation)

Phase 2 (Training — scheduled, not real-time):
  Triggered ONLY when:
    - Phone is charging
    - App is in background
    - WiFi is available
    - Queue has >= N new samples (e.g., N=20)
  Training runs server-side (not on phone) using queued samples
  Uses replay buffer to prevent catastrophic forgetting
    → keeps 20% of old training data mixed with new data
    → only fine-tunes acoustic classification head (not backbone)

Phase 3 (Deployment):
  New model pushed to phone silently
  Phone loads new model at next cold start
  Worker never experiences any interruption
```

**What "self-training" actually improves over time:**
- Acoustic anomaly thresholds adapt to this site's specific noise profile
- New near-miss sound signatures specific to this site's equipment get added
- False positive rate decreases as the model learns what is *not* a near-miss on this site
- Visual recognition improves as more confirmed frames are added to the training set

**What it does NOT do:**
- It does not make the model generally smarter
- It does not learn from unlabeled observations (it needs the supervisor's confirm/dismiss as the label)
- It does not update in real-time
- It does not retrain on-device (too expensive; server-side only)

**The key dependency: supervisor labels are the supervision signal.** Every time a supervisor clicks Confirm or Dismiss on a near-miss candidate, they are generating a labeled training sample. This is the loop. Without supervisor engagement, self-training cannot happen. Design the dashboard to make confirm/dismiss so fast (< 5 seconds) that supervisors actually do it.

---

## 5. The Full ML Architecture

The system has three distinct ML layers, each with a different purpose, training regime, and deployment pattern.

### Layer 1 — Acoustic Event Detector (Always Running, On-Device)

**Model:** YAMNet (TFLite INT8 quantized)  
**Backbone:** MobileNet-v1 depthwise-separable convolutions  
**Input:** 0.96-second audio window, 16kHz mono, mel spectrogram  
**Output:** 521-class probability distribution (AudioSet ontology)  
**Relevant classes:** Impact (heavy objects), Crash, Shout, Alarm, Bang, Thud, Breaking  
**Inference time:** ~15ms per window on phone CPU; ~5ms on NPU  
**Update regime:** Base model frozen. Only deployment is initial install. Site-specific adaptation (Phase 2) fine-tunes only the classification head via server-side training.

**What it detects vs what it cannot:**  
Detects: high-energy impulsive events that significantly exceed the acoustic baseline  
Cannot reliably detect: low-energy near-misses (near-trip without sound), events drowned by identical machinery noise, near-misses that produce no acoustic signature

**Baseline calibration:**  
On first deployment per site, record 60 seconds of ambient audio. Compute mean power spectral density per frequency band. This becomes the background model. Detection threshold θ₁ is set at N standard deviations above baseline (N tunable per site noise level, default N=2.5).

### Layer 2 — Visual Context Classifier (Event-Triggered, On-Device)

**Model:** MobileNet-v3-Small (TFLite INT8) with custom classification head  
**Input:** Single 480p frame captured at event trigger  
**Output:** Multi-label classification: PPE compliance (hard hat, vest, gloves), zone type (height work, confined space, excavation), hazard proximity  
**Inference time:** ~30–50ms on phone NPU  
**Update regime:** Admin uploads → server fine-tunes head → new TFLite pushed to device

**Why MobileNet-v3-Small specifically:**  
Research on construction edge deployment confirms MobileNet-class models achieve the right accuracy/speed tradeoff for on-device inference. MobileNet-v3-Small is ~2.5MB quantized, 60ms inference on a mid-range phone. EfficientNet-B0 would be more accurate but ~5× slower. At event-trigger frequency (not continuous), the accuracy tradeoff favors MobileNet-v3.

**What it does NOT do:**  
It does not run continuously. It runs once per near-miss trigger event. It is not a real-time PPE detector. Its output is forensic context ("what was in the worker's field of view at the moment of the event"), not a real-time safety alert.

### Layer 3 — Semantic Vision Describer (Cloud, Post-Event)

**Model:** GPT-4o Vision API (or Claude claude-sonnet-4-6 multimodal)  
**Input:** Event frame + structured near-miss metadata  
**Output:** Natural language scene description for supervisor review  
**Latency:** 1.5–3 seconds after event payload reaches server  
**Cost:** ~$0.003–0.01 per event (acceptable at near-miss event frequency)

**Example output:**  
*"Worker appears to be at elevation on scaffolding, approximately 4–5 meters above ground. The frame shows what appears to be falling debris visible in upper right of frame. Worker's hard hat is visible and correctly worn. No immediate safety barrier visible between worker and fall edge."*

This layer does not run on-device. It runs server-side after the event is logged. Its output is attached to the near-miss record before the supervisor review card is displayed. It transforms a raw frame into actionable language that a supervisor can act on without squinting at a low-res image.

### Layer 4 — Manual Router + RAG (On-Demand, Voice-Triggered)

**What it is:**  
This is the secondary feature — the voice copilot from the original brief. When a worker is in a safe moment and has a specific question, they can ask verbally. The system does STT → intent classification → RAG retrieval from site documents → LLM response → TTS.

**Why it is secondary and not primary:**  
It requires worker-initiated action. The passive sensing layers require nothing. Demote this in the pitch. Keep it in the product because it provides immediate, demonstrable value in a hackathon context.

**The "choose the right manual" intelligence:**  
This is what you asked about — instead of searching all documents, the system routes the query to the correct manual or spec based on query intent. This is intent classification before retrieval:

```
Query: "What is the rebar spacing on grid D?"
  → Intent classifier → "structural" → search only structural drawings
  → Higher precision retrieval, lower hallucination risk

Query: "Is it safe to use this chemical near the HVAC intake?"
  → Intent classifier → "safety / SDS" → search only safety data sheets
  → Returns specific chemical handling guidance

Query: "What trades are working in Zone B today?"
  → Intent classifier → "schedule" → search daily work plan documents
  → Returns trade schedule, not structural specs
```

The router is a small fine-tuned classifier (DistilBERT or even a simple embedding cosine-similarity classifier) trained on labeled query-to-document-type mappings. This is not complex ML — it is classification with ~10 classes (structural, safety, schedule, material, electrical, plumbing, inspection, etc.). Train it on 200–300 synthetic examples and it will perform well enough for a demo.

---

## 6. How the Self-Training Works Without Killing Latency

This is the most technically precise answer to your concern. Three principles govern this.

### Principle 1: Training and Inference Are Completely Separated

The phone does inference. The server does training. These never happen simultaneously on the same hardware. The phone's inference loop is never interrupted because training never runs on the phone.

```
Phone (always):          [Inference Loop] → [Event Buffer] → [Upload Queue]
Server (scheduled):      [Training Job] → [Model Export] → [Model Push]
Phone (background):      [Silent Download] → [Staged for next restart]
```

### Principle 2: The Training Queue Is Just File References

When a near-miss event is confirmed by a supervisor, the server marks that event record with `is_training_sample = true`. The phone does not know this has happened. The phone does not do any additional work. The training sample collection is entirely server-side.

There is no on-device computation for training data preparation. The event payload was already uploaded at event time. The supervisor confirmation just flips a database flag.

### Principle 3: Model Updates Are Applied at Cold Start, Not Hot-Swap

When a new model is ready (after a server-side training job), it is pushed to the phone as a background file download (like an app update). The phone downloads it silently. The running inference loop continues using the *old* model uninterrupted. The new model is applied the next time the app is fully restarted — which happens naturally at the start of each shift.

No hot-swap. No model reload mid-inference. No disruption to the running detection pipeline.

### What Triggers a Training Job

```python
trigger_conditions = {
    "new_confirmed_samples": >= 20,       # enough new data to be worth training
    "time_since_last_training": >= 7,     # days (don't retrain more than weekly)
    "server_GPU_utilization": < 0.5,      # don't compete with other workloads
}

# Additional safeguard: if accuracy on validation set drops after training,
# automatically rollback to the previous model version.
```

### Catastrophic Forgetting Prevention

Every training job uses an **experience replay buffer**: the training batch is composed of:
- 30% new confirmed samples from the current site
- 70% randomly sampled from all historical confirmed samples across all sites (anonymized)

This prevents the model from overfitting to recent site conditions at the expense of general detection capability. The 70/30 split is adjustable — for mature deployments with large historical buffers, shift to 50/50.

This is the standard replay-based continual learning approach. It is well-validated, simple to implement, and does not require any exotic architecture changes (no EWC, no LoRA, no sparse memory layers — those are overkill for this problem scale).

---

## 7. Admin Training Pipeline

### What Admins See

A web dashboard section labeled: **"Train for This Site"**

```
[ Upload Images ]  [ Label Them ]  [ Start Training ]  [ View Model Status ]
```

### Step 1: Image Upload

Admin drags and drops images. Supported: JPG, PNG, HEIC (auto-converted). No size limit (server resizes to 224×224 for training). Batch upload supported.

**What admins should upload (with guidance text in UI):**
- PPE specific to this site (type/color of vests, helmet styles, safety shoe requirements)
- Site-specific equipment and machinery (cranes, mixers, formwork types)
- Hazardous zones (confined spaces, excavation edges, electrical panels)
- Materials specific to this project (unusual or non-standard items)
- Site conditions at different times and lighting (dawn start, overcast, dust conditions)

**Minimum per class:** 15 images. UI warns if fewer than 15 are uploaded for any class.

### Step 2: Labeling

After upload, admin sees a simple grid of uploaded images. For each, they type or select a class name from a dropdown (pre-populated with standard construction categories, extensible with custom names).

**No bounding box annotation required.** Image-level classification labels are sufficient for the use case. Asking admins to draw bounding boxes is too much friction — they will not do it.

### Step 3: Training Job

Admin clicks "Train Model". Server does:

```
1. Load MobileNet-v3-Small pretrained on ImageNet (backbone frozen)
2. Replace final classification layer with new head
   (N classes = standard classes + admin-added site-specific classes)
3. Fine-tune head only: 20 epochs, lr=1e-3, batch_size=32
4. Data augmentation: horizontal flip, brightness ±20%, rotation ±10°
   (critical for robustness with small per-class sample counts)
5. Evaluate on held-out validation split (80/20)
6. If val_accuracy > previous model: promote to production
   Else: reject, notify admin, keep old model
7. Quantize to INT8 TFLite
8. Push to device fleet for this site
```

**Training time estimate:** For 10 classes × 20 images = 200 samples, fine-tuning only the head: 2–5 minutes on a V100/A100 GPU. Admin sees a progress indicator. Model status shows "Training... / Ready / Failed."

### Step 4: Validation Report

After training, admin sees:
- Per-class accuracy on validation set
- Classes where accuracy is low (< 70%) with recommendation to upload more images
- Comparison: old model vs new model accuracy
- Option to accept or reject the new model

This last step matters. Do not auto-deploy a worse model. Human approval before deployment is the correct safety gate.

---

## 8. Feature List — Complete

### Core Features (Ship for Hackathon)

| Feature | Layer | Description |
|---|---|---|
| Acoustic near-miss detection | On-device | YAMNet detects crash/shout/impact anomalies continuously |
| IMU startle detection | On-device | Sudden head motion (angular jerk) above threshold |
| Multimodal fusion trigger | On-device | AND gate: audio event + motion event within ±2s window |
| 30s pre-event audio buffer | On-device | Circular buffer always recording; frozen on trigger |
| Event-triggered frame capture | On-device | Single photo at moment of trigger; not streaming |
| Near-miss payload packaging | On-device | Structured JSON: timestamp, GPS, audio clip, frame, YAMNet class |
| Offline queue | On-device | SQLite queue; sync when connectivity restored |
| GPT-4o Vision scene description | Cloud | Natural language context appended to each event |
| Supervisor dashboard | Web | Real-time alert feed, audio player, frame viewer, confirm/dismiss |
| Site heatmap | Web | GPS-plotted near-miss density by zone |
| Trend analytics | Web | Near-miss rate by zone, time of day, worker, activity type |
| OSHA-format report export | Web | PDF near-miss report compatible with OSHA 300 log |
| Voice Q&A copilot | On-device + Cloud | Secondary: STT → intent router → RAG → LLM → TTS |
| Site document ingestion | Cloud | PDF parsing, chunking, embedding, vector DB |

### Extended Features (Post-Hackathon V2)

| Feature | Layer | Description |
|---|---|---|
| Admin image upload + labeling | Web | UI for uploading and labeling site-specific training images |
| Server-side fine-tuning | Cloud | Triggered by admin upload; updates visual classifier head |
| Model push to device | Cloud + Phone | Background OTA update; zero inference disruption |
| Supervised self-training | Cloud | Confirmed near-miss events added to replay buffer; scheduled retraining |
| Experience replay | Cloud | 70/30 historical/new sample mix; prevents catastrophic forgetting |
| Per-site acoustic calibration | On-device | 60s ambient baseline capture; adaptive threshold computation |
| Multilingual voice interface | On-device + Cloud | Whisper auto-detects language; LLM responds in same language |
| Worker-level risk scoring | Cloud | Per-worker near-miss frequency; flag for coaching intervention |
| Predictive zone risk | Cloud | ML model trained on site history to predict high-risk periods |
| Federated learning across sites | Cloud | Model improvements from Site A flow to Site B without raw data sharing |

---

## 9. System Architecture — End to End

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SENTINELSITE — FULL ARCHITECTURE                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│    META RAY-BAN GLASSES     │
│  ─────────────────────────  │
│  Microphone  → BT audio     │
│  IMU         → BT data      │
│  Camera      → event-only   │
└──────────────┬──────────────┘
               │ Bluetooth (Meta Wearables SDK)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ANDROID PHONE                                 │
│  ─────────────────────────────────────────────────────────────── │
│  INFERENCE LAYER (always running, negligible CPU)               │
│    ├── AudioRecord → 30s circular buffer                        │
│    ├── YAMNet TFLite → 521-class inference every ~1s (~15ms)    │
│    ├── IMU reader → angular velocity → jerk detector            │
│    └── Fusion gate → (audio_event AND motion_event) within 2s   │
│                                                                  │
│  EVENT HANDLER (fires on trigger, ~200ms total)                 │
│    ├── Freeze audio buffer (copy 30s to file)                   │
│    ├── Request camera frame (BT, ~200-400ms)                    │
│    ├── Read GPS coordinates                                     │
│    ├── Build NearMissPayload JSON                               │
│    └── Push to upload queue (SQLite)                            │
│                                                                  │
│  VOICE COPILOT (on-demand only)                                 │
│    ├── Wake-word detection → Whisper STT                        │
│    ├── Intent classifier → document type router                 │
│    └── HTTPS → Cloud RAG → TTS response                        │
│                                                                  │
│  BACKGROUND SERVICES                                            │
│    ├── Upload queue processor (LTE / WiFi sync)                 │
│    └── Model updater (silent OTA download, apply at restart)    │
└──────────────────────────────────┬───────────────────────────────┘
                                   │ HTTPS (event-driven, not streaming)
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CLOUD BACKEND (FastAPI + Postgres)            │
│  ─────────────────────────────────────────────────────────────── │
│  INGEST                                                          │
│    ├── /api/events POST → validate → store → queue for analysis │
│    ├── Audio stored → S3 with signed URL                        │
│    └── Frame stored → S3 with signed URL                        │
│                                                                  │
│  ANALYSIS PIPELINE (async, non-blocking)                        │
│    ├── Server-side YAMNet (higher quality, full context)        │
│    ├── GPT-4o Vision → scene description                        │
│    └── Risk scoring → zone heatmap update                       │
│                                                                  │
│  RAG ENGINE (voice copilot)                                     │
│    ├── Document ingestion: PDF → chunks → embeddings → Qdrant   │
│    ├── Query: embed → semantic search → top-k → LLM             │
│    └── Intent router: classify query → target document type     │
│                                                                  │
│  TRAINING PIPELINE (scheduled, server GPU)                      │
│    ├── Admin images → label store → fine-tuning job queue       │
│    ├── Confirmed events → training sample buffer                 │
│    ├── Scheduled training: head-only fine-tune with replay      │
│    ├── Validation: if accuracy improves → promote               │
│    └── INT8 quantize → TFLite export → push to device fleet     │
│                                                                  │
│  WEBSOCKET SERVER                                               │
│    └── Push near-miss alerts to supervisor dashboard in ~100ms  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                SUPERVISOR DASHBOARD (React)                      │
│  ─────────────────────────────────────────────────────────────── │
│  LIVE FEED       → near-miss cards with audio, frame, description│
│  REVIEW          → Confirm / Dismiss (1-click, < 5 seconds)     │
│  HEATMAP         → GPS density plot on site plan overlay        │
│  ANALYTICS       → trend by zone / time / activity type         │
│  ADMIN TRAINING  → upload images, label, trigger training       │
│  MODEL STATUS    → current version, accuracy, last update time  │
│  REPORTS         → OSHA-format PDF export                       │
└──────────────────────────────────────────────────────────────────┘

LATENCY BUDGET
  Acoustic inference:     ~15ms per 1s window        (continuous, negligible)
  IMU processing:         < 1ms per sample           (continuous, negligible)
  Fusion gate check:      < 1ms                      (continuous, negligible)
  Frame capture:          200–400ms (BT)             (on trigger only)
  Payload upload:         async, non-blocking         (no UX impact)
  Supervisor alert:       ~3–5s after event           (acceptable for review)
  Worker experience:      ZERO INTERRUPTION           (passive throughout)
```

---

## 10. What Is Real vs What Is a Bet

Be precise about this. Judges respect teams that know their uncertainty.

| Component | Status | Evidence | Hackathon Risk |
|---|---|---|---|
| YAMNet TFLite on Android | **Proven** | ~15ms inference, open source, AudioSet classes | None |
| AudioRecord circular buffer | **Proven** | Standard Android API, textbook pattern | None |
| IMU access via Meta SDK | **Likely** | SDK docs confirm sensor access; verify at init | Low — fallback to phone IMU |
| Event-triggered frame capture | **Proven** | SDK explicitly supports photo capture | None |
| Multimodal fusion gate | **Proven** | Simple AND gate, no ML, pure logic | None |
| GPT-4o Vision scene description | **Proven** | API available, well-tested | None |
| MobileNet fine-tuning pipeline | **Proven** | Documented, PMC study on construction edge deployment | Low |
| Admin image upload UI | **Buildable** | Standard web upload + labeling UI | Medium (time) |
| Self-training with replay | **Architecturally sound** | Replay buffer is standard continual learning | Medium — needs validation set |
| **YAMNet accuracy at 90dB ambient** | **UNKNOWN** | No published construction-specific study | **HIGH — core bet** |
| Threshold calibration per site | **Buildable** | Baseline PSD computation is signal processing | Medium — needs tuning time |
| Voice Q&A response < 2s | **Likely** | STT + retrieval + LLM adds to ~1.5–2s on LTE | Medium — latency variance |

**The one bet you must make explicit to judges:**

> "YAMNet's detection accuracy on construction-specific near-miss sounds in a 90dB ambient noise environment is the core open question. We have validated the architecture in a controlled setting. The production path requires either noise-robust fine-tuning on construction-specific audio data, or deployment in site areas with lower ambient noise levels (offices, staging areas, entry/exit zones). We are not claiming this is solved — we are claiming the architecture is correct and the engineering problem is well-defined."

That statement is more impressive than any team that pretends the problem doesn't exist.

---

## 11. Hackathon Demo Plan

### Priority Order (If Time Runs Short, Cut From Bottom)

**Must have for demo:**
1. End-to-end pipeline: audio event → fusion trigger → payload → supervisor card on dashboard
2. Audio player on dashboard (play the 30s pre-event buffer)
3. Frame viewer (show what the worker was seeing)
4. Confirm/Dismiss interaction (show human-in-the-loop)
5. At least one scripted near-miss scenario working reliably

**Should have:**
6. Site heatmap with GPS pins for confirmed events
7. YAMNet classification label + confidence on event card
8. GPT-4o Vision description on event card
9. OSHA-format PDF export from one confirmed event
10. Intent router for voice Q&A (shows manual selection intelligence)

**Nice to have:**
11. Admin image upload UI (upload → show model status change)
12. Multilingual voice demo (ask in Hindi, get answer in Hindi)
13. Trend analytics view (zone risk over time)

### The Three Demo Scenarios

**Scenario 1 — Classic Near-Miss (Primary)**  
Teammate wears glasses, is "working" with both hands occupied.  
Someone drops a heavy toolbox 3 meters away.  
Teammate flinches/head snaps up.  
**Expected:** Dashboard receives alert within 4 seconds.  
**Show:** Audio clip plays (crash sound). Frame shows POV. YAMNet: "Impact, heavy objects — 0.87". GPT-4o: "Worker appears to be at ground level near material storage. Object has fallen from surface at right of frame."

**Scenario 2 — Verbal Warning Near-Miss**  
Worker is moving near a "hazardous zone" (taped off area on floor).  
Colleague shouts "Watch out!" loudly.  
Worker's head motion fires fusion gate.  
**Show:** System logs event. Supervisor clicks Confirm. Selects OSHA category "Struck-by". PDF report generates with one click.  
**This shows the review-to-report loop.**

**Scenario 3 — Voice Copilot with Intent Routing (Secondary)**  
Worker says: "What are the concrete mix specifications for the columns?"  
System routes to "structural specifications" document (not safety manual, not schedule).  
Answers specifically with cited section.  
**Show:** The intent router choosing the right document — this is the "not just a generic RAG" differentiator.

### What You Say in the First 30 Seconds

> "Construction sites capture less than half of workplace injuries and almost no near-misses — not because workers don't care, but because reporting always requires the worker to do something after a stressful event, and they almost never do. We built the first system that captures near-miss events automatically, from the worker's first-person perspective, without interrupting their work. The worker does nothing. We do the witnessing. And because the model learns from confirmed events and from what the site manager teaches it, it gets more accurate for this specific site over time."

That is 75 words. It covers the problem, the insight, the passive sensing innovation, and the learning system — all before showing a single line of code.

---

## 12. Why You Win

### Against Other Hackathon Teams

| What They Build | Why Yours Wins |
|---|---|
| Voice chatbot + RAG | Requires worker attention. Adds cognitive load. Generic LLM without site-specific grounding. Your system requires zero worker action and gets smarter per-site. |
| Real-time YOLO PPE detection | Bluetooth cannot support reliable streaming. Latency kills the safety value. Your architecture sidesteps streaming entirely with event-triggered captures. |
| Generic "AI safety assistant" | Vague. Cannot show a concrete loop. You show: event → detection → log → supervisor review → risk intelligence. A complete system. |
| BIM AR overlay | Hardware doesn't support it. Anyone claiming this is lying about the specs. |

### The Data Moat (What Makes This a Real Company, Not a Hackathon Project)

Every confirmed near-miss event is a labeled training sample. After 100 sites × 6 months, you have a dataset that does not exist anywhere in the world: first-person, acoustically-classified, visually-contextualized, supervisor-confirmed near-miss records at scale.

With that dataset:
- Fine-tune a construction-specific acoustic model far more accurate than stock YAMNet on construction noise
- Build predictive risk models (what site conditions predict elevated near-miss frequency)
- License risk intelligence to construction insurers (AIG, Liberty Mutual actively price policies on risk data)
- Partner with OSHA research programs
- The model improves across the entire fleet every time any site's supervisor confirms an event

**The product sells safety. The moat is data. That is a fundamentally different company than a voice chatbot.**

### The Sentence That Wins the Room

> *"Every construction AI today talks to the trailer. We put the AI on the worker — not to interrupt them, but to watch for them. The worker does their job. We do the witnessing. And every site that deploys us makes the model smarter for every site that comes after."*

---

