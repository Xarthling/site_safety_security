# Worksite Safety Monitoring with Computer Vision

A system that watches work-site video and turns it into logged safety events that site managers can act on. It detects missing PPE (helmets, hi-vis vests) and vehicles, confirms violations over time, stores each event with evidence, and shows the results on a dashboard.

**Status:** Phase 1 in progress (research and design complete, working skeleton exists, not yet run on real footage). See [Current status](#current-status).

## Objective

Prove the full path from detection to dashboard on recorded CCTV footage:

> recorded clip → detect → track → apply rules → log event with snapshot → show on dashboard

Phase 1 is a pipeline proof with honestly measured accuracy. It is not a claim of production accuracy.

## Phase 1 scope

| Included | Deferred |
|---|---|
| Person, helmet, vest detection (ready-made PPE model) | Gloves, boots |
| Minimal vehicle detection (one coarse class, `vehicle`) | Plant-specific classes (excavator, loader and so on) |
| Rules: helmet missing, vest missing, vehicle in zone | Person-to-vehicle proximity |
| Recorded clips, offline processing | Live RTSP streams, edge deployment |
| SQLite event store, snapshots, annotated video | Notifications (email, SMS, chat) |
| Simple Django dashboard with false-positive review | Multi-site, user roles, reporting |
| Evaluation harness against hand-labelled events | Licence plate reading (needs privacy review) |

## Guiding principles

1. **Evaluate first.** Build ground truth and a scoring harness before choosing or training any model.
2. **Ready-made models first, own models second.** Train our own only when measured results say the public ones fall short.
3. **Independent models.** The PPE and vehicle models have separate weights, configs and versions, and meet only at the merge step.
4. **Events, not video, flow downstream.** Only event records and snapshots reach the database and dashboard.
5. **Model- and camera-specific details live in config,** not code: class names, thresholds, zones.
6. **Judge by events, not frames.**
7. **Record decisions and licences** so every result is reproducible and defensible.

## Architecture

Six layers, each talking only to the next through a small data contract, with evaluation and configuration spanning all of them.

| Layer | Job | Contract out |
|---|---|---|
| 1. Input | Turn a clip into timestamped frames | frame, video time, camera ID, clip name |
| 2. Perception | Find people, PPE and vehicles (independent models behind one `Detector` interface) | list of `Det(class, confidence, box)` |
| 3. Tracking | Stable IDs per camera (ByteTrack) | `TrackedPerson(track_id, box, confidence)` |
| 4. Rules | Turn observations into confirmed events | `Event(type, track_id, time, confidence, box)` |
| 5. Storage | Audit trail | `events` and `runs` tables plus snapshot files |
| 6. Dashboard | See and act on events | Django views reading events only |

How violations are decided:

- **Helmet / vest:** a "missing" class (for example `no-helmet`) in the expected body region means missing; the item itself there means ok; neither means unknown. Where a model has no "missing" class, missing can be inferred from absence on a person large enough to judge.
- **Small people:** anyone below a minimum pixel height is *unknown*, never a violation.
- **Timing:** a violation must persist for a confirmation time, produces one event per episode, and re-arms after a clear time with no violation. All times are video time, so recorded footage behaves repeatably.
- **Vehicle in zone:** a tracked vehicle inside the camera's zone polygon for the confirmation time produces one event per episode.

## Current status

A working skeleton (`ppe_pipeline/`) exists and covers layers 1 to 5 in simplified form:

- Rule engine with timers, size gating and one-event-per-episode logic
- ByteTrack wrapper (via the `trackers` package), SQLite event store, snapshots, annotated video
- Ultralytics detector wrapper, config loader, training script, weights download script
- 13 tests passing (rule logic plus an end-to-end run on a synthetic video with a scripted detector)

Not done yet:

- The detector wrapper has **not** been run against real weights or real footage
- Evaluation harness, `runs` table, review-status columns
- Minimal vehicle detector and `vehicle_in_zone` rule
- Django dashboard
- Restructure into the target layout below

## Quick start (current skeleton)

```bash
pip install -r requirements.txt
python -m pytest -q                       # no model needed

python download_weights.py                # saves weights/ppe.pt (see licensing)
python run_pipeline.py --video clips/site_a.mp4 --camera-id cam1 \
    --clip-start 2026-10-19T08:30:00+10:00 --out outputs/site_a_annotated.mp4
```

`run_pipeline.py` prints the model's class names and warns if `config.yaml` names a class the model does not have. Pass the real recording start with `--clip-start`; otherwise the file's modified time is used.

Outputs: `outputs/events.db` (table `events`), `outputs/snapshots/*.jpg`, and the annotated video.

## Clone and run with an NVIDIA GPU on Windows

The application passes `--device 0` to Ultralytics for GPU inference. For an NVIDIA
GTX 1660 Super or newer, use the CUDA-enabled dependency file rather than installing
the CPU-only PyTorch build:

```powershell
git clone https://github.com/Xarthling/site_safety_security.git
cd site_safety_security
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python verify_gpu.py
python download_weights.py
python run.py --labels data/labels.csv --device 0 --annotate
```

For the live wall:

```powershell
python live.py --labels data/labels.csv --device 0
```

The default configuration samples video at 5 FPS for inference. This is intentional:
the rules use video timestamps, so processing every source frame is unnecessary and
can make live playback fall far behind on slower hardware. Use `--device 0` after
`verify_gpu.py` reports CUDA availability. `--speed` changes playback speed, not
the detector's inference throughput.

## Local review dashboard

After at least one run has created `outputs/events.db`, start the local dashboard
from the repository root:

```powershell
python dashboard/manage.py runserver 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/> to review safety-event evidence, mark events as
confirmed or false positives, and inspect logged vehicles and plates. The dashboard
reads the existing SQLite tables directly and serves snapshots only from the
configured `outputs/snapshots` directory. It has no authentication and is intended
for local use; do not expose this development server to a network.

The computer must have a current NVIDIA display driver. `verify_gpu.py` must report
`CUDA available: True` before GPU inference is expected. Use `--device cpu` when
running without CUDA. The project keeps clips, labels, model weights, databases,
snapshots, and annotated videos out of Git; copy those files separately to a cloned
machine.

Do not publish real footage, labels containing personal information, or readable
licence plates in a public repository. Before sharing this public repository, change
`gate.store_plate` in `config.yaml` to `hash` or `none` and use a private data copy
for any required readable plate testing.

## Configuration

`config.yaml` is the single config file. It holds the detector weights and class names, processing rate, per-item rules and thresholds, and (as sections inside the same file) each camera's zones. Starting values, to be tuned on real clips:

| Parameter | Start value |
|---|---|
| Processing rate | 5 frames per second |
| Inference size | 960 px |
| Detection confidence | 0.25 |
| Minimum person height | 80 px |
| Confirmation time | 3 s |
| Clear time | 2 s |
| Helmet band | top 40% of person box |
| Vest band | 15% to 75% of person box |

## Target repository structure

Kept deliberately minimal: one file per layer, plain functions and small data classes, one config file.

```
worksite-safety/
├── README.md
├── requirements.txt
├── config.yaml          model weights, class names, thresholds, zones, paths
├── run.py               entry point: wires the layers together for one clip
├── evaluate.py          scores a run against hand-labelled events
├── safety/
│   ├── types.py         shared data classes: Det, TrackedPerson, Event
│   ├── ingest.py        layer 1: clip to timestamped frames
│   ├── perception.py    layer 2: Detector interface, YOLO wrapper
│   ├── tracking.py      layer 3: ByteTrack wrapper
│   ├── rules.py         layer 4: statuses, timers, events
│   ├── storage.py       layer 5: SQLite events and runs, snapshots
│   └── annotate.py      drawing boxes on frames and snapshots
├── dashboard/           layer 6: Django project, created when we reach it
├── tests/               test_rules.py, test_pipeline.py
├── data/                clips and labels.csv   (gitignored)
├── weights/                                    (gitignored)
└── outputs/             events.db, snapshots, annotated videos   (gitignored)
```

Working rules:

- **One file per layer.** Each file imports only from `types.py` and, at most, the layer directly before it. No layer imports from `run.py`.
- **`run.py` is the only place layers meet.** It reads `config.yaml` and passes plain values into functions, so layers do not depend on the config loader.
- **Keep it simple.** No frameworks or plugin systems, and no abstract base classes beyond the one `Detector` interface needed to swap models.
- **Easy to debug.** Each layer can be run alone with fake inputs, which is what the tests do. Print clear messages for problems such as a class name missing from the model.
- **Split only when needed.** If a file passes roughly 300 lines, split it then.

How later work fits without new structure:

- **Vehicle model:** a second `Detector` instance in `perception.py` plus a config section.
- **Plate reader (if added):** one new file, `plates.py`.
- **Training (Stage B):** a top-level `train.py` when we get there.
- **Runs:** a `runs` table in the database, not a folder per run.

## Plan

| Phase | Focus | Exit criterion |
|---|---|---|
| 0. Foundations | Footage access, licensing and legal decisions, ground truth | Footage, licence decision, labelled evaluation set |
| 1. Proof of path (Stage A) | PPE plus minimal vehicle detection, ready-made models, database, dashboard | Full path works, baseline accuracy measured |
| Gate | Compare results to agreed thresholds | Supervisor decides: continue or move to Stage B |
| 2. Trustworthy (Stage B) | Harvest failures, label, fine-tune, retest on the same set | New model beats baseline on the untouched test set |
| 3. Plant and zones | Plant-specific classes, richer zone rules, proximity | Zone rules validated on clips |
| 4. Live and edge | RTSP streams, on-site inference, notifications | Stable live operation |
| 5. Operable and compliant | Multi-camera and multi-site, roles, retention, privacy controls | Deployable under the target state's rules |

## Evaluation

Accuracy is measured on **events**, not frames. True violations are hand-labelled per clip (clip, camera, type, start, end). A detected event matches a label when camera and type agree and times overlap within an agreed tolerance. Reported: event precision, recall, false alarms per camera-hour, and confirmation delay, broken down by day or night and by worker distance. Training data is always split by clip and camera, never by frame.

## Licensing and compliance

- **Ultralytics YOLO** code and weights are AGPL-3.0 (a commercial licence exists). Do not hand code, containers or hosted access to partners until this is resolved. RF-DETR (smaller sizes, Apache 2.0) is the permissive alternative behind the same `Detector` interface.
- The default downloaded PPE checkpoint comes from `ZakariaHibaoui2/yolo-object-detection`. The repo says MIT, but the checkpoint metadata says AGPL-3.0, no accuracy figures are published, and the training dataset licence is not stated.
- Public datasets (SH17, CHV, Construction-PPE, MOCS, ACID and others) each carry their own terms. Keep a register of every library, weight file and dataset with its licence and source.
- **Privacy and surveillance:** Australian rules vary by state and territory (for example, NSW requires at least 14 days' written notice before camera surveillance and prohibits it in change rooms, toilets and showers). Confirm footage consent and retention, mask private areas, set a snapshot retention period, and obtain legal advice before deploying. Licence plate reading is excluded from Phase 1 pending a privacy review.
- **Standards:** helmets fall under AS/NZS 1801 and hi-vis garments under AS/NZS 4602.1 (verify current editions). A camera can show an item is present, not that it is certified, so rules say "helmet visible", never "compliant".

## Known limitations

- Track IDs are per camera and can switch under occlusion, which can split or repeat events.
- PPE boxes are matched to people by position, so overlapping workers can be mixed up.
- Small, compressed or distant workers may make helmet and vest detection unreliable on average CCTV.
- COCO-pretrained vehicle detection may miss site plant such as excavators.
- Public models and datasets are mostly non-Australian, so a domain gap is expected.

## Documents

- Project plan (high-level and low-level)
- Phase 1 scope summary
- Supervisor summary and decisions needed
