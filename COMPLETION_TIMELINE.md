# Completion Timeline

## Goal

Complete and demonstrate the Phase 1 worksite safety monitoring prototype within
1.5 weeks (10 focused working days).

The target is a reliable offline demonstration:

```text
Video clip
    -> PPE and vehicle detection
    -> Tracking and zone processing
    -> Confirmed safety event
    -> Database record and evidence snapshot
    -> Dashboard review
    -> Measured evaluation result
```

This timeline targets Phase 1 completion. It does not target production
deployment, live RTSP operation, or full legal/compliance certification.

## Expected outcome

By the end of the timeline, the project should demonstrate:

- PPE detection for helmets and hi-vis vests.
- Vehicle detection and tracking.
- Detection and gate-zone behavior.
- Confirmed safety events with timing rules.
- SQLite event and vehicle records.
- Evidence snapshots and annotated videos.
- Local dashboard review.
- Event evaluation against hand-labelled footage.
- Documented limitations, privacy decisions, and reproducible commands.

## Ten-day schedule

### Day 1: Stabilize the codebase

**Objectives**

- Fix the current zone-related test regression.
- Confirm configuration parsing.
- Confirm the existing test suite.
- Freeze the Phase 1 feature scope.

**Tasks**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
python dashboard\manage.py check
python run.py --help
python live.py --help
python define_zone.py --help
```

Update the synthetic pipeline test to use an isolated test configuration
without restrictive production camera zones.

**Deliverable**

- Green automated test suite.
- No configuration parsing errors.
- Frozen list of Phase 1 features.

**Exit criterion**

All automated tests pass.

### Day 2: Validate the complete execution path

**Objectives**

Run the complete path from video input to dashboard records.

**Tasks**

```powershell
python run.py --video data/clips/c4.mp4 --camera-id gate_cam --annotate
python live.py --video data/clips/c4.mp4 --camera-id gate_cam
python dashboard\manage.py runserver 127.0.0.1:8000
```

Verify:

- An annotated video is generated.
- PPE detections and zones are visible.
- Events are written to SQLite.
- Evidence snapshots are created.
- Vehicle records are written.
- Dashboard records are visible.
- Review actions work.

**Deliverable**

One successful end-to-end run with its input clip, annotated video, database
records, snapshots, and dashboard evidence.

**Exit criterion**

The complete path works without manual database editing.

### Day 3: Finalize zone configuration

**Objectives**

Calibrate the camera zones using representative frames.

**Tasks**

```powershell
python define_zone.py --video data/clips/c4.mp4 --zone detection_zone
python define_zone.py --video data/clips/c4.mp4 --zone gate_zone
```

For each camera, check frames:

- Before a vehicle enters.
- While it is inside.
- After it leaves.
- With people at the edges of the detection area.

Confirm that:

- `detection_zone` covers the area where people should be analyzed.
- `gate_zone` covers the meaningful vehicle passage area.
- The gate zone is not only a thin line.
- A vehicle's bottom-center anchor enters the polygon.

**Deliverable**

Final camera-zone values in `config.yaml`.

**Exit criterion**

Annotated output shows correctly positioned zones and vehicle movement through
the gate zone.

### Day 4: Test and tune event behavior

**Objectives**

Reduce unstable, duplicate, or missed events.

**Test cases**

- Person entering and leaving the detection zone.
- Vehicle entering the gate zone.
- Vehicle leaving the gate zone.
- Vehicle stopping inside the gate zone.
- Vehicle disappearing from the frame.
- Multiple vehicles.
- Boundary crossings.
- Brief detection gaps.

Tune only the relevant timing settings initially:

```yaml
rules:
  confirm_seconds: 3.0
  clear_seconds: 2.0

gate:
  zone_hold_frames: 2
  finalize_after_s: 3.0
```

Record expected and actual behavior for every test case.

**Deliverable**

A zone and event validation table.

**Exit criterion**

One continuous violation or vehicle crossing does not create repeated events.

### Day 5: Create the evaluation set

**Objectives**

Prepare a small representative hand-labelled dataset.

Select clips containing:

- At least three camera views, where available.
- Day and low-light conditions, where available.
- Near and distant workers.
- Clear and partially occluded workers.
- Several vehicle passages.
- Helmet and vest violations.
- At least one no-event clip.

Use the existing event-label format:

```text
clip,camera_id,event_type,start_s,end_s,condition
```

Example:

```csv
clip,camera_id,event_type,start_s,end_s,condition
c4.mp4,gate_cam,no_helmet,2.0,6.0,day
c4.mp4,gate_cam,vehicle_in,1.0,3.0,day
```

**Deliverable**

An evaluation file such as:

```text
data/evaluation_labels.csv
```

**Exit criterion**

Every selected clip has understandable labels.

### Day 6: Measure the baseline

**Objectives**

Measure system performance objectively against the evaluation set.

Record:

- True positives.
- False positives.
- False negatives.
- Precision.
- Recall.
- False alarms per camera-hour.
- Confirmation delay.
- Duplicate event count.

Summarize results by:

- Camera.
- Event type.
- Lighting condition.
- Worker distance.

**Deliverable**

A baseline accuracy report.

**Exit criterion**

The team can explain where the system succeeds and where it fails.

### Day 7: Fix high-impact issues

**Objectives**

Improve the most important measured failure modes without broad redesign.

Prioritize:

1. Missed gate events.
2. Duplicate PPE events.
3. Excessive false alarms.
4. Incorrect zone filtering.
5. Broken snapshots or database records.
6. Incorrect timestamps.

Potential changes:

- Adjust zone polygons.
- Adjust confirmation or clear times.
- Adjust `zone_hold_frames`.
- Adjust confidence thresholds.
- Correct event persistence.
- Add regression tests for observed failures.

Run a second measured evaluation and compare it to the baseline.

**Deliverable**

A second result set showing whether the changes improved the system.

**Exit criterion**

At least the major known failure mode improves without breaking other behavior.

### Day 8: Complete dashboard and evidence review

**Objectives**

Make the dashboard ready for local demonstration and review.

Verify:

- Events are visible.
- Vehicle records are visible.
- Evidence snapshots load.
- Filters work.
- Review status changes work.
- Zone events show meaningful details.
- Empty database and empty-filter states work.
- Invalid snapshot paths return `404`.

Keep the scope focused on local review. Do not add authentication, complex
reporting, or multi-site support unless they are explicit project requirements.

**Deliverable**

A clean local dashboard demonstration.

**Exit criterion**

A reviewer can understand what happened, when it happened, which camera
produced it, what evidence exists, and whether it was reviewed.

### Day 9: Documentation, privacy, and reproducibility

**Objectives**

Make the project understandable and defensible.

Update the README to accurately describe:

- Implemented functionality.
- Tested functionality.
- Unvalidated functionality.
- Setup and run commands.
- Zone configuration.
- Dashboard startup.
- Evaluation commands.
- Known limitations.
- Privacy and licensing constraints.

Decide whether licence plates should be stored as:

```yaml
store_plate: text
```

for controlled testing, or:

```yaml
store_plate: hash
```

for a safer demonstration.

Document:

- Model and dataset licences.
- Privacy restrictions.
- Snapshot retention.
- Dashboard localhost-only status.

**Deliverable**

Accurate project documentation and a reproducible run procedure.

**Exit criterion**

Another developer can set up the environment and reproduce the demonstration.

### Day 10: Final demonstration and release decision

**Objectives**

Perform final validation and package the Phase 1 result.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
python run.py --video data/clips/c4.mp4 --camera-id gate_cam --annotate
python dashboard\manage.py check
python dashboard\manage.py runserver 127.0.0.1:8000
```

Prepare:

- Annotated video.
- Dashboard screenshots.
- Evaluation results.
- Test results.
- Known limitations.
- Configuration summary.
- Demonstration script.

The demonstration should show:

1. A clip being processed.
2. Detection and gate zones visible.
3. A PPE event being generated.
4. A vehicle event being generated.
5. A snapshot being stored.
6. The event appearing in the dashboard.
7. An event being reviewed.
8. Evaluation metrics being reported.

**Deliverable**

A Phase 1 completion package.

**Exit criterion**

The offline prototype is demonstrable, tested, and measured.

## Success checklist

- [ ] All automated tests pass.
- [ ] Configuration loads correctly.
- [ ] PPE events are generated from recorded clips.
- [ ] Vehicle events are generated from recorded clips.
- [ ] Detection and gate zones are visible in annotated video.
- [ ] Events and vehicles are stored in SQLite.
- [ ] Evidence snapshots are available.
- [ ] Dashboard displays current database records.
- [ ] Dashboard review actions work.
- [ ] A labelled evaluation subset exists.
- [ ] Baseline precision and recall are measured.
- [ ] Major known errors are documented.
- [ ] Plate-storage and privacy behavior is explicitly decided.
- [ ] README accurately reflects the implementation.
- [ ] A repeatable demonstration procedure exists.

## Scope boundaries

The following items are outside the critical path for this 1.5-week target:

- RTSP and live camera integration.
- Notifications by email, SMS, or chat.
- Multi-user authentication and roles.
- Multi-site support.
- Mobile application.
- Large-scale custom model training.
- Advanced analytics.
- Cloud deployment.
- Full legal or compliance certification.

These should be planned as later phases rather than allowed to delay the
Phase 1 demonstration.

## Final target statement

At the end of this timeline, the project should be described as:

> A functional Phase 1 worksite safety monitoring prototype that completes the
> offline video-to-dashboard path, including PPE rules, vehicle tracking, gate
> zones, evidence storage, annotated video, evaluation tooling, and a local
> review dashboard. Real-footage accuracy, privacy approval, production
> hardening, and live deployment remain outstanding.
