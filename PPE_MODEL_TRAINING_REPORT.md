# PPE Detection Model: Training Requirements and Projected Outcomes

## Executive summary

The project currently detects people, helmets, missing helmets, vests, and
missing vests, then converts persistent detections into PPE events. A custom
model is justified only if the current checkpoint fails on representative site
footage. The recommended approach is transfer learning from a licensed
YOLO-compatible checkpoint, using site-specific CCTV frames and evaluating on
held-out clips and cameras.

The target is not merely high frame-level detection accuracy. The model must
produce reliable event-level results after tracking, region matching, and
confirmation timing.

## Current target classes

The model interface and configuration expect:

| Class | Meaning |
|---|---|
| `person` | Worker/person bounding box |
| `helmet` | Helmet visibly worn |
| `no-helmet` | Helmet absent or visibly missing |
| `vest` | Hi-vis vest visibly worn |
| `no-vest` | Hi-vis vest absent or visibly missing |

Recommended labeling policy:

- Label `person` for every visible worker that is relevant to the camera.
- Label PPE only when the item/status is visually supportable.
- Use `unknown`/ignore regions for severe occlusion, blur, or insufficient size;
  do not force `no-helmet` or `no-vest` labels from ambiguous evidence.
- Do not label certification or compliance; label only visible PPE status.

## Minimum data requirements

### Dataset volume

| Dataset component | Minimum | Preferred |
|---|---:|---:|
| Distinct source clips | 20 | 50+ |
| Distinct cameras/locations | 3 | 5–10 |
| Annotated frames | 5,000 | 15,000–30,000 |
| Worker instances | 10,000 | 30,000+ |
| Helmet/vest positive instances | 3,000 each | 10,000+ each |
| Missing-helmet instances | 1,000 | 3,000+ |
| Missing-vest instances | 1,000 | 3,000+ |
| Difficult/negative frames | 1,000 | 3,000+ |

These are planning targets, not guarantees. Diversity and label quality matter
more than simply increasing frame count.

### Required scene diversity

Include:

- Daylight, shade, low light, glare, and rain where applicable.
- Near, medium, and distant workers.
- Different camera heights, angles, and fields of view.
- Front, rear, side, and partially turned workers.
- Different helmet colors, vest colors, clothing, and skin tones.
- Occlusion by vehicles, equipment, barriers, and other workers.
- Motion blur, compression artifacts, shadows, and crowded scenes.
- Correctly worn PPE and genuine non-compliance.
- Empty scenes and objects that could be confused with PPE.

The training distribution should resemble deployment footage. Public datasets
can improve initialization, but should not replace site footage.

## Annotation requirements

Use one object-detection format consistently, preferably YOLO:

```text
class_id x_center y_center width height
```

Requirements:

- Tight, visible bounding boxes.
- Consistent treatment of partially visible people and PPE.
- A written labeling guide with examples.
- Double-review at least 10% of labels.
- Resolve disagreements before training.
- Track annotator, source clip, camera, lighting, and difficulty metadata.
- Record dataset and model licences.

Avoid frame leakage. Frames from one source clip must remain in one split.
Never place adjacent frames from the same video in both training and test sets.

Recommended split:

```text
70% training clips
15% validation clips
15% final test clips
```

The final test clips and cameras must remain untouched until model selection is
complete. If possible, hold out one complete camera for a domain-generalization
test.

## Training environment

### Hardware

Minimum practical setup:

- NVIDIA GPU with at least 8 GB VRAM.
- 16 GB system RAM.
- SSD storage with at least 50 GB free.
- CUDA-enabled PyTorch matching the installed driver.

Preferred:

- 12–24 GB VRAM.
- 32 GB system RAM.
- Fast SSD.
- Fixed random seeds and recorded package versions.

The repository already provides a CUDA dependency path in
[requirements-gpu.txt](D:/Site_safety_security/requirements-gpu.txt) and a
hardware check in [verify_gpu.py](D:/Site_safety_security/verify_gpu.py).

### Software and reproducibility

Record:

- Python version.
- PyTorch, CUDA, Ultralytics, and OpenCV versions.
- Base checkpoint hash and licence.
- Dataset manifest and version.
- Class order.
- Image size.
- Batch size.
- Epoch count.
- Augmentations.
- Confidence and IoU thresholds.
- Random seed.
- Git commit/config hash.

Do not overwrite the current production/test checkpoint. Save trained weights
under a versioned path such as:

```text
weights/ppe_site_v1.pt
```

## Recommended training configuration

Start with transfer learning rather than training from scratch:

| Parameter | Initial recommendation |
|---|---|
| Model | Small/medium YOLO detector sized for available VRAM |
| Image size | 960 px, matching current inference configuration |
| Epochs | 50–100 with early stopping |
| Batch size | Largest stable batch fitting GPU memory |
| Optimizer | Framework default initially; tune only after baseline |
| Learning rate | Framework transfer-learning default |
| Augmentation | Horizontal flip, scale, crop, brightness/contrast, blur, weather/lighting |
| Mosaic/mixup | Moderate; verify that small PPE remains realistic |
| Early stopping | 15–25 validation epochs without improvement |
| Checkpointing | Save best validation checkpoint and final checkpoint |
| Inference confidence | Begin at current `0.25`, tune on validation data |
| NMS IoU | Use framework default, then validate crowded-scene behavior |

Do not over-augment away the visual characteristics of the site. The training
image size should preserve small helmets and vests without making inference too
slow for the target hardware.

## Evaluation requirements

### Model-level metrics

Report per class:

- Precision.
- Recall.
- F1 score.
- AP50.
- AP50–95.
- Confusion matrix.
- False positives per hour on negative footage.
- Inference FPS/latency.
- GPU memory use.

Break metrics down by:

- Camera.
- Lighting.
- Worker distance/size.
- Occlusion.
- PPE color/type.

### System-level metrics

The project should prioritize event metrics over isolated frame metrics:

- PPE event precision.
- PPE event recall.
- False alarms per camera-hour.
- Duplicate events per violation episode.
- Confirmation delay.
- Missed-event rate.
- Snapshot availability.
- Track fragmentation rate.

An isolated false frame should not become an event because the rules layer
requires persistence. Conversely, a model with good AP but unstable tracks may
still produce poor operational results.

## Acceptance targets for a Phase 1 model gate

These are recommended decision thresholds, not guaranteed outcomes:

| Measure | Minimum gate | Preferred |
|---|---:|---:|
| Person recall on relevant workers | 90% | 95%+ |
| Helmet/vest event recall | 80% | 90%+ |
| Helmet/vest event precision | 85% | 92%+ |
| False alarms | <2 per camera-hour | <1 per camera-hour |
| Duplicate events | <10% of true episodes | <5% |
| Confirmation delay | ≤ configured 3–5 s | ≤3 s |
| Held-out-camera recall drop | <15 percentage points | <10 points |

The model should not be approved because it meets aggregate metrics while
failing a safety-critical camera or lighting condition. Review per-camera
results and inspect false negatives manually.

## Projected outcomes

### If the current checkpoint is already close to site conditions

Expected outcome after fine-tuning:

- Fewer false PPE detections from site-specific backgrounds.
- Better recognition of local helmet and vest colors.
- Better performance at the project’s camera angles and compression levels.
- More stable person/PPE associations.
- Reduced event noise after threshold tuning.
- Improved recall for recurring site-specific worker poses.

Indicative improvement range, subject to dataset quality:

```text
Event precision: +5 to +15 percentage points
Event recall:    +5 to +20 percentage points
False alarms:    20% to 60% reduction
```

### If the current checkpoint has a major domain gap

Fine-tuning may produce larger gains, but the project may still need:

- More distant-worker examples.
- Camera-specific models or thresholds.
- A larger helmet/vest dataset.
- Better labels for ambiguous or occluded cases.
- A different model architecture.
- A separate detector for difficult site conditions.

### What training will not solve by itself

Training cannot guarantee:

- Correct legal or safety compliance decisions.
- Reliable detection below the camera’s effective resolution.
- Stable IDs through severe occlusion.
- Correct event timing if zones or timers are wrong.
- Detection of PPE that is fully hidden.
- Generalization to cameras absent from the dataset.
- Production readiness, privacy approval, or operational uptime.

## Recommended experiment sequence

1. Run the current checkpoint on the locked evaluation set.
2. Record event-level and model-level baseline metrics.
3. Inspect the top false positives and false negatives.
4. Annotate targeted site examples, not random extra frames.
5. Fine-tune one controlled model version.
6. Compare against the same untouched test set.
7. Re-run the full pipeline with tracking, zones, timing, snapshots, and
   dashboard review.
8. Keep the new model only if it improves agreed metrics without unacceptable
   regressions.

Use an experiment table:

| Version | Dataset | AP50 | Event precision | Event recall | False alarms/hr | FPS | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | site test v1 |  |  |  |  |  |  |
| ppe-site-v1 | site train v1 |  |  |  |  |  |  |
| ppe-site-v2 | targeted failures |  |  |  |  |  |  |

## Recommended timeline

| Work | Duration |
|---|---:|
| Define classes and labeling guide | 0.5–1 day |
| Prepare and review 5,000–10,000 frames | 3–7 days |
| Dataset QA and split validation | 1 day |
| Baseline inference/evaluation | 0.5–1 day |
| Fine-tuning experiments | 1–2 days |
| Error analysis and threshold tuning | 1–2 days |
| Final held-out evaluation and integration | 1 day |

An initial fine-tuned model can be produced in 1.5 weeks only with footage and
labels already available. Creating a high-quality, diverse dataset from
scratch generally requires longer.

## Go/no-go decision

Proceed to training when:

- The current model has a measured failure pattern.
- Representative site footage is available.
- Label ownership and privacy permissions are clear.
- The checkpoint and dataset licences permit the intended use.
- A locked test set exists.
- Acceptance thresholds are agreed.

Do not proceed directly to deployment when:

- Evaluation uses only adjacent frames from the training clips.
- Labels force uncertain cases into `no-helmet` or `no-vest`.
- Results are reported only as frame accuracy.
- A single camera dominates the dataset.
- Licence or privacy status is unresolved.
- Event-level behavior has not been tested through tracking and timing.

## Bottom line

The ideal requirement is a licensed, reproducible transfer-learning workflow
built around diverse site footage, carefully reviewed bounding boxes, clip-level
splits, and event-level acceptance tests. A realistic successful outcome is a
meaningful reduction in false alarms and improved recall on the project’s
cameras—not a guarantee of universal PPE compliance detection.
