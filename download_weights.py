"""Download model weights into weights/.

  weights/ppe.pt      PPE model (helmet, no-helmet, vest, no-vest, person)
                      Source: https://github.com/ZakariaHibaoui2/yolo-object-detection (models/best_2025.pt)
                      The repo is MIT licensed, but the checkpoint was made with Ultralytics (its metadata says
                      AGPL-3.0) and the training data licence is not stated: confirm before sharing outside your team.
  weights/yolo11n.pt  General COCO model used for vehicles at the gate (car, truck, bus, motorcycle ...)
                      Source: Ultralytics assets release. AGPL-3.0.

Licence plate models (fast-alpr) download themselves the first time the gate camera runs.

    python download_weights.py            # both
    python download_weights.py ppe        # or: vehicle
"""
import sys
import urllib.request
from pathlib import Path

FILES = {
    "ppe": ("https://github.com/ZakariaHibaoui2/yolo-object-detection/raw/main/models/best_2025.pt",
            Path("weights/ppe.pt")),
    "vehicle": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                Path("weights/yolo11n.pt")),
}

wanted = sys.argv[1:] or list(FILES)
for name in wanted:
    if name not in FILES:
        raise SystemExit(f"Unknown model {name!r}. Choose from: {', '.join(FILES)}")
    url, dest = FILES[name]
    dest.parent.mkdir(exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
