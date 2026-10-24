"""Draw a zone on a frame of a clip and print the numbers to paste into config.yaml.

    python define_zone.py --video data/clips/c7g.mp4 --zone gate_zone
    python define_zone.py --video data/clips/c7g.mp4 --zone detection_zone --time 5

In the window:  left click = add a corner   left/right (or a/d) = change frame
u = undo the last corner   Enter = finish   q or Esc = cancel
Corners go around the zone in order. Zones are printed as fractions of the frame, so they keep working
if the camera resolution changes. Paste the printed line under  cameras:  <camera_id>:  in config.yaml,
then check it with  python live.py --video <clip> --camera-id <camera_id>  (zones are drawn on the picture).
"""
import argparse

import cv2
import numpy as np

from safety.zones import format_polygon


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read frame {frame_index}. Try a smaller --time.")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--zone", default="detection_zone", choices=["detection_zone", "gate_zone"])
    ap.add_argument("--time", type=float, default=0.0, help="which second of the clip to draw on")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, round(args.time * fps))
    if frame_count:
        frame_index = min(frame_index, frame_count - 1)
    frame = read_frame(cap, frame_index)
    h, w = frame.shape[:2]
    points = []
    title = f"{args.zone}: click corners, arrows/A,D = frame, Enter = done, q = cancel"
    max_preview_width, max_preview_height = 1280, 720
    scale = 1.0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((round(x / scale), round(y / scale)))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        scale = min(1.0, max_preview_width / w, max_preview_height / h)
        if scale < 1.0:
            view = cv2.resize(frame, (round(w * scale), round(h * scale)),
                              interpolation=cv2.INTER_AREA)
        else:
            view = frame.copy()
        if points:
            display_points = np.array([(round(x * scale), round(y * scale)) for x, y in points],
                                      dtype=np.int32)
            cv2.polylines(view, [display_points], len(points) >= 3, (220, 0, 220), 2)
            for p in points:
                cv2.circle(view, (round(p[0] * scale), round(p[1] * scale)), 4, (0, 255, 255), -1)
        timestamp = frame_index / fps
        cv2.putText(view, f"Frame {frame_index + 1}"
                    + (f"/{frame_count}" if frame_count else "")
                    + f"  {timestamp:.2f}s",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(title, view)
        key = cv2.waitKeyEx(30)
        if key in (13, 10):  # Enter
            if len(points) < 3:
                print("A zone needs at least 3 corners.")
                continue
            break
        if key in (ord("u"), ord("U")) and points:
            points.pop()
        if key in (ord("a"), ord("A"), 81, 2424832):
            new_index = max(0, frame_index - 1)
        elif key in (ord("d"), ord("D"), 83, 2555904):
            new_index = frame_index + 1
            if frame_count:
                new_index = min(new_index, frame_count - 1)
        else:
            new_index = frame_index
        if new_index != frame_index:
            frame_index = new_index
            frame = read_frame(cap, frame_index)
            points.clear()
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            cap.release()
            raise SystemExit("Cancelled.")
    cv2.destroyAllWindows()
    cap.release()

    print(f"\nPaste this into config.yaml under cameras: <camera_id>:\n")
    print(f"    {args.zone}: {format_polygon(points, w, h)}")


if __name__ == "__main__":
    main()
