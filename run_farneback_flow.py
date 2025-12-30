import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


def parse_roi(roi_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if roi_str is None:
        return None
    parts = roi_str.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be four comma-separated integers: x1,y1,x2,y2")
    try:
        coords = tuple(int(p) for p in parts)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    x1, y1, x2, y2 = coords
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("ROI must satisfy x2>x1, y2>y1 and be non-negative")
    return x1, y1, x2, y2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Farneback optical flow on a hand motion video.")
    parser.add_argument("--video", default="input/hand.avi", help="Path to input video. Default: input/hand.avi")
    parser.add_argument("--out_dir", default="output", help="Directory to save outputs.")
    parser.add_argument("--viz", choices=["hsv", "quiver", "both"], default="both", help="Visualization type to save.")
    parser.add_argument("--resize", type=int, default=0, help="Resize longest side to this size (0 = no resize).")
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help="ROI as x1,y1,x2,y2 (only compute/visualize flow inside this region).",
    )
    parser.add_argument("--quiver_step", type=int, default=16, help="Grid step for quiver visualization.")
    parser.add_argument("--pyr_scale", type=float, default=0.5, help="Parameter, see cv2.calcOpticalFlowFarneback.")
    parser.add_argument("--levels", type=int, default=3, help="Number of pyramid levels including the initial image.")
    parser.add_argument("--winsize", type=int, default=15, help="Averaging window size.")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations at each pyramid level.")
    parser.add_argument("--poly_n", type=int, default=5, help="Size of the pixel neighborhood used to find polynomial expansion.")
    parser.add_argument("--poly_sigma", type=float, default=1.2, help="Standard deviation of the Gaussian used to smooth derivatives.")
    parser.add_argument("--print_every", type=int, default=1, help="How often to print per-frame stats (in frames).")
    return parser


def ensure_video(path: str) -> None:
    if not os.path.exists(path):
        sys.exit(f"Input video not found: {path}. Please place the file in input/hand.avi or pass --video.")


def resize_frame(frame: np.ndarray, target_long_side: int) -> np.ndarray:
    if target_long_side <= 0:
        return frame
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side == target_long_side:
        return frame
    scale = target_long_side / float(long_side)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def validate_roi(roi: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = frame_shape
    x1, y1, x2, y2 = roi
    if x2 > w or y2 > h:
        raise ValueError(f"ROI {roi} is outside the frame after resizing: width={w}, height={h}")
    return roi


def create_writer(path: str, fps: float, frame_size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, str]:
    preferred_codecs = ["mp4v", "MJPG"]
    fourcc_used = ""
    writer = None
    for codec in preferred_codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if writer.isOpened():
            fourcc_used = codec
            break
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {path}")
    return writer, fourcc_used


def flow_to_hsv(flow: np.ndarray, template_shape: Tuple[int, int, int], roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    hsv = np.zeros(template_shape, dtype=np.uint8)
    hsv[..., 1] = 255
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    if roi is None:
        hsv[..., 0] = angle / 2
        hsv[..., 2] = mag_norm.astype(np.uint8)
    else:
        x1, y1, x2, y2 = roi
        hsv[y1:y2, x1:x2, 0] = angle / 2
        hsv[y1:y2, x1:x2, 2] = mag_norm.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_quiver(frame: np.ndarray, flow: np.ndarray, step: int, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    h, w = frame.shape[:2]
    output = frame.copy()
    x_start, y_start, x_end, y_end = 0, 0, w, h
    if roi is not None:
        x_start, y_start, x_end, y_end = roi
    for y in range(y_start, y_end, step):
        for x in range(x_start, x_end, step):
            fx, fy = flow[y - y_start, x - x_start] if roi is not None else flow[y, x]
            end_point = (int(x + fx), int(y + fy))
            cv2.arrowedLine(output, (x, y), end_point, color=(0, 255, 0), thickness=1, tipLength=0.3)
    if roi is not None:
        cv2.rectangle(output, (x_start, y_start), (x_end, y_end), (0, 165, 255), 2)
    return output


def process_video(args: argparse.Namespace) -> None:
    ensure_video(args.video)
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ret, prev_frame = cap.read()
    if not ret:
        sys.exit("Could not read the first frame from the video.")

    prev_frame = resize_frame(prev_frame, args.resize)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    roi = args.roi
    if roi is not None:
        roi = validate_roi(roi, prev_gray.shape)

    frame_h, frame_w = prev_frame.shape[:2]
    writer_size = (frame_w, frame_h)

    hsv_path = os.path.join(args.out_dir, "flow_hsv.mp4")
    quiver_path = os.path.join(args.out_dir, "flow_quiver.mp4")
    hsv_writer = quiver_writer = None
    if args.viz in {"hsv", "both"}:
        hsv_writer, hsv_codec = create_writer(hsv_path, fps, writer_size)
        print(f"HSV video writer initialized with codec {hsv_codec} -> {hsv_path}")
    if args.viz in {"quiver", "both"}:
        quiver_writer, quiver_codec = create_writer(quiver_path, fps, writer_size)
        print(f"Quiver video writer initialized with codec {quiver_codec} -> {quiver_path}")

    total_mag = 0.0
    total_count = 0
    global_max = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_frame(frame, args.resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=args.pyr_scale,
                levels=args.levels,
                winsize=args.winsize,
                iterations=args.iterations,
                poly_n=args.poly_n,
                poly_sigma=args.poly_sigma,
                flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        else:
            x1, y1, x2, y2 = roi
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray[y1:y2, x1:x2],
                gray[y1:y2, x1:x2],
                None,
                pyr_scale=args.pyr_scale,
                levels=args.levels,
                winsize=args.winsize,
                iterations=args.iterations,
                poly_n=args.poly_n,
                poly_sigma=args.poly_sigma,
                flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        frame_idx += 1
        mean_mag = float(np.mean(mag))
        max_mag = float(np.max(mag))
        total_mag += float(np.sum(mag))
        total_count += mag.size
        global_max = max(global_max, max_mag)

        if frame_idx % args.print_every == 0:
            print(f"Frame {frame_idx}: mean_mag={mean_mag:.4f}, max_mag={max_mag:.4f}")

        if args.viz in {"hsv", "both"} and hsv_writer is not None:
            flow_for_hsv = flow if roi is None else pad_flow(flow, frame.shape, roi)
            hsv_frame = flow_to_hsv(flow_for_hsv, frame.shape, roi)
            hsv_writer.write(hsv_frame)

        if args.viz in {"quiver", "both"} and quiver_writer is not None:
            flow_for_quiver = flow if roi is None else flow
            quiver_frame = draw_quiver(frame, flow_for_quiver, args.quiver_step, roi)
            quiver_writer.write(quiver_frame)

        prev_gray = gray

    cap.release()
    if hsv_writer is not None:
        hsv_writer.release()
    if quiver_writer is not None:
        quiver_writer.release()

    if total_count == 0:
        print("No frames processed.")
        return

    overall_mean = total_mag / total_count
    print("---- Summary ----")
    print(f"Frames processed: {frame_idx}")
    print(f"Overall mean magnitude: {overall_mean:.4f}")
    print(f"Global max magnitude: {global_max:.4f}")


def pad_flow(flow: np.ndarray, frame_shape: Tuple[int, int, int], roi: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame_shape[:2]
    padded = np.zeros((h, w, 2), dtype=flow.dtype)
    x1, y1, x2, y2 = roi
    padded[y1:y2, x1:x2] = flow
    return padded


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
