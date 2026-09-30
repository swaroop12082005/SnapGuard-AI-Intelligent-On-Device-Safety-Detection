#!/usr/bin/env python3
"""
SnapGuard AI - Intelligent On-Device Safety Detection
Main Application Entry Point.
Runs real-time computer vision safety monitoring on webcam feeds.
"""

import argparse
import sys
import time
import cv2
import numpy as np

from src.pose_processor import PoseProcessor
from src.fall_detector import FallDetector
from src.alert import AlertManager


def create_synthetic_frame(frame_num: int, width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Generates a synthetic camera frame with an animated human figure to test SnapGuard AI
    in environments without an active camera attached.
    
    Animation Cycle (180 frames @ 30 FPS = 6 sec cycle):
      - Frames 0-60: Standing upright pose (SAFE)
      - Frames 61-80: Rapid downward drop & tilt (POSSIBLE FALL)
      - Frames 81-140: Horizontal low position on floor (FALL DETECTED)
      - Frames 141-180: Standing up back upright (RECOVERY)
    """
    frame = np.full((height, width, 3), (30, 35, 45), dtype=np.uint8)
    
    # Grid pattern background
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (40, 45, 55), 1)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (40, 45, 55), 1)

    cycle_frame = frame_num % 180
    
    center_x = width // 2
    
    if cycle_frame <= 60:
        # Standing upright
        head_y = 200
        shoulder_y = 260
        hip_y = 420
        ankle_y = 600
        angle_tilt = 0.0
    elif cycle_frame <= 80:
        # Rapid fall transition
        progress = (cycle_frame - 60) / 20.0
        head_y = int(200 + progress * 300)
        shoulder_y = int(260 + progress * 260)
        hip_y = int(420 + progress * 150)
        ankle_y = int(600 + progress * 20)
        angle_tilt = progress * 75.0  # degrees horizontal
    elif cycle_frame <= 140:
        # Fallen horizontal on ground
        head_y = 520
        shoulder_y = 530
        hip_y = 550
        ankle_y = 560
        angle_tilt = 80.0
    else:
        # Standing up recovery
        progress = (180 - cycle_frame) / 40.0
        head_y = int(200 + progress * 320)
        shoulder_y = int(260 + progress * 270)
        hip_y = int(420 + progress * 130)
        ankle_y = 600
        angle_tilt = progress * 80.0

    # Draw synthetic figure sticks/joints for MediaPipe detection compatibility
    cos_t = np.cos(np.radians(angle_tilt))
    sin_t = np.sin(np.radians(angle_tilt))
    
    def transform_pt(x_off, y_base):
        dx = x_off * cos_t - (y_base - 300) * sin_t
        dy = x_off * sin_t + (y_base - 300) * cos_t
        return int(center_x + dx), int(300 + dy)

    head_pt = transform_pt(0, head_y)
    sh_l = transform_pt(-40, shoulder_y)
    sh_r = transform_pt(40, shoulder_y)
    hip_l = transform_pt(-35, hip_y)
    hip_r = transform_pt(35, hip_y)
    ankle_l = transform_pt(-30, ankle_y)
    ankle_r = transform_pt(30, ankle_y)

    # Draw head & torso
    cv2.circle(frame, head_pt, 25, (220, 220, 220), -1)
    cv2.line(frame, sh_l, sh_r, (200, 200, 200), 8)
    cv2.line(frame, hip_l, hip_r, (200, 200, 200), 8)
    cv2.line(frame, ((sh_l[0]+sh_r[0])//2, (sh_l[1]+sh_r[1])//2),
             ((hip_l[0]+hip_r[0])//2, (hip_l[1]+hip_r[1])//2), (200, 200, 200), 10)
    cv2.line(frame, sh_l, ankle_l, (180, 180, 180), 6)
    cv2.line(frame, sh_r, ankle_r, (180, 180, 180), 6)

    # Synthetic watermark indicator
    cv2.putText(frame, "SIMULATED WEBCAM TEST FEED", (width - 340, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 140, 160), 1)

    return frame


def run_main(args):
    print("=" * 60)
    print("  SNAPGUARD AI - Intelligent On-Device Safety Detection")
    print("  Local Edge AI Computer Vision Fall Detector")
    print("=" * 60)

    # Initialize Modules
    pose_processor = PoseProcessor(
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence
    )
    fall_detector = FallDetector(
        confirmation_frames=args.confirmation_frames,
        angle_threshold_deg=args.angle_threshold
    )
    alert_manager = AlertManager(
        cooldown_seconds=args.cooldown,
        enable_sound=not args.no_sound
    )

    # Open Camera Capture
    cap = None
    use_synthetic = args.synthetic

    if not use_synthetic:
        print(f"[INFO] Opening Webcam Camera ID: {args.camera_id}...")
        cap = cv2.VideoCapture(args.camera_id)
        if args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if not cap.isOpened():
            print(f"[WARNING] Camera ID {args.camera_id} could not be opened.")
            print("[INFO] Falling back to synthetic simulation video feed.")
            use_synthetic = True

    if use_synthetic:
        print("[INFO] Running in Synthetic Simulation Mode.")

    print("\n[CONTROLS]")
    print("  'q' or ESC - Exit Application")
    print("  'r'        - Reset Fall Detector & Alerts")
    print("  'd'        - Toggle Telemetry Debug Overlay")
    print("  's'        - Force Fall Test Trigger (Synthetic Mode)\n")

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    debug_mode = False

    try:
        while True:
            loop_start = time.time()

            if use_synthetic:
                frame = create_synthetic_frame(frame_count)
                time.sleep(0.03)  # ~30 FPS throttle
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARNING] Failed to capture frame from webcam. Retrying...")
                    time.sleep(0.1)
                    continue
                # Horizontal flip for intuitive mirror webcam experience
                frame = cv2.flip(frame, 1)

            # Step 1: Process Pose Landmarks
            results, landmarks_dict, metrics = pose_processor.process_frame(frame)

            # Step 2: Fall Detection State Machine
            detection_result = fall_detector.process_metrics(metrics)

            # Step 3: Emergency Alert Handler
            alert_manager.handle_detection(detection_result)

            # Step 4: Render Skeleton Overlay
            is_warning = detection_result.status in ("POSSIBLE_FALL", "FALL_DETECTED")
            frame = pose_processor.draw_landmarks(frame, results, is_warning=is_warning)

            # Step 5: Render Telemetry HUD
            frame = alert_manager.render_hud(frame, detection_result, fps, debug_mode=debug_mode)

            # Display Frame in OpenCV Window
            window_title = "SnapGuard AI - Safety Monitoring"
            cv2.imshow(window_title, frame)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                print("[INFO] Exiting SnapGuard AI...")
                break
            elif key == ord('r'):
                fall_detector.reset()
                print("[INFO] Fall detector state reset.")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug overlay set to {debug_mode}")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        if cap and cap.isOpened():
            cap.release()
        pose_processor.close()
        cv2.destroyAllWindows()
        print("[INFO] SnapGuard AI shutdown complete.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="SnapGuard AI - Intelligent On-Device Safety Detection")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device camera ID index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Webcam frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Webcam frame height (default: 720)")
    parser.add_argument("--confirmation-frames", type=int, default=5, help="Consecutive abnormal frames to confirm fall (default: 5)")
    parser.add_argument("--angle-threshold", type=float, default=40.0, help="Torso horizontal angle threshold in degrees (default: 40.0)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Alert audio cooldown in seconds (default: 5.0)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="MediaPipe Pose min detection confidence (default: 0.5)")
    parser.add_argument("--synthetic", action="store_true", help="Run with simulated synthetic webcam feed for headless/CI testing")
    parser.add_argument("--no-sound", action="store_true", help="Disable alert audio sound playback")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_main(args)
