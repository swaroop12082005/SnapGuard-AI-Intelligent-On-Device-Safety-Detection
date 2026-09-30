"""
SnapGuard AI - Alert & HUD Rendering Module
Handles visual telemetry HUD overlays on video frames and triggers audio alerts with cooldown management.
"""

import os
import sys
import time
import threading
import subprocess
import cv2
import numpy as np


class AlertManager:
    """
    Manages HUD graphics rendering, alert audio notifications, and alert cooldowns.
    """

    def __init__(self, cooldown_seconds: float = 5.0, enable_sound: bool = True):
        """
        Args:
            cooldown_seconds: Cooldown time in seconds between consecutive sound alerts.
            enable_sound: Whether audio alerts are enabled.
        """
        self.cooldown_seconds = cooldown_seconds
        self.enable_sound = enable_sound
        self.last_alert_time = 0.0
        self.last_fall_time_str = "N/A"

    def handle_detection(self, result):
        """
        Triggers emergency alert actions if a new fall event is detected and cooldown expired.
        """
        if result.status == "FALL_DETECTED":
            self.last_fall_time_str = result.timestamp

            now = time.time()
            if (now - self.last_alert_time) >= self.cooldown_seconds:
                self.last_alert_time = now
                if self.enable_sound:
                    self._play_alert_sound_async()

    def _play_alert_sound_async(self):
        """
        Spawns a background thread to play an alert audio sound without blocking video rendering.
        """
        thread = threading.Thread(target=self._play_sound_task, daemon=True)
        thread.start()

    def _play_sound_task(self):
        """
        Executes cross-platform local system alert audio playback.
        """
        try:
            if sys.platform == "darwin":
                # macOS sound playback
                sound_file = "/System/Library/Sounds/Glass.aiff"
                if os.path.exists(sound_file):
                    subprocess.run(["afplay", sound_file], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(["say", "Warning! Fall detected!"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform.startswith("linux"):
                # Linux fallback bell or paplay
                os.system("echo -e '\a'")
            elif sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 500)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            # Fallback bell
            sys.stdout.write("\a")
            sys.stdout.flush()

    def render_hud(self, frame: np.ndarray, result, fps: float, debug_mode: bool = False) -> np.ndarray:
        """
        Renders the SnapGuard AI telemetry HUD directly on the OpenCV image frame.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()

        # Modern Dark Glass HUD Header Bar
        cv2.rectangle(overlay, (0, 0), (w, 70), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Header Title: SNAPGUARD AI
        cv2.putText(frame, "SNAPGUARD AI", (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Processing Pill Badge: LOCAL / ON-DEVICE
        badge_text = "PROCESSING: LOCAL / ON-DEVICE"
        badge_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_x = w - badge_size[0] - 30
        badge_y = 42
        cv2.rectangle(frame, (badge_x - 10, badge_y - 20), (badge_x + badge_size[0] + 10, badge_y + 8), (45, 55, 72), -1)
        cv2.rectangle(frame, (badge_x - 10, badge_y - 20), (badge_x + badge_size[0] + 10, badge_y + 8), (0, 215, 255), 1)
        cv2.putText(frame, badge_text, (badge_x, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        # Telemetry Card (FPS & Confidence)
        telemetry_y = 100
        cv2.rectangle(overlay, (20, telemetry_y), (260, telemetry_y + 90), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (20, telemetry_y), (260, telemetry_y + 90), (60, 70, 90), 1)

        fps_text = f"FPS: {fps:.1f}"
        conf_text = f"Confidence: {result.confidence:.1f}%"
        cv2.putText(frame, fps_text, (35, telemetry_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (35, telemetry_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)

        # Status Display Banner
        if result.status == "SAFE":
            # Emerald Green Safe Box
            box_color = (40, 160, 60)
            text_color = (255, 255, 255)
            status_str = "Status: SAFE"
            
            box_y = h - 75
            cv2.rectangle(frame, (20, box_y), (300, box_y + 50), box_color, -1)
            cv2.rectangle(frame, (20, box_y), (300, box_y + 50), (100, 255, 120), 2)
            cv2.putText(frame, status_str, (35, box_y + 33), cv2.FONT_HERSHEY_DUPLEX, 0.75, text_color, 2, cv2.LINE_AA)

        elif result.status == "POSSIBLE_FALL":
            # Amber Warning Box
            box_color = (0, 140, 255)
            status_str = "WARNING: POSSIBLE FALL DETECTED"
            
            box_y = h - 85
            cv2.rectangle(frame, (20, box_y), (520, box_y + 55), box_color, -1)
            cv2.rectangle(frame, (20, box_y), (520, box_y + 55), (0, 220, 255), 2)
            cv2.putText(frame, status_str, (35, box_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        elif result.status == "FALL_DETECTED":
            # Large Emergency Red Warning Box
            # Pulsating border effect
            pulse = int((time.time() * 4) % 2 * 255)
            border_color = (0, 0, max(180, pulse))

            # Full Screen Red Vignette Alert Edge
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

            banner_y = h - 150
            banner_w = min(620, w - 40)
            
            # Dark Red Background Banner
            cv2.rectangle(frame, (20, banner_y), (20 + banner_w, banner_y + 125), (15, 15, 180), -1)
            cv2.rectangle(frame, (20, banner_y), (20 + banner_w, banner_y + 125), (50, 50, 255), 3)

            # Prominent Emergency Text
            cv2.putText(frame, "WARNING: FALL DETECTED", (35, banner_y + 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Possible emergency detected.", (35, banner_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 255), 1, cv2.LINE_AA)
            
            time_display = self.last_fall_time_str if self.last_fall_time_str != "N/A" else result.timestamp
            cv2.putText(frame, f"Time: {time_display}", (35, banner_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Optional Debug Overlay
        if debug_mode and result.metrics:
            m = result.metrics
            debug_y = 210
            cv2.rectangle(overlay, (20, debug_y), (280, debug_y + 110), (10, 10, 15), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            cv2.putText(frame, f"Torso Angle: {m['torso_angle_deg']:.1f} deg", (30, debug_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, f"Hip Y-Pos: {m['hip_y']:.2f}", (30, debug_y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, f"Aspect Ratio: {m['aspect_ratio']:.2f}", (30, debug_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, f"Abnormal Frames: {result.consecutive_frames}", (30, debug_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        return frame
