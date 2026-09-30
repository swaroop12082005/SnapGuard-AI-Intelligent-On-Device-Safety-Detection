"""
SnapGuard AI - Fall Detector Module
Implements multi-frame temporal analysis to detect human falls accurately while preventing single-frame false positives.
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Dict, Optional


@dataclass
class DetectionResult:
    status: str              # "SAFE", "POSSIBLE_FALL", or "FALL_DETECTED"
    confidence: float        # Percentage confidence (0.0 - 100.0)
    consecutive_frames: int  # Number of consecutive frames abnormal pose observed
    is_new_fall: bool        # True on the first frame a fall is confirmed
    metrics: Optional[Dict]  # Raw metrics dictionary
    timestamp: str           # Timestamp string (HH:MM:SS)


class FallDetector:
    """
    Stateful fall detector analyzing pose landmark metrics over consecutive video frames.
    """

    # Status Constants
    STATUS_SAFE = "SAFE"
    STATUS_POSSIBLE_FALL = "POSSIBLE_FALL"
    STATUS_FALL_DETECTED = "FALL_DETECTED"

    def __init__(
        self,
        history_size: int = 30,
        angle_threshold_deg: float = 40.0,
        low_hip_threshold: float = 0.60,
        rapid_drop_threshold: float = 0.12,
        confirmation_frames: int = 5
    ):
        """
        Args:
            history_size: Number of past frames stored in sliding history buffer.
            angle_threshold_deg: Torso angle below which body is considered horizontal (deg).
            low_hip_threshold: Normalized Y coordinate (0 top, 1 bottom) indicating low hip position.
            rapid_drop_threshold: Normalized downward displacement threshold over recent frames.
            confirmation_frames: Minimum consecutive frames required to transition to FALL_DETECTED.
        """
        self.history_size = history_size
        self.angle_threshold_deg = angle_threshold_deg
        self.low_hip_threshold = low_hip_threshold
        self.rapid_drop_threshold = rapid_drop_threshold
        self.confirmation_frames = confirmation_frames

        # Sliding window history buffer
        self.history = deque(maxlen=history_size)

        # State tracking variables
        self.current_status = self.STATUS_SAFE
        self.consecutive_abnormal_frames = 0
        self.was_falling = False

    def reset(self):
        """
        Resets detector state and history buffer.
        """
        self.history.clear()
        self.current_status = self.STATUS_SAFE
        self.consecutive_abnormal_frames = 0
        self.was_falling = False

    def process_metrics(self, metrics: Optional[Dict]) -> DetectionResult:
        """
        Processes current frame pose metrics and updates fall state machine.
        
        Returns:
            DetectionResult containing updated status, confidence score, and fall flags.
        """
        timestamp_str = time.strftime("%H:%M:%S")

        # If no pose metrics detected (e.g. subject missing from frame)
        if metrics is None or metrics.get("visibility", 0) < 0.3:
            # Gradually decay abnormal frame counter
            if self.consecutive_abnormal_frames > 0:
                self.consecutive_abnormal_frames -= 1
            
            status = self.STATUS_FALL_DETECTED if (self.consecutive_abnormal_frames >= self.confirmation_frames) else self.STATUS_SAFE
            return DetectionResult(
                status=status,
                confidence=0.0,
                consecutive_frames=self.consecutive_abnormal_frames,
                is_new_fall=False,
                metrics=metrics,
                timestamp=timestamp_str
            )

        # Append current metrics to history
        self.history.append(metrics)

        # Extract current frame parameters
        current_angle = metrics["torso_angle_deg"]
        current_hip_y = metrics["hip_y"]
        aspect_ratio = metrics["aspect_ratio"]

        # Calculate rapid drop (downward movement over history window)
        rapid_drop = self._calculate_downward_drop()

        # Check conditions
        is_horizontal = current_angle < self.angle_threshold_deg
        is_low_position = current_hip_y > self.low_hip_threshold
        is_rapid_drop = rapid_drop > self.rapid_drop_threshold
        is_wide_aspect = aspect_ratio > 1.0

        # Evaluate if frame exhibits fall characteristics
        is_frame_abnormal = (
            (is_horizontal and (is_low_position or is_rapid_drop)) or
            (is_rapid_drop and is_low_position) or
            (is_horizontal and is_wide_aspect)
        )

        # Update consecutive abnormal counter
        if is_frame_abnormal:
            self.consecutive_abnormal_frames += 1
        else:
            if self.consecutive_abnormal_frames > 0:
                self.consecutive_abnormal_frames -= 1

        # Determine state
        if self.consecutive_abnormal_frames >= self.confirmation_frames:
            new_status = self.STATUS_FALL_DETECTED
        elif self.consecutive_abnormal_frames > 0 or is_rapid_drop:
            new_status = self.STATUS_POSSIBLE_FALL
        else:
            new_status = self.STATUS_SAFE

        # Check for new fall event transition
        is_new_fall = (new_status == self.STATUS_FALL_DETECTED and not self.was_falling)
        self.was_falling = (new_status == self.STATUS_FALL_DETECTED)
        self.current_status = new_status

        # Calculate overall detection confidence score (0 - 100%)
        confidence = self._compute_confidence(
            angle=current_angle,
            hip_y=current_hip_y,
            drop=rapid_drop,
            aspect_ratio=aspect_ratio,
            abnormal_frames=self.consecutive_abnormal_frames
        )

        return DetectionResult(
            status=new_status,
            confidence=confidence,
            consecutive_frames=self.consecutive_abnormal_frames,
            is_new_fall=is_new_fall,
            metrics=metrics,
            timestamp=timestamp_str
        )

    def _calculate_downward_drop(self) -> float:
        """
        Calculates maximum downward displacement of hip over the recent frame history.
        In image coordinates, larger Y means lower in the frame.
        """
        if len(self.history) < 3:
            return 0.0

        current_hip_y = self.history[-1]["hip_y"]
        # Look back up to 15 frames (~0.5 seconds)
        lookback = min(15, len(self.history))
        past_hip_ys = [self.history[-i]["hip_y"] for i in range(2, lookback + 1)]
        
        min_past_hip_y = min(past_hip_ys)
        drop = current_hip_y - min_past_hip_y
        return max(0.0, drop)

    def _compute_confidence(
        self,
        angle: float,
        hip_y: float,
        drop: float,
        aspect_ratio: float,
        abnormal_frames: int
    ) -> float:
        """
        Computes a continuous confidence score (0.0 to 100.0) reflecting fall likelihood.
        """
        if abnormal_frames == 0 and angle > 60.0 and hip_y < 0.5:
            return 5.0  # Baseline minimal score when standing upright

        # Component 1: Angle factor (closer to 0 deg horizontal -> higher score)
        angle_score = max(0.0, (90.0 - angle) / 90.0) * 40.0

        # Component 2: Hip depth factor (lower in frame -> higher score)
        hip_score = min(1.0, max(0.0, (hip_y - 0.3) / 0.6)) * 30.0

        # Component 3: Rapid drop factor
        drop_score = min(1.0, drop / 0.20) * 20.0

        # Component 4: Frame persistence multiplier
        persistence_mult = min(1.0, abnormal_frames / float(self.confirmation_frames))

        raw_confidence = (angle_score + hip_score + drop_score) * (0.5 + 0.5 * persistence_mult)
        return round(float(np.clip(raw_confidence, 0.0, 99.9)), 1)
