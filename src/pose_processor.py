"""
SnapGuard AI - Pose Processor Module
Uses MediaPipe Pose to extract body landmarks and calculate spatial body metrics.
"""

import math
import cv2
import numpy as np
import mediapipe as mp


class PoseProcessor:
    """
    Processes video frames using MediaPipe Pose to detect body landmarks
    and compute structural orientation metrics (torso angle, hip height, etc.).
    """

    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the MediaPipe Pose detector.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Landmark indexes of interest
        self.KEY_LANDMARKS = {
            "NOSE": self.mp_pose.PoseLandmark.NOSE,
            "LEFT_SHOULDER": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            "RIGHT_SHOULDER": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "LEFT_HIP": self.mp_pose.PoseLandmark.LEFT_HIP,
            "RIGHT_HIP": self.mp_pose.PoseLandmark.RIGHT_HIP,
            "LEFT_KNEE": self.mp_pose.PoseLandmark.LEFT_KNEE,
            "RIGHT_KNEE": self.mp_pose.PoseLandmark.RIGHT_KNEE,
            "LEFT_ANKLE": self.mp_pose.PoseLandmark.LEFT_ANKLE,
            "RIGHT_ANKLE": self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }

    def process_frame(self, frame: np.ndarray):
        """
        Processes a single BGR image frame.
        
        Returns:
            tuple: (processed_results, landmarks_dict, metrics_dict)
        """
        # Convert BGR image to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return results, None, None

        # Extract normalized landmarks
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        landmarks_dict = {}
        for name, landmark_enum in self.KEY_LANDMARKS.items():
            lm = landmarks[landmark_enum.value]
            landmarks_dict[name] = {
                "x_norm": lm.x,
                "y_norm": lm.y,
                "z_norm": lm.z,
                "visibility": lm.visibility,
                "pixel_x": int(lm.x * w),
                "pixel_y": int(lm.y * h)
            }

        metrics = self._calculate_metrics(landmarks_dict, h, w)
        return results, landmarks_dict, metrics

    def _calculate_metrics(self, landmarks_dict: dict, frame_h: int, frame_w: int) -> dict:
        """
        Calculates key pose metrics such as torso inclination angle, hip vertical position,
        and bounding dimensions.
        """
        left_sh = landmarks_dict["LEFT_SHOULDER"]
        right_sh = landmarks_dict["RIGHT_SHOULDER"]
        left_hip = landmarks_dict["LEFT_HIP"]
        right_hip = landmarks_dict["RIGHT_HIP"]

        # Calculate midpoints
        mid_shoulder_x = (left_sh["x_norm"] + right_sh["x_norm"]) / 2.0
        mid_shoulder_y = (left_sh["y_norm"] + right_sh["y_norm"]) / 2.0

        mid_hip_x = (left_hip["x_norm"] + right_hip["x_norm"]) / 2.0
        mid_hip_y = (left_hip["y_norm"] + right_hip["y_norm"]) / 2.0

        # Torso vector from mid-hip to mid-shoulder
        dx = mid_shoulder_x - mid_hip_x
        # Image coordinates: y increases downwards, so shoulder y is smaller than hip y when standing
        dy = mid_hip_y - mid_shoulder_y  # positive when standing upright

        # Torso angle relative to horizontal ground (0 deg = horizontal lying down, 90 deg = upright vertical)
        angle_rad = math.atan2(abs(dy), abs(dx))
        torso_angle_deg = math.degrees(angle_rad)

        # Calculate average visibility of core torso landmarks
        core_visibility = (left_sh["visibility"] + right_sh["visibility"] +
                           left_hip["visibility"] + right_hip["visibility"]) / 4.0

        # Estimate bounding box height/width ratio
        all_x = [lm["x_norm"] for lm in landmarks_dict.values()]
        all_y = [lm["y_norm"] for lm in landmarks_dict.values()]
        bbox_w = max(all_x) - min(all_x)
        bbox_h = max(all_y) - min(all_y)
        aspect_ratio = bbox_w / (bbox_h + 1e-6)

        return {
            "mid_shoulder": (mid_shoulder_x, mid_shoulder_y),
            "mid_hip": (mid_hip_x, mid_hip_y),
            "torso_angle_deg": torso_angle_deg,
            "hip_y": mid_hip_y,
            "shoulder_y": mid_shoulder_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "aspect_ratio": aspect_ratio,
            "visibility": core_visibility
        }

    def draw_landmarks(self, frame: np.ndarray, results, is_warning=False):
        """
        Draws pose landmarks and connection skeleton on the frame with custom styling.
        """
        if not results or not results.pose_landmarks:
            return frame

        connection_color = (0, 0, 255) if is_warning else (0, 255, 128)
        joint_color = (0, 255, 255) if is_warning else (255, 255, 0)

        # Custom landmark drawing style
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=joint_color, thickness=3, circle_radius=4
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=connection_color, thickness=2, circle_radius=2
            )
        )
        return frame

    def close(self):
        """
        Release MediaPipe resources.
        """
        self.pose.close()
