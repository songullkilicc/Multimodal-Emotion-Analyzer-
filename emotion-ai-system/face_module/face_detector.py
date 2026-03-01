# face_module/face_detector.py
# MediaPipe ile yüz landmarklarını tespit eder.
# Döndürdüğü landmarks → emotion_features.py'a gider.

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,      # iris noktaları da gelir
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("[FaceDetector] Başlatıldı.")

    def detect(self, frame: np.ndarray) -> Tuple[Optional[object], np.ndarray]:
        """
        Verilen frame'i işler.
        Döner: (landmarks | None, annotated_frame)
        """
        # MediaPipe RGB ister
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.face_mesh.process(rgb_frame)

        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Yüz mesh çiz
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
            )

            # Kontur çiz
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style()
            )

            return face_landmarks, annotated_frame

        return None, annotated_frame

    def get_landmark_coords(self, landmarks, index: int,
                             frame_w: int, frame_h: int) -> Tuple[int, int]:
        """
        Verilen index'teki landmark'ın piksel koordinatlarını döndürür.
        """
        lm = landmarks.landmark[index]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        return x, y

    def __del__(self):
        self.face_mesh.close()