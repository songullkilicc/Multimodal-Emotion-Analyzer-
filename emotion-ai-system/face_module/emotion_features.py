# face_module/emotion_features.py
# Yüz landmarklarından duygu özelliklerini hesaplar.
# EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), kaş mesafesi, baş eğimi.

import numpy as np
import time
from config import EAR_THRESHOLD, BLINK_CONSEC_FRAMES


class EmotionFeatureExtractor:
    def __init__(self):
        # Blink sayacı
        self.blink_counter = 0
        self.blink_total = 0
        self.consec_frames = 0
        self.start_time = time.time()

        # MediaPipe landmark indeksleri
        # Sol göz
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        # Sağ göz
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Ağız
        self.MOUTH = [61, 291, 13, 14, 17, 0, 78, 308]
        # Kaşlar
        self.LEFT_EYEBROW = [276, 283, 282, 295, 285]
        self.RIGHT_EYEBROW = [46, 53, 52, 65, 55]
        # Burun ucu (baş eğimi için)
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EAR_POINT = 234
        self.RIGHT_EAR_POINT = 454

        print("[EmotionFeatureExtractor] Başlatıldı.")

    def extract(self, landmarks, frame) -> dict:
        """
        Landmarks'tan tüm duygu özelliklerini çıkarır.
        Döner: dict
        """
        h, w = frame.shape[:2]

        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE, w, h)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Blink tespiti
        blink_rate = self._detect_blink(avg_ear)

        # Ağız açıklığı
        mar = self._mouth_aspect_ratio(landmarks, w, h)

        # Kaş mesafesi (stres göstergesi)
        eyebrow_distance = self._eyebrow_distance(landmarks, w, h)

        # Baş eğimi
        head_tilt = self._head_tilt(landmarks, w, h)

        # Gülümseme skoru (0-1)
        smile_score = self._smile_score(landmarks, w, h)

        return {
            "ear": round(avg_ear, 3),
            "blink_rate": round(blink_rate, 3),
            "mouth_open": round(mar, 3),
            "eyebrow_distance": round(eyebrow_distance, 3),
            "head_tilt": round(head_tilt, 2),
            "smile_score": round(smile_score, 3)
        }

    def _get_point(self, landmarks, index, w, h) -> np.ndarray:
        lm = landmarks.landmark[index]
        return np.array([lm.x * w, lm.y * h])

    def _eye_aspect_ratio(self, landmarks, eye_indices, w, h) -> float:
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        Göz kapandıkça EAR düşer.
        """
        points = [self._get_point(landmarks, i, w, h) for i in eye_indices]

        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])

        if C == 0:
            return 0.0

        ear = (A + B) / (2.0 * C)
        return ear

    def _detect_blink(self, ear: float) -> float:
        """
        EAR threshold altına düştüğünde blink sayar.
        Döner: blink/saniye oranı
        """
        if ear < EAR_THRESHOLD:
            self.consec_frames += 1
        else:
            if self.consec_frames >= BLINK_CONSEC_FRAMES:
                self.blink_total += 1
            self.consec_frames = 0

        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.blink_total / elapsed
        return 0.0

    def _mouth_aspect_ratio(self, landmarks, w, h) -> float:
        """
        MAR: ağız dikey açıklığı / yatay genişliği
        """
        top_lip = self._get_point(landmarks, 13, w, h)
        bottom_lip = self._get_point(landmarks, 14, w, h)
        left_mouth = self._get_point(landmarks, 61, w, h)
        right_mouth = self._get_point(landmarks, 291, w, h)

        vertical = np.linalg.norm(top_lip - bottom_lip)
        horizontal = np.linalg.norm(left_mouth - right_mouth)

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    def _eyebrow_distance(self, landmarks, w, h) -> float:
        """
        İki kaşın birbirine olan mesafesi (normalleştirilmiş).
        Kaşlar çatıldığında mesafe düşer → stres sinyali.
        """
        left_brow = self._get_point(landmarks, self.LEFT_EYEBROW[2], w, h)
        right_brow = self._get_point(landmarks, self.RIGHT_EYEBROW[2], w, h)

        distance = np.linalg.norm(left_brow - right_brow)
        # Yüz genişliğine göre normalize et
        face_width = w
        return distance / face_width

    def _head_tilt(self, landmarks, w, h) -> float:
        """
        Sol kulak - sağ kulak noktasından baş eğim açısı (derece).
        """
        left_point = self._get_point(landmarks, self.LEFT_EAR_POINT, w, h)
        right_point = self._get_point(landmarks, self.RIGHT_EAR_POINT, w, h)

        delta_y = right_point[1] - left_point[1]
        delta_x = right_point[0] - left_point[0]

        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle

    def _smile_score(self, landmarks, w, h) -> float:
        """
        Ağız köşelerinin burun ucuna göre yüksekliği → gülümseme tahmini.
        0 = gülümseme yok, 1 = güçlü gülümseme
        """
        nose_tip = self._get_point(landmarks, self.NOSE_TIP, w, h)
        left_mouth = self._get_point(landmarks, 61, w, h)
        right_mouth = self._get_point(landmarks, 291, w, h)

        left_lift = nose_tip[1] - left_mouth[1]
        right_lift = nose_tip[1] - right_mouth[1]

        avg_lift = (left_lift + right_lift) / 2.0
        # Normalize (0-1 arasına sıkıştır)
        score = np.clip(avg_lift / 50.0, 0, 1)
        return float(score)