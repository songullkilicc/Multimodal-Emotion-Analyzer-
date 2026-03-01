# utils/data_fusion.py
# Face + Arduino + Speech verilerini tek bir yapıda birleştirir.
# AI modülüne gitmeden önceki son durak.

import time
from datetime import datetime


class DataFusion:
    def __init__(self):
        self.history = []
        self.max_history = 10
        print("[DataFusion] Initialized.")

    def fuse(self, face_data: dict, sensor_data: dict, speech_data: dict) -> dict:
        """
        Combines all module outputs into a single dict.
        """
        fused = {
            "timestamp": datetime.now().isoformat(),
            "face": face_data if face_data else {},
            "sensors": sensor_data if sensor_data else {},
            "speech": speech_data if speech_data else {},
            "quality": self._assess_quality(face_data, sensor_data, speech_data)
        }

        self.history.append(fused)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return fused

    def _assess_quality(self, face: dict, sensors: dict, speech: dict) -> dict:
        """
        Evaluates data quality for each module.
        Returns: {"face": bool, "sensors": bool, "speech": bool, "overall": str}
        """
        face_ok = bool(face and face.get("ear") is not None)
        sensors_ok = bool(sensors and sensors.get("heart_rate", 0) > 0)
        speech_ok = bool(speech and speech.get("transcript"))

        active_sources = sum([face_ok, sensors_ok, speech_ok])

        if active_sources == 3:
            overall = "excellent"
        elif active_sources == 2:
            overall = "good"
        elif active_sources == 1:
            overall = "limited"
        else:
            overall = "no_data"

        return {
            "face_active": face_ok,
            "sensors_active": sensors_ok,
            "speech_active": speech_ok,
            "active_sources": active_sources,
            "overall": overall
        }

    def get_averaged_face_data(self) -> dict:
        """
        Returns the average of face data from the last N frames.
        Smooths out sudden changes.
        """
        face_records = [
            entry["face"] for entry in self.history
            if entry.get("face")
        ]

        if not face_records:
            return {}

        keys = ["ear", "blink_rate", "mouth_open",
                "eyebrow_distance", "head_tilt", "smile_score"]
        averaged = {}

        for key in keys:
            values = [r[key] for r in face_records if key in r]
            if values:
                averaged[key] = round(sum(values) / len(values), 3)

        return averaged

    def get_summary(self) -> dict:
        """
        Returns a summary of the entire history.
        Useful for logs or reports.
        """
        if not self.history:
            return {}

        heart_rates = [
            e["sensors"].get("heart_rate", 0)
            for e in self.history
            if e.get("sensors")
        ]
        sentiments = [
            e["speech"].get("sentiment", "")
            for e in self.history
            if e.get("speech")
        ]

        avg_hr = round(sum(heart_rates) / len(heart_rates), 1) if heart_rates else 0
        dominant_sentiment = max(
            set(sentiments), key=sentiments.count
        ) if sentiments else "unknown"

        return {
            "total_records": len(self.history),
            "avg_heart_rate": avg_hr,
            "dominant_sentiment": dominant_sentiment,
            "first_timestamp": self.history[0]["timestamp"],
            "last_timestamp": self.history[-1]["timestamp"]
        }