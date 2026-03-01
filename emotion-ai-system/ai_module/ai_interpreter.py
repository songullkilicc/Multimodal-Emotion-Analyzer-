# ai_module/ai_interpreter.py

import google.genai as genai
from google.genai import types
from config import GEMINI_API_KEY, AI_MODEL, MAX_TOKENS

class AIInterpreter:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.system_prompt = """
You are an expert emotion analyst. You receive multimodal sensor data from a person:
- Face analysis (eye blink rate, mouth openness, eyebrow distance, head tilt, smile score)
- Biometric data (heart rate, skin temperature, GSR stress level)
- Speech analysis (transcript, sentiment, confidence)

Based on this data, provide a concise emotional state analysis.
Your response must follow this exact format:

EMOTION: [primary emotion in one word]
INTENSITY: [low/medium/high]
CONFIDENCE: [0.0 to 1.0]
ANALYSIS: [2-3 sentence explanation]
RECOMMENDATION: [one actionable suggestion]
"""
        print("[AIInterpreter] Başlatıldı.")

    def interpret(self, fused_data: dict) -> str:
        if not fused_data:
            return "Yeterli veri yok."

        prompt = self.system_prompt + "\n\n" + self._build_prompt(fused_data)

        try:
            print("[AIInterpreter] Gemini'ye gönderiliyor...")
            response = self.client.models.generate_content(
                model=AI_MODEL,
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            print(f"[AIInterpreter] API hatası: {e}")
            return f"AI analizi başarısız: {str(e)}"

    def _build_prompt(self, data: dict) -> str:
        face = data.get("face", {})
        sensors = data.get("sensors", {})
        speech = data.get("speech", {})

        prompt_parts = ["Here is the current sensor data:\n"]

        if face:
            prompt_parts.append("FACE ANALYSIS:")
            prompt_parts.append(f"  - Blink rate: {face.get('blink_rate', 'N/A')} blinks/sec")
            prompt_parts.append(f"  - Mouth openness (MAR): {face.get('mouth_open', 'N/A')}")
            prompt_parts.append(f"  - Eyebrow distance: {face.get('eyebrow_distance', 'N/A')}")
            prompt_parts.append(f"  - Head tilt: {face.get('head_tilt', 'N/A')} degrees")
            prompt_parts.append(f"  - Smile score: {face.get('smile_score', 'N/A')}")
            prompt_parts.append("")

        if sensors:
            prompt_parts.append("BIOMETRIC DATA:")
            prompt_parts.append(f"  - Heart rate: {sensors.get('heart_rate', 'N/A')} BPM")
            prompt_parts.append(f"  - Skin temperature: {sensors.get('temperature', 'N/A')} C")
            if sensors.get("gsr"):
                prompt_parts.append(f"  - GSR (stress): {sensors.get('gsr', 'N/A')}")
            prompt_parts.append("")

        if speech and speech.get("transcript"):
            prompt_parts.append("SPEECH ANALYSIS:")
            prompt_parts.append(f"  - Transcript: \"{speech.get('transcript', '')}\"")
            prompt_parts.append(f"  - Sentiment: {speech.get('sentiment', 'N/A')}")
            prompt_parts.append(f"  - Confidence: {speech.get('confidence', 'N/A')}")
            prompt_parts.append("")

        prompt_parts.append("Please analyze the emotional state of this person.")
        return "\n".join(prompt_parts)