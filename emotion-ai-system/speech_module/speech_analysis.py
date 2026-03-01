# speech_module/speech_analysis.py
# Mikrofondan ses kaydeder, Whisper ile metne çevirir,
# sentiment analizi yapar.

import numpy as np
import wave
import tempfile
import os
import time
import whisper
from config import RECORD_SECONDS, SAMPLE_RATE, WHISPER_MODEL

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[SpeechAnalyzer] pyaudio bulunamadı. Ses analizi devre dışı.")


class SpeechAnalyzer:
    def __init__(self):
        self.whisper_model = None
        self.audio = None

        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                print("[SpeechAnalyzer] Mikrofon hazır.")
            except Exception as e:
                print(f"[SpeechAnalyzer] Mikrofon başlatılamadı: {e}")

        # Whisper modeli yükle (ilk seferinde indirir)
        print(f"[SpeechAnalyzer] Whisper '{WHISPER_MODEL}' modeli yükleniyor...")
        try:
            self.whisper_model = whisper.load_model(WHISPER_MODEL)
            print("[SpeechAnalyzer] Whisper hazır.")
        except Exception as e:
            print(f"[SpeechAnalyzer] Whisper yüklenemedi: {e}")

        # Basit sentiment kelime listeleri
        self.positive_words = {
            "happy", "great", "good", "love", "excellent", "wonderful",
            "amazing", "fantastic", "joy", "excited", "calm", "relaxed",
            "mutlu", "iyi", "güzel", "harika", "seviyorum", "mükemmel"
        }
        self.negative_words = {
            "sad", "bad", "hate", "terrible", "awful", "angry", "fear",
            "stress", "anxiety", "nervous", "tired", "scared", "worried",
            "üzgün", "kötü", "korku", "stres", "sinirli", "yorgun", "endişe"
        }

        print("[SpeechAnalyzer] Başlatıldı.")

    def analyze(self) -> dict:
        """
        Ses kaydeder → metne çevirir → sentiment analizi yapar.
        Döner: {"transcript": str, "sentiment": str, "confidence": float, "word_count": int}
        """
        default = {
            "transcript": "",
            "sentiment": "neutral",
            "confidence": 0.0,
            "word_count": 0
        }

        if not PYAUDIO_AVAILABLE or self.audio is None:
            return default

        # Ses kaydet
        audio_path = self._record_audio()
        if audio_path is None:
            return default

        # Whisper ile metne çevir
        transcript = self._transcribe(audio_path)

        # Geçici dosyayı sil
        try:
            os.remove(audio_path)
        except:
            pass

        if not transcript:
            return default

        # Sentiment analizi
        sentiment, confidence = self._analyze_sentiment(transcript)

        return {
            "transcript": transcript,
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "word_count": len(transcript.split())
        }

    def _record_audio(self) -> str:
        """
        Mikrofondan ses kaydeder, geçici .wav dosyasına yazar.
        Döner: dosya yolu
        """
        try:
            chunk = 1024
            audio_format = pyaudio.paInt16
            channels = 1

            print(f"[SpeechAnalyzer] {RECORD_SECONDS} saniyelik ses kaydediliyor...")

            stream = self.audio.open(
                format=audio_format,
                channels=channels,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=chunk
            )

            frames = []
            for _ in range(0, int(SAMPLE_RATE / chunk * RECORD_SECONDS)):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # Geçici dosyaya kaydet
            tmp_file = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            tmp_path = tmp_file.name
            tmp_file.close()

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(self.audio.get_sample_size(audio_format))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))

            print(f"[SpeechAnalyzer] Kayıt tamamlandı.")
            return tmp_path

        except Exception as e:
            print(f"[SpeechAnalyzer] Kayıt hatası: {e}")
            return None

    def _transcribe(self, audio_path: str) -> str:
        """
        Whisper ile ses dosyasını metne çevirir.
        """
        if self.whisper_model is None:
            return ""

        try:
            print("[SpeechAnalyzer] Transkripsiyon yapılıyor...")
            result = self.whisper_model.transcribe(
                audio_path,
                language=None,       # otomatik dil tespiti
                fp16=False
            )
            transcript = result.get("text", "").strip()
            print(f"[SpeechAnalyzer] Transkript: '{transcript}'")
            return transcript

        except Exception as e:
            print(f"[SpeechAnalyzer] Transkripsiyon hatası: {e}")
            return ""

    def _analyze_sentiment(self, text: str):
        """
        Basit kelime bazlı sentiment analizi.
        Döner: (sentiment_str, confidence_float)
        """
        words = set(text.lower().split())

        positive_hits = len(words & self.positive_words)
        negative_hits = len(words & self.negative_words)
        total_hits = positive_hits + negative_hits

        if total_hits == 0:
            return "neutral", 0.5

        if positive_hits > negative_hits:
            confidence = positive_hits / total_hits
            return "positive", confidence
        elif negative_hits > positive_hits:
            confidence = negative_hits / total_hits
            return "negative", confidence
        else:
            return "neutral", 0.5

    def __del__(self):
        if self.audio:
            self.audio.terminate()