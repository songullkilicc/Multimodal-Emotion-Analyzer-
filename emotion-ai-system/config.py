# config.py
# Tüm proje ayarları burada. Bir şeyi değiştirmek istersen buraya bakarsın.

import os

# --- GENEL AYARLAR ---
PROJECT_NAME = "Multimodal Emotion Analyzer"
VERSION = "1.0.0"

# --- KAMERA ---
CAMERA_INDEX = 0          # 0 = dahili webcam, 1 = harici kamera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# --- ARDUINO ---
ARDUINO_PORT = "COM3"     # Windows için COM3/COM4, Linux için /dev/ttyACM0
ARDUINO_BAUDRATE = 9600
ARDUINO_TIMEOUT = 1       # saniye

# --- GEMINI API ---
GEMINI_API_KEY = "AIzaSyARMhineq9LkxqJmsPgzJDKX-KKV7DZZ7U"
AI_MODEL = "gemini-2.0-flash"
MAX_TOKENS = 300

# --- SPEECH ---
RECORD_SECONDS = 5        # kaç saniyelik ses kaydı alınsın
SAMPLE_RATE = 16000
WHISPER_MODEL = "base"    # tiny, base, small, medium, large

# --- BLINK DETECTION ---
EAR_THRESHOLD = 0.25      # Eye Aspect Ratio eşiği (altı = göz kapalı)
BLINK_CONSEC_FRAMES = 2   # kaç frame üst üste kapalı → blink sayılır

# --- DATA FUSION ---
ANALYSIS_INTERVAL = 10    # her kaç saniyede bir AI'a gönderilsin

# --- LOGGING ---
LOG_LEVEL = "INFO"
SAVE_RESULTS = True
RESULTS_PATH = "results/"