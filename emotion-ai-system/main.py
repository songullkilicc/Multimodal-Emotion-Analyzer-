import time
import random
import json
import os
from datetime import datetime

from config import ANALYSIS_INTERVAL, SAVE_RESULTS, RESULTS_PATH
from ai_module.ai_interpreter import AIInterpreter
from utils.data_fusion import DataFusion

DUMMY_MODE = True


def generate_dummy_face_data() -> dict:
    return {
        "ear": round(random.uniform(0.2, 0.4), 3),
        "blink_rate": round(random.uniform(0.1, 1.2), 3),
        "mouth_open": round(random.uniform(0.0, 0.5), 3),
        "eyebrow_distance": round(random.uniform(0.3, 0.6), 3),
        "head_tilt": round(random.uniform(-15, 15), 2),
        "smile_score": round(random.uniform(0.0, 1.0), 3)
    }


def generate_dummy_sensor_data() -> dict:
    return {
        "heart_rate": random.randint(60, 110),
        "temperature": round(random.uniform(36.0, 37.5), 1),
        "gsr": random.randint(400, 700)
    }


def generate_dummy_speech_data() -> dict:
    samples = [
        ("I feel a bit tired and stressed today.", "negative", 0.75),
        ("Everything is going great, I am happy!", "positive", 0.88),
        ("I am not sure how I feel right now.", "neutral", 0.5),
        ("This is really frustrating and annoying.", "negative", 0.82),
        ("I am calm and focused.", "positive", 0.71),
    ]
    transcript, sentiment, confidence = random.choice(samples)
    return {
        "transcript": transcript,
        "sentiment": sentiment,
        "confidence": confidence,
        "word_count": len(transcript.split())
    }


def save_result(data: dict, result: str):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_PATH, f"result_{timestamp}.json")
    output = {
        "timestamp": timestamp,
        "sensor_data": data,
        "ai_result": result
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Sonuç kaydedildi: {filename}")


def main():
    print("=" * 55)
    print("   Multimodal Emotion Analyzer - DUMMY MODE")
    print("=" * 55)
    print("[INFO] Gerçek donanım yok, sahte veri üretiliyor...\n")

    ai_interpreter = AIInterpreter()
    data_fusion = DataFusion()

    cycle = 1

    try:
        while True:
            print(f"\n--- Cycle {cycle} ---")

            face_data = generate_dummy_face_data()
            sensor_data = generate_dummy_sensor_data()
            speech_data = generate_dummy_speech_data()

            print(f"[FACE]    blink_rate={face_data['blink_rate']}  smile={face_data['smile_score']}  tilt={face_data['head_tilt']}")
            print(f"[SENSORS] heart_rate={sensor_data['heart_rate']} BPM  temp={sensor_data['temperature']}C  gsr={sensor_data['gsr']}")
            print(f"[SPEECH]  \"{speech_data['transcript']}\"")
            print(f"          sentiment={speech_data['sentiment']}  confidence={speech_data['confidence']}")

            fused_data = data_fusion.fuse(face_data, sensor_data, speech_data)

            print("\n[AI] Analiz yapılıyor...")
            result = ai_interpreter.interpret(fused_data)

            print("\n" + "=" * 55)
            print("EMOTION ANALYSIS RESULT:")
            print(result)
            print("=" * 55)

            if SAVE_RESULTS:
                save_result(fused_data, result)

            cycle += 1
            print(f"\n[INFO] Sonraki analiz {ANALYSIS_INTERVAL} saniye sonra... (Durdurmak icin CTRL+C)")
            time.sleep(ANALYSIS_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n[INFO] Sistem durduruldu. Toplam cycle:", cycle - 1)
        summary = data_fusion.get_summary()
        if summary:
            print("\nSESSION SUMMARY:")
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()