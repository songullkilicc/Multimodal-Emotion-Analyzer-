# arduino_module/sensor_reader.py
# Arduino'dan serial port üzerinden sensör verisi okur.
# Arduino şu format'ta veri gönderir: "HR:85,TEMP:36.7\n"

import serial
import serial.tools.list_ports
import time
from config import ARDUINO_PORT, ARDUINO_BAUDRATE, ARDUINO_TIMEOUT


class ArduinoSensorReader:
    def __init__(self):
        self.serial_conn = None
        self.last_data = {
            "heart_rate": 0,
            "temperature": 0.0,
            "gsr": 0
        }
        print("[ArduinoSensorReader] Başlatıldı.")

    def connect(self) -> bool:
        """
        Arduino'ya bağlanmayı dener.
        Başarılıysa True, değilse False döner.
        """
        try:
            self.serial_conn = serial.Serial(
                port=ARDUINO_PORT,
                baudrate=ARDUINO_BAUDRATE,
                timeout=ARDUINO_TIMEOUT
            )
            time.sleep(2)  # Arduino reset süresi
            print(f"[ArduinoSensorReader] Bağlandı: {ARDUINO_PORT}")
            return True

        except serial.SerialException as e:
            print(f"[ArduinoSensorReader] Bağlantı hatası: {e}")
            print("[ArduinoSensorReader] Mevcut portlar:")
            self._list_ports()
            return False

    def read(self) -> dict:
        """
        Serial port'tan bir satır okur ve parse eder.
        Döner: {"heart_rate": int, "temperature": float, "gsr": int}
        """
        if self.serial_conn is None or not self.serial_conn.is_open:
            return self.last_data

        try:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode("utf-8").strip()
                parsed = self._parse_line(line)
                if parsed:
                    self.last_data = parsed

        except (serial.SerialException, UnicodeDecodeError) as e:
            print(f"[ArduinoSensorReader] Okuma hatası: {e}")

        return self.last_data

    def _parse_line(self, line: str) -> dict:
        """
        "HR:85,TEMP:36.7,GSR:512" formatını parse eder.
        """
        result = {}
        try:
            parts = line.split(",")
            for part in parts:
                if ":" not in part:
                    continue
                key, value = part.split(":")
                key = key.strip().upper()
                value = value.strip()

                if key == "HR":
                    result["heart_rate"] = int(value)
                elif key == "TEMP":
                    result["temperature"] = float(value)
                elif key == "GSR":
                    result["gsr"] = int(value)

            return result if result else None

        except (ValueError, IndexError) as e:
            print(f"[ArduinoSensorReader] Parse hatası '{line}': {e}")
            return None

    def _list_ports(self):
        ports = serial.tools.list_ports.comports()
        if ports:
            for port in ports:
                print(f"  - {port.device}: {port.description}")
        else:
            print("  Hiç port bulunamadı.")

    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[ArduinoSensorReader] Bağlantı kapatıldı.")