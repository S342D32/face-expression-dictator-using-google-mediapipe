import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "face-esp32-project with apk support", "main.py")

P1 = '''# ============================================================================
# Face Expression AI System — APK edition
#
# APK mode: PC webcam + cv2.imshow DISPLAY LOOP removed.
#   - Camera frames arrive from the Expo phone app over WebSocket.
#   - STT is driven by the phone mic -> server Whisper.
#   - ESP32-S3 still publishes distance + LED commands.
# ============================================================================

import os
import asyncio
import logging
import threading
import time

# APK: cv2 optional — headless boot works without it
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

from camera.face_detector import FaceDetector
from processing.expression import analyze_face

from agent.graph import agent

from audio.stt import STT
from audio.tts import TTS
from server.websocket import run_server, distance_state, face_state as ws_face_state

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
'''
with open(OUT, "w", encoding="utf-8") as f:
    f.write(P1)
print("P1 OK", len(P1))
