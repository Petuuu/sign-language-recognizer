"""Minimal browser-based sign language recognizer"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker
from src.image_pipeline import (
    BASE_OPTIONS,
    LABEL_TO_LETTER,
    normalize_landmarks,
)
from src.model.architecture import MLP, load_model
from src.model.helpers import classify

HOST = "0.0.0.0"
PORT = 8000
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MODEL_PATH = "models/model.json"


def load_recognizer() -> tuple[MLP, HandLandmarker]:
    """Create the model and landmark detector used by the web server

    Returns:
        (tuple):
            (MLP): loaded model (classifier)
            (HandLandmarker): hand landmark detector
    """
    model = MLP()
    load_model(model, MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=BASE_OPTIONS,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return model, vision.HandLandmarker.create_from_options(options)


MODEL, DETECTOR = load_recognizer()


def predict(image_data: bytes) -> dict[str, object]:
    """Predict a sign language letter from encoded image data

    Args:
        image_data (bytes): JPG, JPEG, or PNG image bytes received from the browser

    Returns:
        (dict) A dictionary containing the predicted letter and its confidence.
        `UNK` with zero confidence is returned when no usable hand is
        detected
    """
    image = cv.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Request body is not a valid image")

    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    result = DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image))
    if not result.hand_landmarks:
        return {"prediction": "UNK", "confidence": 0.0}

    landmarks = normalize_landmarks(result.hand_landmarks[0])
    if landmarks is None:
        return {"prediction": "UNK", "confidence": 0.0}

    model_input = np.array([result.handedness[0][0].index, *landmarks])
    probabilities, label = classify(MODEL(model_input))
    return {
        "prediction": LABEL_TO_LETTER[label],
        "confidence": float(max(probabilities)),
    }


class RequestHandler(BaseHTTPRequestHandler):
    """Handle requests for the browser UI and prediction endpoint"""

    def do_GET(self) -> None:
        """Serve the browser application for the root URL"""
        if self.path != "/":
            self.send_error(404)
            return

        page = Path(__file__).with_name("web.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:
        """Return prediction for the image to "/predict" """
        if self.path != "/predict":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if not 0 < content_length <= MAX_IMAGE_SIZE:
            self.send_error(413, "Image is missing or too large")
            return

        try:
            result = predict(self.rfile.read(content_length))
        except (ValueError, cv.error) as error:
            self.send_error(400, str(error))
            return

        response = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main() -> None:
    """Start the HTTP server and serve the browser application"""
    server = HTTPServer((HOST, PORT), RequestHandler)
    print(f"Open http://localhost:{PORT} in a browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        DETECTOR.close()
        server.server_close()


if __name__ == "__main__":
    main()
