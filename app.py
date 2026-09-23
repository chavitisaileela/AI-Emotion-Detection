from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from deepface import DeepFace
import os

app = Flask(__name__)

# ============================================================
# HAAR CASCADE FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascades"
)

face_cascade = cv2.CascadeClassifier(
    os.path.join(
        CASCADE_PATH,
        "haarcascade_frontalface_default.xml"
    )
)

eye_cascade = cv2.CascadeClassifier(
    os.path.join(
        CASCADE_PATH,
        "haarcascade_eye.xml"
    )
)

smile_cascade = cv2.CascadeClassifier(
    os.path.join(
        CASCADE_PATH,
        "haarcascade_smile.xml"
    )
)

# Check that cascade files loaded correctly
if face_cascade.empty():
    print("ERROR: Face cascade file could not be loaded.")

if eye_cascade.empty():
    print("ERROR: Eye cascade file could not be loaded.")

if smile_cascade.empty():
    print("ERROR: Smile cascade file could not be loaded.")


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def analyze_emotion(face_image):

    try:

        result = DeepFace.analyze(
            face_image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get(
            "emotion",
            {}
        )

        dominant_emotion = result.get(
            "dominant_emotion",
            "neutral"
        )

        confidence = emotions.get(
            dominant_emotion,
            0
        )

        return {
            "emotion": dominant_emotion,
            "confidence": float(confidence),
            "emotions": {
                key.lower(): float(value)
                for key, value in emotions.items()
            }
        }

    except Exception as error:

        print(
            "Emotion analysis error:",
            error
        )

        return {
            "emotion": "neutral",
            "confidence": 0,
            "emotions": {}
        }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE CAMERA FRAME
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    try:

        # Receive JSON data
        data = request.get_json()

        if not data or "frame" not in data:

            return jsonify({
                "success": False,
                "error": "No frame received"
            }), 400

        # ----------------------------------------------------
        # Decode Base64 image
        # ----------------------------------------------------

        image_data = data["frame"]

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]

        image_bytes = base64.b64decode(
            image_data
        )

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "success": False,
                "error": "Could not decode image"
            }), 400

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        frame = cv2.resize(
            frame,
            (640, 360)
        )

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        face_results = []

        # ====================================================
        # PROCESS EACH FACE
        # ====================================================

        for (x, y, w, h) in faces:

            # ------------------------------------------------
            # Extract face
            # ------------------------------------------------

            face_image = frame[
                y:y + h,
                x:x + w
            ]

            # ------------------------------------------------
            # Emotion detection
            # ------------------------------------------------

            emotion_data = analyze_emotion(
                face_image
            )

            # ------------------------------------------------
            # Face grayscale
            # ------------------------------------------------

            face_gray = gray[
                y:y + h,
                x:x + w
            ]

            # ------------------------------------------------
            # Eye detection
            # ------------------------------------------------

            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(15, 15)
            )

            eye_results = []

            for (ex, ey, ew, eh) in eyes[:4]:

                eye_results.append({

                    "x": int(x + ex),

                    "y": int(y + ey),

                    "w": int(ew),

                    "h": int(eh)

                })

            # ------------------------------------------------
            # Smile detection
            # ------------------------------------------------

            smiles = smile_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(25, 25)
            )

            smile_result = None

            if len(smiles) > 0:

                sx, sy, sw, sh = smiles[0]

                smile_result = {

                    "x": int(x + sx),

                    "y": int(y + sy),

                    "w": int(sw),

                    "h": int(sh)

                }

            # ------------------------------------------------
            # Store face result
            # ------------------------------------------------

            face_results.append({

                "x": int(x),

                "y": int(y),

                "w": int(w),

                "h": int(h),

                "emotion": emotion_data[
                    "emotion"
                ],

                "confidence": emotion_data[
                    "confidence"
                ],

                "emotions": emotion_data[
                    "emotions"
                ],

                "eyes": eye_results,

                "smile": smile_result

            })

        # ====================================================
        # SEND RESULT TO BROWSER
        # ====================================================

        return jsonify({

            "success": True,

            "face_count": len(
                face_results
            ),

            "faces": face_results

        })

    except Exception as error:

        print(
            "Frame processing error:",
            error
        )

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )