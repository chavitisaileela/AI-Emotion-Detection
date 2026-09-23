from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import time

app = Flask(__name__)

# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------------------------------------------------
# HAAR CASCADES
# ---------------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"
smile_cascade = cv2.CascadeClassifier(smile_path)

# ---------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------

last_analysis = None
last_analysis_time = 0

ANALYSIS_INTERVAL = 0.8


# ---------------------------------------------------------
# EMOTION ANALYSIS
# ---------------------------------------------------------

def analyze_emotion(frame):

    global last_analysis
    global last_analysis_time

    current_time = time.time()

    # Don't run DeepFace on every frame
    if (
        last_analysis is not None
        and current_time - last_analysis_time < ANALYSIS_INTERVAL
    ):
        return last_analysis

    try:

        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        if isinstance(result, list):

            if len(result) > 0:
                result = result[0]
            else:
                return last_analysis

        emotion_data = result.get("emotion", {})

        dominant = result.get(
            "dominant_emotion",
            "neutral"
        )

        confidence = emotion_data.get(
            dominant,
            0
        )

        data = {
            "dominant": dominant,
            "confidence": float(confidence),
            "emotions": emotion_data
        }

        last_analysis = data
        last_analysis_time = current_time

        return data

    except Exception:

        return last_analysis


# ---------------------------------------------------------
# FACIAL FEATURE DETECTION
# ---------------------------------------------------------

def detect_features(face_gray, face_color):

    features = {
        "eyes": [],
        "smiles": [],
        "eyebrows": [],
        "nose": None
    }

    # -------------------------
    # EYES
    # -------------------------

    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(25, 25)
    )

    for (x, y, w, h) in eyes:

        features["eyes"].append(
            (x, y, w, h)
        )

    # -------------------------
    # SMILE
    # -------------------------

    smile_region_y = int(face_gray.shape[0] * 0.48)

    smile_region = face_gray[
        smile_region_y:,
        :
    ]

    if smile_region.size > 0:

        smiles = smile_cascade.detectMultiScale(
            smile_region,
            scaleFactor=1.7,
            minNeighbors=15,
            minSize=(40, 20)
        )

        for (x, y, w, h) in smiles:

            features["smiles"].append(
                (
                    x,
                    y + smile_region_y,
                    w,
                    h
                )
            )

    # -------------------------
    # EYEBROWS
    # Approximate eyebrow regions
    # -------------------------

    face_h, face_w = face_gray.shape

    eyebrow_y = int(face_h * 0.18)
    eyebrow_h = int(face_h * 0.14)

    left_eyebrow = (
        int(face_w * 0.15),
        eyebrow_y,
        int(face_w * 0.25),
        eyebrow_h
    )

    right_eyebrow = (
        int(face_w * 0.60),
        eyebrow_y,
        int(face_w * 0.25),
        eyebrow_h
    )

    features["eyebrows"] = [
        left_eyebrow,
        right_eyebrow
    ]

    # -------------------------
    # NOSE
    # Approximate nose region
    # -------------------------

    nose_x = int(face_w * 0.38)
    nose_y = int(face_h * 0.30)
    nose_w = int(face_w * 0.24)
    nose_h = int(face_h * 0.30)

    features["nose"] = (
        nose_x,
        nose_y,
        nose_w,
        nose_h
    )

    return features


# ---------------------------------------------------------
# DRAW TEXT
# ---------------------------------------------------------

def put_label(
    frame,
    text,
    position,
    color=(255, 255, 255),
    background=(80, 40, 180)
):

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.55
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (x, y - th - 10),
        (x + tw + 12, y + 5),
        background,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 6, y - 2),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ---------------------------------------------------------
# VIDEO STREAM
# ---------------------------------------------------------

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Smaller frame for processing
        processing_frame = cv2.resize(
            frame,
            (640, 360)
        )

        gray = cv2.cvtColor(
            processing_frame,
            cv2.COLOR_BGR2GRAY
        )

        # -------------------------------------------------
        # FACE DETECTION
        # -------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80)
        )

        # -------------------------------------------------
        # EMOTION
        # -------------------------------------------------

        emotion_result = analyze_emotion(
            processing_frame
        )

        # -------------------------------------------------
        # PROCESS EACH FACE
        # -------------------------------------------------

        for index, (x, y, w, h) in enumerate(faces):

            face_gray = gray[
                y:y+h,
                x:x+w
            ]

            face_color = processing_frame[
                y:y+h,
                x:x+w
            ]

            if face_gray.size == 0:
                continue

            features = detect_features(
                face_gray,
                face_color
            )

            # Scale factor from 640x360 to 1280x720
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 360

            # -------------------------------------------------
            # FACE BOX
            # -------------------------------------------------

            fx = int(x * scale_x)
            fy = int(y * scale_y)
            fw = int(w * scale_x)
            fh = int(h * scale_y)

            cv2.rectangle(
                frame,
                (fx, fy),
                (fx + fw, fy + fh),
                (50, 220, 80),
                2
            )

            put_label(
                frame,
                f"Face {index + 1}",
                (fx, fy - 5),
                background=(50, 160, 80)
            )

            # -------------------------------------------------
            # EYES
            # -------------------------------------------------

            for ex, ey, ew, eh in features["eyes"]:

                ex2 = int((x + ex) * scale_x)
                ey2 = int((y + ey) * scale_y)

                ew2 = int(ew * scale_x)
                eh2 = int(eh * scale_y)

                cv2.rectangle(
                    frame,
                    (ex2, ey2),
                    (ex2 + ew2, ey2 + eh2),
                    (255, 150, 40),
                    2
                )

            # -------------------------------------------------
            # EYEBROWS
            # -------------------------------------------------

            for bx, by, bw, bh in features["eyebrows"]:

                bx2 = int((x + bx) * scale_x)
                by2 = int((y + by) * scale_y)

                bw2 = int(bw * scale_x)
                bh2 = int(bh * scale_y)

                cv2.rectangle(
                    frame,
                    (bx2, by2),
                    (bx2 + bw2, by2 + bh2),
                    (180, 70, 255),
                    2
                )

            # -------------------------------------------------
            # NOSE
            # -------------------------------------------------

            if features["nose"]:

                nx, ny, nw, nh = features["nose"]

                nx2 = int((x + nx) * scale_x)
                ny2 = int((y + ny) * scale_y)

                nw2 = int(nw * scale_x)
                nh2 = int(nh * scale_y)

                cv2.rectangle(
                    frame,
                    (nx2, ny2),
                    (nx2 + nw2, ny2 + nh2),
                    (0, 180, 255),
                    2
                )

                put_label(
                    frame,
                    "Nose",
                    (nx2, ny2),
                    background=(0, 130, 220)
                )

            # -------------------------------------------------
            # SMILE
            # -------------------------------------------------

            for sx, sy, sw, sh in features["smiles"]:

                sx2 = int((x + sx) * scale_x)
                sy2 = int((y + sy) * scale_y)

                sw2 = int(sw * scale_x)
                sh2 = int(sh * scale_y)

                cv2.rectangle(
                    frame,
                    (sx2, sy2),
                    (sx2 + sw2, sy2 + sh2),
                    (255, 80, 180),
                    2
                )

                put_label(
                    frame,
                    "Smile",
                    (sx2, sy2),
                    background=(200, 40, 140)
                )

            # -------------------------------------------------
            # FEATURE LABELS
            # -------------------------------------------------

            label_y = fy + fh + 25

            eye_text = f"Eyes: {len(features['eyes'])}"

            smile_text = (
                "Smile: Detected"
                if len(features["smiles"]) > 0
                else "Smile: Not detected"
            )

            eyebrow_text = "Eyebrows: Detected"

            cv2.putText(
                frame,
                eye_text,
                (fx, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 150, 40),
                2
            )

            cv2.putText(
                frame,
                smile_text,
                (fx, label_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 80, 180),
                2
            )

            cv2.putText(
                frame,
                eyebrow_text,
                (fx, label_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 70, 255),
                2
            )

        # -------------------------------------------------
        # EMOTION INFORMATION
        # -------------------------------------------------

        if emotion_result:

            dominant = emotion_result["dominant"]

            confidence = emotion_result["confidence"]

            emotions = emotion_result["emotions"]

            # Crying indication
            if dominant.lower() == "sad":

                display_emotion = "Sad / Possible Crying"

            else:

                display_emotion = dominant.capitalize()

            # Background panel
            cv2.rectangle(
                frame,
                (20, 20),
                (380, 145),
                (25, 25, 45),
                -1
            )

            cv2.rectangle(
                frame,
                (20, 20),
                (380, 145),
                (120, 90, 220),
                2
            )

            cv2.putText(
                frame,
                "Detected Emotion",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 220, 255),
                2
            )

            cv2.putText(
                frame,
                display_emotion,
                (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 220, 255),
                2
            )

            # -------------------------------------------------
            # EMOTION BARS
            # -------------------------------------------------

            emotion_names = [
                "happy",
                "sad",
                "angry",
                "fear",
                "surprise",
                "disgust",
                "neutral"
            ]

            start_y = 175

            for i, name in enumerate(emotion_names):

                value = float(
                    emotions.get(name, 0)
                )

                bar_y = start_y + i * 25

                cv2.putText(
                    frame,
                    name.capitalize(),
                    (25, bar_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

                bar_width = int(
                    min(value, 100) * 1.8
                )

                cv2.rectangle(
                    frame,
                    (110, bar_y - 12),
                    (110 + bar_width, bar_y),
                    (130, 80, 220),
                    -1
                )

                cv2.putText(
                    frame,
                    f"{value:.1f}%",
                    (300, bar_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (230, 230, 230),
                    1
                )

        # -------------------------------------------------
        # FACE COUNT
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (frame.shape[1] - 170, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # -------------------------------------------------
        # CAMERA STATUS
        # -------------------------------------------------

        cv2.circle(
            frame,
            (30, frame.shape[0] - 30),
            8,
            (50, 220, 100),
            -1
        )

        cv2.putText(
            frame,
            "Camera Active",
            (48, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # -------------------------------------------------
        # ENCODE
        # -------------------------------------------------

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ---------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )