from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from deepface import DeepFace
import os
import time
import threading
import traceback


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascades"
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

MIN_FACE_WIDTH = 60
MIN_FACE_HEIGHT = 60

FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5

ANALYSIS_MAX_FACES = 5

EMOTION_NAMES = [
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "neutral"
]

MODEL_LOCK = threading.Lock()

MODEL_WARMED_UP = False


# ============================================================
# APPLICATION STATE
# ============================================================

SERVER_START_TIME = time.time()

TOTAL_REQUESTS = 0
SUCCESSFUL_REQUESTS = 0
FAILED_REQUESTS = 0

REQUEST_LOCK = threading.Lock()


# ============================================================
# HAAR CASCADE PATHS
# ============================================================

FACE_CASCADE_FILE = os.path.join(
    CASCADE_PATH,
    "haarcascade_frontalface_default.xml"
)

EYE_CASCADE_FILE = os.path.join(
    CASCADE_PATH,
    "haarcascade_eye.xml"
)

SMILE_CASCADE_FILE = os.path.join(
    CASCADE_PATH,
    "haarcascade_smile.xml"
)


# ============================================================
# LOAD FACE CASCADE
# ============================================================

face_cascade = cv2.CascadeClassifier(
    FACE_CASCADE_FILE
)


# ============================================================
# LOAD EYE CASCADE
# ============================================================

eye_cascade = cv2.CascadeClassifier(
    EYE_CASCADE_FILE
)


# ============================================================
# LOAD SMILE CASCADE
# ============================================================

smile_cascade = cv2.CascadeClassifier(
    SMILE_CASCADE_FILE
)


# ============================================================
# CHECK CASCADE FILES
# ============================================================

if face_cascade.empty():

    print(
        "ERROR: Face cascade file could not be loaded:"
    )

    print(
        FACE_CASCADE_FILE
    )


if eye_cascade.empty():

    print(
        "WARNING: Eye cascade file could not be loaded:"
    )

    print(
        EYE_CASCADE_FILE
    )


if smile_cascade.empty():

    print(
        "WARNING: Smile cascade file could not be loaded:"
    )

    print(
        SMILE_CASCADE_FILE
    )


# ============================================================
# PRINT STARTUP INFORMATION
# ============================================================

print("=" * 60)

print(
    "EMOTION DETECTION USING AI"
)

print("=" * 60)

print(
    "Base directory:",
    BASE_DIR
)

print(
    "Cascade directory:",
    CASCADE_PATH
)

print(
    "Face cascade:",
    FACE_CASCADE_FILE
)

print(
    "Eye cascade:",
    EYE_CASCADE_FILE
)

print(
    "Smile cascade:",
    SMILE_CASCADE_FILE
)

print(
    "Frame size:",
    FRAME_WIDTH,
    "x",
    FRAME_HEIGHT
)

print("=" * 60)


# ============================================================
# UTILITY: INCREMENT REQUEST COUNTER
# ============================================================

def increment_request_count():

    global TOTAL_REQUESTS

    with REQUEST_LOCK:

        TOTAL_REQUESTS += 1


# ============================================================
# UTILITY: INCREMENT SUCCESS COUNTER
# ============================================================

def increment_success_count():

    global SUCCESSFUL_REQUESTS

    with REQUEST_LOCK:

        SUCCESSFUL_REQUESTS += 1


# ============================================================
# UTILITY: INCREMENT FAILURE COUNTER
# ============================================================

def increment_failure_count():

    global FAILED_REQUESTS

    with REQUEST_LOCK:

        FAILED_REQUESTS += 1


# ============================================================
# UTILITY: GET STATISTICS
# ============================================================

def get_statistics():

    with REQUEST_LOCK:

        return {
            "total_requests": TOTAL_REQUESTS,
            "successful_requests": SUCCESSFUL_REQUESTS,
            "failed_requests": FAILED_REQUESTS
        }


# ============================================================
# UTILITY: CREATE ZERO EMOTIONS
# ============================================================

def empty_emotions():

    return {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "neutral": 0.0
    }


# ============================================================
# UTILITY: NORMALIZE EMOTION VALUES
# ============================================================

def normalize_emotions(emotions):

    result = empty_emotions()

    if not isinstance(
        emotions,
        dict
    ):

        return result

    for key, value in emotions.items():

        emotion_name = str(
            key
        ).strip().lower()

        if emotion_name not in EMOTION_NAMES:

            continue

        try:

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            numeric_value = 0.0

        if not np.isfinite(
            numeric_value
        ):

            numeric_value = 0.0

        numeric_value = max(
            0.0,
            min(
                100.0,
                numeric_value
            )
        )

        result[
            emotion_name
        ] = numeric_value

    return result


# ============================================================
# UTILITY: FIND DOMINANT EMOTION
# ============================================================

def find_dominant_emotion(
    emotions
):

    normalized = normalize_emotions(
        emotions
    )

    dominant = max(
        normalized,
        key=normalized.get
    )

    confidence = normalized[
        dominant
    ]

    return (
        dominant,
        confidence
    )


# ============================================================
# UTILITY: VALIDATE EMOTION RESULT
# ============================================================

def validate_emotion_result(
    result
):

    if result is None:

        return False

    if not isinstance(
        result,
        dict
    ):

        return False

    if "emotion" not in result:

        return False

    if "confidence" not in result:

        return False

    if "emotions" not in result:

        return False

    emotions = result[
        "emotions"
    ]

    if not isinstance(
        emotions,
        dict
    ):

        return False

    return True


# ============================================================
# DEEPAFACE EMOTION ANALYSIS
# ============================================================

def analyze_emotion(
    face_image
):

    start_time = time.time()

    try:

        if face_image is None:

            raise ValueError(
                "Face image is empty."
            )

        if face_image.size == 0:

            raise ValueError(
                "Face image contains no pixels."
            )

        face_height, face_width = (
            face_image.shape[:2]
        )

        if face_width < 20:

            raise ValueError(
                "Face image is too small for emotion analysis."
            )

        if face_height < 20:

            raise ValueError(
                "Face image is too small for emotion analysis."
            )

        # ----------------------------------------------------
        # Add a small border around the detected face.
        # This gives the emotion model slightly more context.
        # ----------------------------------------------------

        padding_x = int(
            face_width * 0.10
        )

        padding_y = int(
            face_height * 0.10
        )

        padded_face = cv2.copyMakeBorder(
            face_image,
            padding_y,
            padding_y,
            padding_x,
            padding_x,
            cv2.BORDER_REPLICATE
        )

        # ----------------------------------------------------
        # DeepFace receives an already detected face.
        #
        # IMPORTANT:
        # detector_backend="skip"
        #
        # prevents DeepFace from trying to detect another
        # face inside our already cropped face.
        # ----------------------------------------------------

        with MODEL_LOCK:

            result = DeepFace.analyze(

                img_path=padded_face,

                actions=[
                    "emotion"
                ],

                detector_backend="skip",

                enforce_detection=False,

                silent=True
            )

        # ----------------------------------------------------
        # DeepFace can return either a dictionary or list.
        # ----------------------------------------------------

        if isinstance(
            result,
            list
        ):

            if len(result) == 0:

                raise ValueError(
                    "DeepFace returned an empty result."
                )

            result = result[0]

        # ----------------------------------------------------
        # Validate result.
        # ----------------------------------------------------

        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "DeepFace returned an unexpected result."
            )

        # ----------------------------------------------------
        # Read emotion probabilities.
        # ----------------------------------------------------

        raw_emotions = result.get(
            "emotion",
            {}
        )

        emotions = normalize_emotions(
            raw_emotions
        )

        # ----------------------------------------------------
        # Determine dominant emotion ourselves.
        #
        # This makes the result independent of whether the
        # DeepFace version includes dominant_emotion.
        # ----------------------------------------------------

        dominant_emotion = result.get(
            "dominant_emotion"
        )

        if dominant_emotion is not None:

            dominant_emotion = str(
                dominant_emotion
            ).strip().lower()

        if dominant_emotion not in EMOTION_NAMES:

            dominant_emotion, confidence = (
                find_dominant_emotion(
                    emotions
                )
            )

        else:

            confidence = emotions.get(
                dominant_emotion,
                0.0
            )

        # ----------------------------------------------------
        # If all values are zero, report an analysis problem
        # instead of pretending that the emotion is neutral.
        # ----------------------------------------------------

        total_emotion_value = sum(
            emotions.values()
        )

        if total_emotion_value <= 0:

            raise ValueError(
                "DeepFace returned zero emotion probabilities."
            )

        # ----------------------------------------------------
        # Make sure confidence is numeric.
        # ----------------------------------------------------

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = emotions.get(
                dominant_emotion,
                0.0
            )

        confidence = max(
            0.0,
            min(
                100.0,
                confidence
            )
        )

        processing_time = (
            time.time() - start_time
        )

        print(
            "Emotion:",
            dominant_emotion,
            "| Confidence:",
            round(
                confidence,
                2
            ),
            "| Time:",
            round(
                processing_time,
                3
            ),
            "sec"
        )

        print(
            "Emotions:",
            {
                key: round(
                    value,
                    2
                )
                for key, value
                in emotions.items()
            }
        )

        return {

            "success": True,

            "emotion": dominant_emotion,

            "confidence": confidence,

            "emotions": emotions,

            "processing_time": round(
                processing_time,
                3
            ),

            "error": None

        }

    except Exception as error:

        processing_time = (
            time.time() - start_time
        )

        print(
            "=" * 60
        )

        print(
            "DEEPAFACE EMOTION ANALYSIS ERROR"
        )

        print(
            "Error:",
            str(error)
        )

        print(
            "Processing time:",
            round(
                processing_time,
                3
            ),
            "seconds"
        )

        print(
            traceback.format_exc()
        )

        print(
            "=" * 60
        )

        return {

            "success": False,

            "emotion": None,

            "confidence": 0.0,

            "emotions": empty_emotions(),

            "processing_time": round(
                processing_time,
                3
            ),

            "error": str(error)

        }


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(
    frame
):

    if frame is None:

        return []

    if frame.size == 0:

        return []

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Improve contrast slightly.
    # --------------------------------------------------------

    gray = cv2.equalizeHist(
        gray
    )

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=FACE_SCALE_FACTOR,

        minNeighbors=FACE_MIN_NEIGHBORS,

        minSize=(
            MIN_FACE_WIDTH,
            MIN_FACE_HEIGHT
        )

    )

    if faces is None:

        return []

    return faces


# ============================================================
# EYE DETECTION
# ============================================================

def detect_eyes(
    face_gray,
    offset_x,
    offset_y
):

    if eye_cascade.empty():

        return []

    if face_gray is None:

        return []

    if face_gray.size == 0:

        return []

    try:

        eyes = eye_cascade.detectMultiScale(

            face_gray,

            scaleFactor=1.1,

            minNeighbors=5,

            minSize=(15, 15)

        )

    except Exception as error:

        print(
            "Eye detection error:",
            error
        )

        return []

    eye_results = []

    for (
        ex,
        ey,
        ew,
        eh
    ) in eyes[:4]:

        eye_results.append({

            "x": int(
                offset_x + ex
            ),

            "y": int(
                offset_y + ey
            ),

            "w": int(
                ew
            ),

            "h": int(
                eh
            )

        })

    return eye_results


# ============================================================
# SMILE DETECTION
# ============================================================

def detect_smile(
    face_gray,
    offset_x,
    offset_y
):

    if smile_cascade.empty():

        return None

    if face_gray is None:

        return None

    if face_gray.size == 0:

        return None

    try:

        smiles = smile_cascade.detectMultiScale(

            face_gray,

            scaleFactor=1.7,

            minNeighbors=20,

            minSize=(25, 25)

        )

    except Exception as error:

        print(
            "Smile detection error:",
            error
        )

        return None

    if len(smiles) == 0:

        return None

    sx, sy, sw, sh = (
        smiles[0]
    )

    return {

        "x": int(
            offset_x + sx
        ),

        "y": int(
            offset_y + sy
        ),

        "w": int(
            sw
        ),

        "h": int(
            sh
        )

    }


# ============================================================
# PROCESS SINGLE FACE
# ============================================================

def process_face(
    frame,
    gray,
    face_box,
    face_index
):

    x, y, w, h = (
        face_box
    )

    # --------------------------------------------------------
    # Keep coordinates inside the image.
    # --------------------------------------------------------

    x = max(
        0,
        int(x)
    )

    y = max(
        0,
        int(y)
    )

    w = max(
        1,
        int(w)
    )

    h = max(
        1,
        int(h)
    )

    frame_height, frame_width = (
        frame.shape[:2]
    )

    if x + w > frame_width:

        w = frame_width - x

    if y + h > frame_height:

        h = frame_height - y

    # --------------------------------------------------------
    # Extract face.
    # --------------------------------------------------------

    face_image = frame[
        y:y + h,
        x:x + w
    ]

    face_gray = gray[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------------
    # Analyze emotion.
    # --------------------------------------------------------

    emotion_data = analyze_emotion(
        face_image
    )

    # --------------------------------------------------------
    # Detect eyes.
    # --------------------------------------------------------

    eye_results = detect_eyes(

        face_gray,

        x,

        y

    )

    # --------------------------------------------------------
    # Detect smile.
    # --------------------------------------------------------

    smile_result = detect_smile(

        face_gray,

        x,

        y

    )

    # --------------------------------------------------------
    # Prepare emotion response.
    # --------------------------------------------------------

    if emotion_data["success"]:

        emotion = emotion_data[
            "emotion"
        ]

        confidence = emotion_data[
            "confidence"
        ]

        emotions = emotion_data[
            "emotions"
        ]

        analysis_status = "success"

        analysis_error = None

    else:

        emotion = None

        confidence = 0.0

        emotions = empty_emotions()

        analysis_status = "error"

        analysis_error = emotion_data[
            "error"
        ]

    # --------------------------------------------------------
    # Return complete face object.
    # --------------------------------------------------------

    return {

        "index": int(
            face_index
        ),

        "x": int(x),

        "y": int(y),

        "w": int(w),

        "h": int(h),

        "emotion": emotion,

        "confidence": float(
            confidence
        ),

        "emotions": emotions,

        "eyes": eye_results,

        "smile": smile_result,

        "analysis_status":
            analysis_status,

        "analysis_error":
            analysis_error,

        "processing_time":
            emotion_data.get(
                "processing_time",
                0.0
            )

    }


# ============================================================
# DECODE BASE64 IMAGE
# ============================================================

def decode_base64_image(
    image_data
):

    if not image_data:

        raise ValueError(
            "Image data is empty."
        )

    # --------------------------------------------------------
    # Remove data URL prefix.
    # --------------------------------------------------------

    if "," in image_data:

        image_data = image_data.split(
            ",",
            1
        )[1]

    try:

        image_bytes = base64.b64decode(
            image_data,
            validate=True
        )

    except Exception as error:

        raise ValueError(
            "Invalid Base64 image data: "
            + str(error)
        )

    if not image_bytes:

        raise ValueError(
            "Decoded image contains no data."
        )

    # --------------------------------------------------------
    # Convert bytes into NumPy array.
    # --------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    if image_array.size == 0:

        raise ValueError(
            "Image array is empty."
        )

    # --------------------------------------------------------
    # Decode JPEG/PNG image.
    # --------------------------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise ValueError(
            "OpenCV could not decode the image."
        )

    return frame


# ============================================================
# RESIZE FRAME
# ============================================================

def resize_frame(
    frame
):

    if frame is None:

        raise ValueError(
            "Cannot resize an empty frame."
        )

    return cv2.resize(

        frame,

        (
            FRAME_WIDTH,
            FRAME_HEIGHT
        ),

        interpolation=cv2.INTER_AREA

    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    uptime = (
        time.time()
        - SERVER_START_TIME
    )

    statistics = get_statistics()

    return jsonify({

        "status": "ok",

        "application":
            "Emotion Detection Using AI",

        "uptime_seconds":
            round(
                uptime,
                2
            ),

        "statistics":
            statistics,

        "face_cascade_loaded":
            not face_cascade.empty(),

        "eye_cascade_loaded":
            not eye_cascade.empty(),

        "smile_cascade_loaded":
            not smile_cascade.empty()

    })


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# SERVER INFORMATION
# ============================================================

@app.route(
    "/api/info",
    methods=["GET"]
)
def api_info():

    return jsonify({

        "success": True,

        "application":
            "Emotion Detection Using AI",

        "version":
            "2.0",

        "framework":
            "Flask",

        "emotion_engine":
            "DeepFace",

        "face_detector":
            "OpenCV Haar Cascade",

        "supported_emotions":
            EMOTION_NAMES,

        "frame_width":
            FRAME_WIDTH,

        "frame_height":
            FRAME_HEIGHT

    })


# ============================================================
# ANALYZE CAMERA FRAME
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    request_start = time.time()

    increment_request_count()

    try:

        # ----------------------------------------------------
        # Read JSON body.
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if data is None:

            increment_failure_count()

            return jsonify({

                "success": False,

                "error":
                    "Request body must contain JSON data."

            }), 400

        # ----------------------------------------------------
        # Check frame field.
        # ----------------------------------------------------

        if "frame" not in data:

            increment_failure_count()

            return jsonify({

                "success": False,

                "error":
                    "No frame received."

            }), 400

        image_data = data[
            "frame"
        ]

        if not isinstance(
            image_data,
            str
        ):

            increment_failure_count()

            return jsonify({

                "success": False,

                "error":
                    "Frame must be a Base64 string."

            }), 400

        # ----------------------------------------------------
        # Decode image.
        # ----------------------------------------------------

        frame = decode_base64_image(
            image_data
        )

        # ----------------------------------------------------
        # Resize image.
        # ----------------------------------------------------

        frame = resize_frame(
            frame
        )

        # ----------------------------------------------------
        # Convert to grayscale.
        # ----------------------------------------------------

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY

        )

        # ----------------------------------------------------
        # Improve grayscale image.
        # ----------------------------------------------------

        gray = cv2.equalizeHist(
            gray
        )

        # ----------------------------------------------------
        # Detect faces.
        # ----------------------------------------------------

        faces = detect_faces(
            frame
        )

        # ----------------------------------------------------
        # Limit number of faces.
        # ----------------------------------------------------

        faces = faces[
            :ANALYSIS_MAX_FACES
        ]

        face_results = []

        # ----------------------------------------------------
        # Process every detected face.
        # ----------------------------------------------------

        for face_index, face_box in enumerate(
            faces
        ):

            try:

                face_result = process_face(

                    frame,

                    gray,

                    face_box,

                    face_index

                )

                face_results.append(
                    face_result
                )

            except Exception as face_error:

                print(
                    "Face processing error:",
                    face_error
                )

                traceback.print_exc()

                x, y, w, h = (
                    face_box
                )

                face_results.append({

                    "index":
                        int(face_index),

                    "x":
                        int(x),

                    "y":
                        int(y),

                    "w":
                        int(w),

                    "h":
                        int(h),

                    "emotion":
                        None,

                    "confidence":
                        0.0,

                    "emotions":
                        empty_emotions(),

                    "eyes":
                        [],

                    "smile":
                        None,

                    "analysis_status":
                        "error",

                    "analysis_error":
                        str(face_error),

                    "processing_time":
                        0.0

                })

        # ----------------------------------------------------
        # Calculate total processing time.
        # ----------------------------------------------------

        processing_time = (
            time.time()
            - request_start
        )

        # ----------------------------------------------------
        # Determine whether at least one emotion analysis
        # was successful.
        # ----------------------------------------------------

        successful_faces = [

            face

            for face in face_results

            if face.get(
                "analysis_status"
            ) == "success"

        ]

        analysis_success = (
            len(successful_faces) > 0
        )

        if analysis_success:

            increment_success_count()

        else:

            increment_failure_count()

        # ----------------------------------------------------
        # Return result.
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "analysis_success":
                analysis_success,

            "face_count":
                len(face_results),

            "faces":
                face_results,

            "processing_time":
                round(
                    processing_time,
                    3
                ),

            "server_message":
                (
                    "Emotion analysis completed."
                    if analysis_success
                    else
                    "Face detected, but emotion analysis failed."
                )

        })

    except ValueError as error:

        increment_failure_count()

        print(
            "Input processing error:",
            str(error)
        )

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 400

    except Exception as error:

        increment_failure_count()

        print(
            "=" * 60
        )

        print(
            "FRAME PROCESSING ERROR"
        )

        print(
            "Error:",
            str(error)
        )

        traceback.print_exc()

        print(
            "=" * 60
        )

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ============================================================
# GLOBAL 404 HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error":
            "Requested endpoint was not found."

    }), 404


# ============================================================
# GLOBAL 413 HANDLER
# ============================================================

@app.errorhandler(413)
def request_too_large(error):

    return jsonify({

        "success": False,

        "error":
            "Uploaded frame is too large."

    }), 413


# ============================================================
# GLOBAL 500 HANDLER
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "success": False,

        "error":
            "Internal server error."

    }), 500


# ============================================================
# WARMUP FUNCTION
# ============================================================

def warmup_message():

    print(
        "=" * 60
    )

    print(
        "DeepFace emotion model will initialize "
        "when the first face is analyzed."
    )

    print(
        "Using detector_backend='skip' "
        "for already-cropped faces."
    )

    print(
        "=" * 60
    )


# ============================================================
# STARTUP VALIDATION
# ============================================================

def validate_startup():

    errors = []

    if face_cascade.empty():

        errors.append(
            "Face cascade is not loaded."
        )

    if not os.path.exists(
        FACE_CASCADE_FILE
    ):

        errors.append(
            "Face cascade file does not exist."
        )

    if not os.path.exists(
        EYE_CASCADE_FILE
    ):

        print(
            "WARNING: Eye cascade file does not exist."
        )

    if not os.path.exists(
        SMILE_CASCADE_FILE
    ):

        print(
            "WARNING: Smile cascade file does not exist."
        )

    if errors:

        print(
            "STARTUP WARNINGS:"
        )

        for error in errors:

            print(
                "-",
                error
            )

    else:

        print(
            "Startup validation completed successfully."
        )


# ============================================================
# APPLICATION START
# ============================================================

validate_startup()

warmup_message()


# ============================================================
# LOCAL DEVELOPMENT SERVER
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(

            "PORT",

            5000

        )

    )

    print(
        "Starting Flask server..."
    )

    print(
        "Host: 0.0.0.0"
    )

    print(
        "Port:",
        port
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False,

        threaded=True

    )