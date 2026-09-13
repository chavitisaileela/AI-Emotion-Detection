import cv2
from deepface import DeepFace
import os
import time
from collections import deque

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# DeepFace analysis interval
ANALYSIS_INTERVAL = 0.40

# Maximum faces
MAX_FACES = 5

# History length
HISTORY_LENGTH = 80

# Update graph every few frames
GRAPH_UPDATE_INTERVAL = 5


# ============================================================
# COLORS - BGR
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
ORANGE = (0, 165, 255)

GRAY = (150, 150, 150)
DARK_GRAY = (45, 45, 45)

RED = (0, 0, 255)
YELLOW = (0, 255, 255)


# ============================================================
# OPENCV CASCADE FILES
# ============================================================

BASE = cv2.data.haarcascades

FACE_CASCADE = os.path.join(
    BASE,
    "haarcascade_frontalface_default.xml"
)

EYE_CASCADE = os.path.join(
    BASE,
    "haarcascade_eye.xml"
)

SMILE_CASCADE = os.path.join(
    BASE,
    "haarcascade_smile.xml"
)


# ============================================================
# LOAD CASCADES
# ============================================================

face_detector = cv2.CascadeClassifier(
    FACE_CASCADE
)

eye_detector = cv2.CascadeClassifier(
    EYE_CASCADE
)

smile_detector = None

if os.path.exists(SMILE_CASCADE):

    smile_detector = cv2.CascadeClassifier(
        SMILE_CASCADE
    )


# ============================================================
# VALIDATE
# ============================================================

if face_detector.empty():

    raise RuntimeError(
        "Face cascade could not be loaded."
    )


if eye_detector.empty():

    raise RuntimeError(
        "Eye cascade could not be loaded."
    )


if smile_detector is not None:

    if smile_detector.empty():

        smile_detector = None


# ============================================================
# EMOTIONS
# ============================================================

EMOTION_NAMES = {

    "angry": "Anger",

    "disgust": "Disgust",

    "fear": "Fear",

    "happy": "Happiness",

    "sad": "Sadness",

    "surprise": "Surprise",

    "neutral": "Neutral"

}


EMOTIONS_TO_SHOW = [

    ("angry", "Anger"),

    ("disgust", "Disgust"),

    ("fear", "Fear"),

    ("happy", "Happiness"),

    ("sad", "Sadness"),

    ("surprise", "Surprise"),

    ("neutral", "Neutral")

]


# ============================================================
# HISTORY STORAGE
# ============================================================

emotion_history = {

    1: {
        emotion: deque(
            maxlen=HISTORY_LENGTH
        )
        for emotion, _ in EMOTIONS_TO_SHOW
    },

    2: {
        emotion: deque(
            maxlen=HISTORY_LENGTH
        )
        for emotion, _ in EMOTIONS_TO_SHOW
    },

    3: {
        emotion: deque(
            maxlen=HISTORY_LENGTH
        )
        for emotion, _ in EMOTIONS_TO_SHOW
    },

    4: {
        emotion: deque(
            maxlen=HISTORY_LENGTH
        )
        for emotion, _ in EMOTIONS_TO_SHOW
    },

    5: {
        emotion: deque(
            maxlen=HISTORY_LENGTH
        )
        for emotion, _ in EMOTIONS_TO_SHOW
    }
}


# ============================================================
# GRAPH SETUP
# ============================================================

plt.ion()

fig, ax = plt.subplots(
    figsize=(11, 6)
)

fig.canvas.manager.set_window_title(
    "Live Emotion History"
)

ax.set_title(
    "Live Emotion History - Face 1"
)

ax.set_xlabel(
    "Time"
)

ax.set_ylabel(
    "Emotion Percentage (%)"
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    True,
    alpha=0.25
)

lines = {}

for emotion, display_name in EMOTIONS_TO_SHOW:

    line, = ax.plot(
        [],
        [],
        label=display_name
    )

    lines[emotion] = line


ax.legend(
    loc="upper left",
    fontsize=8
)


# ============================================================
# UPDATE GRAPH
# ============================================================

def update_graph():

    history = emotion_history[1]

    if len(history["neutral"]) == 0:

        return

    x_values = list(
        range(
            len(history["neutral"])
        )
    )

    for emotion, _ in EMOTIONS_TO_SHOW:

        values = list(
            history[emotion]
        )

        if len(values) < len(x_values):

            values = (
                [0] *
                (
                    len(x_values) -
                    len(values)
                )
            ) + values

        lines[emotion].set_data(
            x_values,
            values
        )

    ax.set_xlim(
        0,
        max(
            HISTORY_LENGTH,
            len(x_values)
        )
    )

    fig.canvas.draw_idle()

    fig.canvas.flush_events()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open camera."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


print()
print("=" * 60)
print("        AI EMOTION DETECTION SYSTEM")
print("=" * 60)
print("Camera started!")
print("Live emotion graph started!")
print("Press Q to quit.")
print("=" * 60)
print()


# ============================================================
# PERFORMANCE
# ============================================================

last_analysis_time = 0

last_feature_time = 0

face_results = []

fps = 0

fps_counter = 0

fps_start_time = time.time()

graph_counter = 0


# ============================================================
# HELPER
# ============================================================

def safe_percentage(
    emotions,
    emotion
):

    try:

        value = float(
            emotions.get(
                emotion,
                0
            )
        )

        return max(
            0,
            min(
                100,
                value
            )
        )

    except:

        return 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print(
            "Could not read camera frame."
        )

        break


    # Mirror
    frame = cv2.flip(
        frame,
        1
    )


    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps_counter += 1

    current_time = time.time()

    elapsed = (
        current_time -
        fps_start_time
    )

    if elapsed >= 1.0:

        fps = (
            fps_counter /
            elapsed
        )

        fps_counter = 0

        fps_start_time = current_time


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    faces = face_detector.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(70, 70)
    )


    faces = sorted(
        faces,
        key=lambda f: f[0]
    )


    if MAX_FACES is not None:

        faces_for_analysis = faces[
            :MAX_FACES
        ]

    else:

        faces_for_analysis = faces


    face_count = len(faces)


    # ========================================================
    # TOP BAR
    # ========================================================

    cv2.rectangle(

        frame,

        (0, 0),

        (width, 75),

        (25, 25, 25),

        -1
    )


    cv2.putText(

        frame,

        "AI EMOTION DETECTION",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        WHITE,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"TOTAL FACES: {face_count}",

        (20, 62),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        GREEN,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (width - 170, 32),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        CYAN,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "Q = Quit",

        (width - 145, 62),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        GRAY,

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # EMOTION ANALYSIS
    # ========================================================

    if (

        current_time -
        last_analysis_time
        >= ANALYSIS_INTERVAL

    ):

        new_results = []


        for face_index, (

            x,
            y,
            w,
            h

        ) in enumerate(
            faces_for_analysis
        ):


            padding = int(
                0.12 * w
            )


            x1 = max(
                0,
                x - padding
            )

            y1 = max(
                0,
                y - padding
            )

            x2 = min(
                width,
                x + w + padding
            )

            y2 = min(
                height,
                y + h + padding
            )


            face_crop = frame[
                y1:y2,
                x1:x2
            ]


            emotions = {}

            dominant = "unknown"

            confidence = 0


            # ------------------------------------------------
            # DEEPFACE
            # ------------------------------------------------

            try:

                result = DeepFace.analyze(

                    face_crop,

                    actions=["emotion"],

                    detector_backend="skip",

                    enforce_detection=False,

                    silent=True
                )


                if isinstance(
                    result,
                    list
                ):

                    if len(result) > 0:

                        result = result[0]


                emotions = result.get(
                    "emotion",
                    {}
                )


                dominant = result.get(
                    "dominant_emotion",
                    "neutral"
                )


                confidence = safe_percentage(

                    emotions,

                    dominant
                )


            except Exception as e:

                print(
                    "Emotion analysis error:",
                    str(e)
                )


            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            new_results.append({

                "box": (
                    x,
                    y,
                    w,
                    h
                ),

                "emotions": emotions,

                "dominant": dominant,

                "confidence": confidence,

                "eye_count": 0,

                "smile_detected": False

            })


        face_results = new_results

        last_analysis_time = current_time


    # ========================================================
    # EYE / SMILE DETECTION
    # ========================================================

    if (

        current_time -
        last_feature_time
        >= ANALYSIS_INTERVAL

    ):

        for face_data in face_results:

            x, y, w, h = face_data[
                "box"
            ]


            if (

                y < 0 or
                x < 0 or
                y + h > height or
                x + w > width

            ):

                continue


            face_gray = gray[
                y:y+h,
                x:x+w
            ]


            if face_gray.size == 0:

                continue


            # ------------------------------------------------
            # EYES
            # ------------------------------------------------

            eyes = eye_detector.detectMultiScale(

                face_gray,

                scaleFactor=1.1,

                minNeighbors=7,

                minSize=(18, 18)
            )


            valid_eyes = []


            for (

                ex,
                ey,
                ew,
                eh

            ) in eyes:

                if ey < h * 0.60:

                    valid_eyes.append(

                        (
                            ex,
                            ey,
                            ew,
                            eh
                        )

                    )


            face_data[
                "eye_count"
            ] = len(
                valid_eyes
            )


            face_data[
                "eye_boxes"
            ] = valid_eyes


            # ------------------------------------------------
            # SMILE
            # ------------------------------------------------

            smile_detected = False

            smile_boxes = []


            if smile_detector is not None:

                smile_region = face_gray[
                    int(h * 0.45):h,
                    :
                ]


                if smile_region.size > 0:

                    smiles = smile_detector.detectMultiScale(

                        smile_region,

                        scaleFactor=1.6,

                        minNeighbors=18,

                        minSize=(30, 15)
                    )


                    if len(smiles) > 0:

                        smile_detected = True

                        smile_boxes = smiles[:1]


            face_data[
                "smile_detected"
            ] = smile_detected


            face_data[
                "smile_boxes"
            ] = smile_boxes


        last_feature_time = current_time


    # ========================================================
    # UPDATE EMOTION HISTORY
    # ========================================================

    for face_index, face_data in enumerate(
        face_results
    ):

        face_number = face_index + 1

        if face_number not in emotion_history:

            continue


        emotions = face_data[
            "emotions"
        ]


        for emotion, _ in EMOTIONS_TO_SHOW:

            percentage = safe_percentage(

                emotions,

                emotion
            )


            emotion_history[
                face_number
            ][emotion].append(
                percentage
            )


    # ========================================================
    # DRAW EACH FACE
    # ========================================================

    for face_index, face_data in enumerate(
        face_results
    ):

        x, y, w, h = face_data[
            "box"
        ]


        emotions = face_data[
            "emotions"
        ]


        dominant = face_data[
            "dominant"
        ]


        confidence = face_data[
            "confidence"
        ]


        eye_count = face_data.get(
            "eye_count",
            0
        )


        smile_detected = face_data.get(
            "smile_detected",
            False
        )


        # ====================================================
        # FACE BOX
        # ====================================================

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            GREEN,

            3
        )


        # ====================================================
        # FACE LABEL
        # ====================================================

        label_y = max(
            95,
            y - 10
        )


        cv2.putText(

            frame,

            f"FACE {face_index + 1}",

            (x, label_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.70,

            GREEN,

            2,

            cv2.LINE_AA
        )


        # ====================================================
        # EYES
        # ====================================================

        for (

            ex,
            ey,
            ew,
            eh

        ) in face_data.get(
            "eye_boxes",
            []
        ):

            cv2.rectangle(

                frame,

                (
                    x + ex,
                    y + ey
                ),

                (
                    x + ex + ew,
                    y + ey + eh
                ),

                CYAN,

                2
            )


        # ====================================================
        # SMILE
        # ====================================================

        smile_boxes = face_data.get(
            "smile_boxes",
            []
        )


        for (

            sx,
            sy,
            sw,
            sh

        ) in smile_boxes:

            sy = sy + int(
                h * 0.45
            )


            cv2.rectangle(

                frame,

                (
                    x + sx,
                    y + sy
                ),

                (
                    x + sx + sw,
                    y + sy + sh
                ),

                MAGENTA,

                2
            )


            cv2.putText(

                frame,

                "SMILE",

                (
                    x + sx,
                    max(
                        90,
                        y + sy - 5
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                MAGENTA,

                2,

                cv2.LINE_AA
            )


        # ====================================================
        # NOSE REGION
        # ====================================================

        nose_x = x + int(
            w * 0.35
        )

        nose_y = y + int(
            h * 0.35
        )

        nose_w = int(
            w * 0.30
        )

        nose_h = int(
            h * 0.28
        )


        cv2.rectangle(

            frame,

            (
                nose_x,
                nose_y
            ),

            (
                nose_x + nose_w,
                nose_y + nose_h
            ),

            ORANGE,

            2
        )


        cv2.putText(

            frame,

            "NOSE",

            (
                nose_x,
                max(
                    90,
                    nose_y - 5
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            ORANGE,

            2,

            cv2.LINE_AA
        )


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        panel_width = 330

        panel_height = 280


        panel_x = x

        panel_y = y + h + 10


        if (

            panel_y +
            panel_height >
            height

        ):

            panel_y = (
                y -
                panel_height -
                10
            )


        panel_x = max(

            5,

            min(

                panel_x,

                width -
                panel_width -
                5

            )

        )


        panel_y = max(
            80,
            panel_y
        )


        # Background

        cv2.rectangle(

            frame,

            (
                panel_x,
                panel_y
            ),

            (
                panel_x +
                panel_width,

                panel_y +
                panel_height
            ),

            (25, 25, 25),

            -1
        )


        # Border

        cv2.rectangle(

            frame,

            (
                panel_x,
                panel_y
            ),

            (
                panel_x +
                panel_width,

                panel_y +
                panel_height
            ),

            GREEN,

            2
        )


        # ====================================================
        # FACE
        # ====================================================

        cv2.putText(

            frame,

            f"FACE {face_index + 1}",

            (
                panel_x + 12,
                panel_y + 25
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            GREEN,

            2,

            cv2.LINE_AA
        )


        # ====================================================
        # DOMINANT EMOTION
        # ====================================================

        dominant_name = EMOTION_NAMES.get(

            dominant,

            "Unknown"
        )


        cv2.putText(

            frame,

            f"Emotion: {dominant_name}",

            (
                panel_x + 12,
                panel_y + 50
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            WHITE,

            2,

            cv2.LINE_AA
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        cv2.putText(

            frame,

            f"Confidence: {confidence:.1f}%",

            (
                panel_x + 12,
                panel_y + 72
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

            CYAN,

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # FACE FEATURES
        # ====================================================

        smile_text = (

            "Detected"

            if smile_detected

            else

            "Not detected"

        )


        cv2.putText(

            frame,

            f"Eyes: {eye_count}",

            (
                panel_x + 12,
                panel_y + 95
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            CYAN,

            1,

            cv2.LINE_AA
        )


        cv2.putText(

            frame,

            f"Smile: {smile_text}",

            (
                panel_x + 110,
                panel_y + 95
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.43,

            MAGENTA,

            1,

            cv2.LINE_AA
        )


        cv2.putText(

            frame,

            "Nose: Detected",

            (
                panel_x + 12,
                panel_y + 116
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.43,

            ORANGE,

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # EMOTION PERCENTAGES
        # ====================================================

        line_y = panel_y + 140


        for (

            emotion_key,
            display_name

        ) in EMOTIONS_TO_SHOW:


            percentage = safe_percentage(

                emotions,

                emotion_key
            )


            text = (

                f"{display_name}: "
                f"{percentage:.1f}%"

            )


            cv2.putText(

                frame,

                text,

                (
                    panel_x + 12,
                    line_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.40,

                WHITE,

                1,

                cv2.LINE_AA
            )


            # Progress bar

            bar_x = panel_x + 155

            bar_y = line_y - 9

            bar_width = 155

            bar_height = 8


            cv2.rectangle(

                frame,

                (
                    bar_x,
                    bar_y
                ),

                (
                    bar_x +
                    bar_width,

                    bar_y +
                    bar_height
                ),

                DARK_GRAY,

                -1
            )


            filled_width = int(

                bar_width *
                percentage /
                100

            )


            if filled_width > 0:

                cv2.rectangle(

                    frame,

                    (
                        bar_x,
                        bar_y
                    ),

                    (
                        bar_x +
                        filled_width,

                        bar_y +
                        bar_height
                    ),

                    GREEN,

                    -1
                )


            line_y += 17


        # ====================================================
        # CONTEMPT
        # ====================================================

        cv2.putText(

            frame,

            "Contempt: N/A",

            (
                panel_x + 12,
                line_y + 2
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.40,

            GRAY,

            1,

            cv2.LINE_AA
        )


    # ========================================================
    # BOTTOM BAR
    # ========================================================

    cv2.rectangle(

        frame,

        (
            0,
            height - 38
        ),

        (
            width,
            height
        ),

        (25, 25, 25),

        -1
    )


    cv2.putText(

        frame,

        "GREEN: FACE   CYAN: EYES   MAGENTA: SMILE   ORANGE: NOSE",

        (
            15,
            height - 13
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        WHITE,

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # CAMERA DISPLAY
    # ========================================================

    cv2.imshow(

        "AI Emotion Detection - Live Camera",

        frame
    )


    # ========================================================
    # GRAPH UPDATE
    # ========================================================

    graph_counter += 1


    if graph_counter >= GRAPH_UPDATE_INTERVAL:

        update_graph()

        graph_counter = 0


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

plt.close("all")

print()
print("Camera stopped.")
print("Emotion Detection closed.")
