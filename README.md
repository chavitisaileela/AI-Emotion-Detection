# AI Emotion Detection

A real-time **AI-based facial emotion detection system** using Python, OpenCV, DeepFace, and TensorFlow. The application uses a webcam to detect multiple faces and display individual emotion predictions, emotion percentages, eye count, smile detection, nose region, confidence, FPS, and total face count directly on the camera interface.

## Project Overview

The **AI Emotion Detection** system is a computer vision application designed to analyze facial expressions in real time through a webcam.

The system can detect multiple faces simultaneously and provide individual analysis for each detected face.

### Features

* Real-time webcam emotion detection
* Multiple face detection
* Individual Face 1, Face 2, Face 3, etc. identification
* Total face count
* Individual emotion percentages
* Dominant emotion detection
* Emotion confidence percentage
* Eye detection and eye count
* Smile detection
* Nose region detection
* Real-time FPS display
* Live camera interface
* Mirror camera view
* Emotion percentage progress bars
* Optimized analysis interval for better performance

## Emotions Detected

The system analyzes the following facial emotion categories:

| Emotion  | Display Name          |
| -------- | --------------------- |
| Angry    | Anger                 |
| Disgust  | Disgust               |
| Fear     | Fear                  |
| Happy    | Happiness (Enjoyment) |
| Sad      | Sadness               |
| Surprise | Surprise              |
| Neutral  | Neutral               |

### Contempt

The current DeepFace emotion model does **not provide a separate contempt prediction**, so contempt is displayed as `N/A` rather than being falsely predicted.

## Facial Features

The application also provides visual facial feature detection:

### Face

* Detects multiple faces
* Draws a bounding box around each face
* Labels faces individually

Example:

```text
FACE 1
FACE 2
FACE 3
```

### Eyes

* Detects visible eyes using OpenCV Haar Cascade
* Displays eye bounding boxes
* Shows the detected eye count

### Smile

* Uses OpenCV smile detection
* Displays a smile bounding box when detected

### Nose

* Displays the approximate anatomical nose region of the detected face.

> Note: The nose region is represented using the approximate central facial region and is not claimed as a dedicated AI nose detector.

## Live Camera Information

The camera interface displays:

```text
AI EMOTION DETECTION

FACES: 3
FPS: 24.5
Q = Quit
```

For every detected face, the application displays information such as:

```text
FACE 1

Emotion: Happiness
Confidence: 98.4%

Eyes: 2
Smile: Detected
Nose: Detected

Anger:       0.1%
Disgust:     0.0%
Fear:        0.2%
Happiness:  98.4%
Sadness:     0.5%
Surprise:    0.3%
Neutral:     0.5%
```

## Technologies Used

### Programming Language

* Python

### Development Environment

* Visual Studio Code (VS Code)
* Windows
* Python Virtual Environment
* Git
* GitHub
* notepad

### Libraries and Frameworks

* OpenCV
* OpenCV-Contrib
* DeepFace
* TensorFlow
* NumPy

## Python Libraries

The main dependencies used in this project are:

```text
opencv-contrib-python==5.0.0.93
deepface
tensorflow
numpy
```

All required packages can be installed using:

```bash
pip install -r requirements.txt
```

## Computer Vision

OpenCV is used for:

* Webcam access
* Real-time video processing
* Face detection
* Eye detection
* Smile detection
* Image preprocessing
* Drawing detection boxes
* FPS calculation

The project uses OpenCV Haar Cascade classifiers including:

```text
haarcascade_frontalface_default.xml
haarcascade_eye.xml
haarcascade_smile.xml
```

## Artificial Intelligence

DeepFace is used for facial emotion analysis.

The system extracts the facial region detected by OpenCV and sends it to the DeepFace emotion model.

The model returns emotion scores for:

```text
Anger
Disgust
Fear
Happiness
Sadness
Surprise
Neutral
```

The emotion with the highest score is displayed as the dominant emotion.

## TensorFlow

TensorFlow is used as the machine learning framework supporting the DeepFace emotion recognition model.

The required model weights are downloaded automatically by DeepFace when they are needed for the first time.

## NumPy

NumPy is used internally for numerical and image-array processing.

## Project Structure

```text
AI-Emotion-Detection/
│
├── emotion_detection.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Important

The `venv` folder is intentionally **not uploaded to GitHub**.

A virtual environment is system-specific and can be recreated using the project requirements.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chavitisaileela/AI-Emotion-Detection.git
```

### 2. Open the Project

```bash
cd AI-Emotion-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python emotion_detection.py
```

The webcam window should open automatically.

Press:

```text
Q
```

to close the application.

## Requirements

### Hardware

* Windows computer
* Webcam
* Internet connection for the first-time DeepFace model download

### Software

* Python
* Visual Studio Code
* Git
* GitHub

## Performance

The application uses an emotion analysis interval to reduce the number of DeepFace predictions performed per second.

This helps balance:

```text
Detection Accuracy
        +
Processing Speed
        =
Better Real-Time Performance
```

OpenCV handles continuous face detection while DeepFace performs emotion analysis at controlled intervals.

## Camera Interface

The application provides a live interface containing:

* Total number of faces
* Face bounding boxes
* Individual face labels
* Dominant emotion
* Emotion confidence
* Emotion percentages
* Emotion progress bars
* Eye count
* Smile status
* Nose region
* FPS
* Keyboard control

## Color Indicators

```text
GREEN    → Face
CYAN     → Eyes
MAGENTA  → Smile
ORANGE   → Nose
```

## How the System Works

```text
Webcam
   ↓
Capture Video Frame
   ↓
Mirror Frame
   ↓
Convert to Grayscale
   ↓
Face Detection
   ↓
Detect Individual Faces
   ↓
Extract Face Region
   ↓
DeepFace Emotion Analysis
   ↓
Calculate Emotion Percentages
   ↓
Eye Detection
   ↓
Smile Detection
   ↓
Display Results
   ↓
Live Camera Output
```

## Use Cases

This project can be used for educational and experimental purposes in:

* Computer Vision
* Artificial Intelligence
* Facial Expression Recognition
* Human-Computer Interaction
* AI-based user interfaces
* Real-time image processing
* Machine Learning demonstrations
* Academic projects

## Limitations

Emotion recognition is an AI-based estimation and should not be interpreted as a definitive measurement of a person's actual emotional state.

Performance can vary depending on:

* Lighting conditions
* Camera quality
* Face orientation
* Distance from the camera
* Facial visibility
* Multiple faces
* Processing hardware

Eye and smile detection using Haar Cascades may also produce occasional false detections.

## Future Improvements

Possible future improvements include:

* Emotion history graphs
* Emotion trend visualization
* Better face tracking
* Advanced facial landmark detection
* Improved nose and facial feature detection
* GPU acceleration
* Web-based interface
* Flask integration
* Three.js visualization
* Emotion data logging
* CSV export
* Real-time analytics dashboard

## Development Platform

This project was developed using:

```text
Operating System : Windows
IDE              : Visual Studio Code
Language         : Python
Version Control  : Git
Repository       : GitHub
Camera           : Webcam
```

## Author

**Chaviti Sai Leela**

GitHub:

https://github.com/chavitisaileela

## Repository

**AI Emotion Detection**

https://github.com/chavitisaileela/AI-Emotion-Detection

## License

This project is intended for educational and academic purposes.
