# Real-Time Drowsiness Detection System

A simple Python-based computer vision system that detects drowsiness (eye closure) and yawning from a webcam stream and raises alerts. 

## Features

- Detects face and facial landmarks in real time.
- Computes Eye Aspect Ratio (EAR) to identify prolonged eye closure (drowsiness).
- Computes lip distance to identify yawning.
- Displays annotated video frames with EAR/Yawn metrics.
- Optional alarm sound on alert (if a valid WAV file path is provided).

## Repository contents

- `drowsiness_yawn.py` - Main application script.
- `haarcascade_frontalface_default.xml` - OpenCV pre-trained Haar cascade for face detection.
- `requirements.txt` - Project Python dependency list.
- `README.md` - This file.

## Prerequisites

- Windows, macOS, or Linux machine with webcam.
- Python 3.8–3.11 (recommended). Python 3.12+ may have compatibility issues with older binary packages.
- pip and virtual environment tooling.


## Project structure

- `drowsiness_yawn.py` - Main script implementing the camera loop, face detection, EAR/yawn calculation, alert display and sound thread.
- `haarcascade_frontalface_default.xml` - OpenCV Haar cascade for face detection.
- `shape_predictor_68_face_landmarks.dat` (not in repo) - dlib facial landmarks model (download from dlib model zoo).
- `requirements.txt` - pinned versions of libraries.
- `README.md` - this file.

## dependencies (requirements.txt)

`requirements.txt` contains:

- `dlib==19.16.0` - face landmark detector (dlib shape predictor). Required for `shape_predictor_68_face_landmarks.dat` and landmark extraction.
- `imutils==0.5.2` - convenience utilities for image resizing and video stream handling (`imutils.video.VideoStream`, `imutils.resize`).
- `numpy==1.15.4` - numerical array operations for EAR/yawn distance math.
- `opencv-python==4.0.0.21` - video capture, image processing and drawing APIs (face detection cascade, rectangles, contours, text overlay).
- `playsound==1.2.2` - simple cross-platform audio playback for alert sound if provided.
- `scipy==1.2.0` - spatial distance for Euclidean distance computation in EAR.
  
## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10.11 | Core language |
| OpenCV | 4.x | Video capture & image processing |
| dlib-bin | 20.0.1 | 68-point facial landmark detection |
| imutils | 0.5.4 | Video stream utilities |
| scipy | Latest | Euclidean distance (EAR calculation) |
| numpy | Latest | Array operations |
| playsound | 1.2.2 | Audio alarm playback |

## Setup and Run

1. Clone repo:

```bash
git clone https://github.com/KumarVishal91/DrowsinessDetection.git
cd DrowsinessDetection
```

2. (Optional but recommended) Create and activate venv:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```


5. Run the app:

```powershell
python drowsiness_yawn.py
```

6. Quit app:

- Press `q` in the display window.


## Usage

- The app uses `dlib` and OpenCV to detect the face and facial landmarks.
- `EAR` is computed with `scipy.spatial.distance.euclidean`.
- Eyes closed for 30 frames triggers drowsiness alert.
- Lip-opening distance > 20 triggers yawn alert.
- `playsound` asserts sound when `--alarm` path is valid.

## Keyboard controls

- Press `q` to quit the window and stop the app.

## License

This project is provided as-is for learning and experimentation.
