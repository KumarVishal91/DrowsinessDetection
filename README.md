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

## Dependencies

Install required packages:

```powershell
python -m pip install -r requirements.txt
```


## Usage

Run the app with the webcam index (default 0):

```powershell
python drowsiness_yawn.py --webcam 0 --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm "C:\path\to\Alert.WAV"
```

Parameters:
- `--webcam`, `-w`: Webcam index (default `0`).
- `--cascade`, `-c`: Cascade XML file path (default `haarcascade_frontalface_default.xml`).
- `--shape-predictor`, `-p`: dlib model path (default `shape_predictor_68_face_landmarks.dat`).
- `--alarm`, `-a`: Optional alarm WAV path. Leave blank to disable sound.

Example without sound:

```powershell
python drowsiness_yawn.py --webcam 0 --alarm ""
```

## Keyboard controls

- Press `q` to quit the window and stop the app.

## Troubleshooting

- If you get `ModuleNotFoundError: No module named 'scipy'`: check dependencies and re-run pip install command.
- If `dlib` fails to compile on Windows, use a prebuilt wheel.
- If `shape_predictor_68_face_landmarks.dat` is missing, the script will raise an error and prompt for the correct file.

## Notes

- If using in a production or safety-critical environment, build a more robust pipeline and avoid using the webcam-based alert as the only safety mechanism.
- Adjust thresholds in `drowsiness_yawn.py` as needed:
  - `EYE_AR_THRESH` (default `0.3`),
  - `EYE_AR_CONSEC_FRAMES` (default `30`),
  - `YAWN_THRESH` (default `20`).

## License

This project is provided as-is for learning and experimentation.
