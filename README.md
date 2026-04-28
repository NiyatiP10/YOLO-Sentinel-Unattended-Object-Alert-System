# YOLOv9 Real-Time Object Detection & Unattended Object Alert System

This project implements a **real-time object detection system** using YOLOv9, with additional logic to detect **unattended objects** and generate **priority alerts** based on time duration.

It uses OpenCV for video processing and logs detections into JSON and CSV formats for further analysis.

---

## Features

* Real-time object detection using YOLOv9
* Detection of unattended objects
* Time-based priority alert system (P1, P2, P3)
* Live video feed with bounding boxes and labels
* Logging system:

  * JSON detection logs
  * CSV class counts
  * Terminal alert logs
* Configurable camera input via JSON

---

## Tech Stack

* Python
* OpenCV (`cv2`)
* Ultralytics YOLOv9
* Supervision library
* JSON / CSV logging

---

## Project Structure

```
├── main.py
├── config.json
├── yolov9e.pt
├── output.json
├── unattended_objects.json
├── class_counts.csv
├── terminal_output.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python supervision
```

---

## Usage

Run the script:

```bash
python main.py --config config.json
```

---

## Configuration

Edit `config.json`:

```json
{
  "cameras": [
    {
      "rtsp_url": "your_camera_stream_url"
    }
  ]
}
```

> Currently, the script uses webcam (`cv2.VideoCapture(0)`).
> You can replace it with RTSP stream if needed.

---

## How It Works

### Object Detection

* Uses YOLOv9 model (`yolov9e.pt`)
* Detects objects frame-by-frame

### Unattended Object Logic

* Certain object classes are tracked
* If an object remains without a person nearby for:

  * ≥10 sec → flagged
  * ≥20 sec → P2 alert
  * ≥30 sec → P1 alert

### Priority Levels

| Priority | Condition               |
| -------- | ----------------------- |
| P1       | ≥ 30 seconds unattended |
| P2       | ≥ 20 seconds unattended |
| P3       | ≥ 10 seconds unattended |

---

## Output Files

* `output.json` → All detections with timestamps
* `unattended_objects.json` → Unattended objects with priority
* `class_counts.csv` → Count of detected classes per timestamp
* `terminal_output.txt` → Priority alert logs

---

## Example Output

```
2026-04-29 12:00:10 - Priority Alert: P3
2026-04-29 12:00:20 - Priority Alert: P2
2026-04-29 12:00:30 - Priority Alert: P1
```

---

## Limitations

* Uses basic unattended logic (no object tracking IDs)
* May produce false positives in crowded scenes
* Webcam used by default instead of RTSP stream

---

## Future Improvements

* Add object tracking (DeepSORT / ByteTrack)
* Improve unattended detection logic
* Add GUI dashboard
* Cloud logging / alerts (Firebase, MQTT, etc.)
* Multi-camera support

---

## Contributing

Feel free to fork this repo and submit pull requests.


---

## Author

Niyati Patil (NiyatiP10)
