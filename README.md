# AI-Powered Accurate People Counter using YOLOv8 & ByteTrack

This project is an advanced computer vision application designed to count people in video streams with high precision while preventing duplicate counts. By leveraging the **YOLOv8m** object detection model integrated with the **ByteTrack** tracking algorithm, the system tracks the unique identities (IDs) of each individual. It calculates the total cumulative count without duplicating the same person when they move across or within the frame.

---

## 🚀 Key Features
- **High Accuracy:** Utilizes the medium-sized model (`yolov8m.pt`), providing an optimal balance between processing speed and high detection accuracy.
- **Intelligent Tracking:** Implements the `ByteTrack` algorithm to maintain a consistent ID for each person throughout their appearance.
- **Smart Filtering & Custom Confidence:** Configured with a strict confidence threshold (`conf=0.7`) to eliminate false positives, dedicated exclusively to the person class (`classes=[0]`).
- **Duplicate Prevention:** Accurately counts unique individuals by tracking and updating a set of unique continuous IDs.
- **Live Informative Display:** Real-time visual overlay showing both the current frame's unique person count and the cumulative total directly on the video stream.

---

## 🛠️ Prerequisites

Ensure you have **Python 3.8+** installed on your system, along with the following required libraries:

```bash
pip install opencv-python ultralytics