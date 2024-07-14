# Real-Time Object Detection App

This application performs real-time object detection using a webcam feed. It leverages the YOLO (You Only Look Once) model for object detection and OpenCV for capturing and displaying video frames.

## Features

- Real-time object detection using YOLOv8.
- Displays bounding boxes around detected objects with confidence scores and class names.
- Supports a wide range of object classes.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/repo-name.git
    cd repo-name
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your webcam is connected and working.

2. Run the script:

    ```bash
    python main.py
    ```

3. The application will start the webcam feed and display detected objects in real-time.

## Code Overview

Here's a brief explanation of the main components of the script:

- **Import necessary libraries**: The script imports `YOLO` from `ultralytics` and `cv2` for OpenCV functionality.
  
    ```python
    from ultralytics import YOLO
    import cv2
    import math 
    ```

- **Initialize webcam**: The script starts the webcam and sets the resolution to the highest possible.

    ```python
    HIGH_VALUE = 10000
    WIDTH = HIGH_VALUE
    HEIGHT = HIGH_VALUE
    
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ```

- **Load YOLO model**: The script loads the YOLO model with specified weights.

    ```python
    model = YOLO("yolo-Weights/yolov8n.pt")
    ```

- **Define object classes**: The script defines the list of class names that YOLO can detect.

    ```python
    classNames = ["person", "bicycle", "car", ... "toothbrush"]
    ```

- **Main loop for real-time detection**: The script continuously captures frames from the webcam, performs object detection, and displays the results with bounding boxes, confidence scores, and class names.

    ```python
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, f'{classNames[cls]} {confidence}', org, font, 0.6, (255, 255, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.