:: Custom Object Detection for Inclined-Angle Drone Footage using YOLOv8 🚀

![alt text](https://img.shields.io/badge/Project-Custom%20Object%20Detection-blue)

![alt text](https://img.shields.io/badge/Python-3.9+-blue.svg)

![alt text](https://img.shields.io/badge/Framework-YOLOv8-red)

![alt text](https://img.shields.io/badge/License-MIT-green)

This project delivers a complete workflow for training a custom YOLOv8 model to perform object detection on video captured from a low, inclined-angle drone perspective. By creating a bespoke dataset and leveraging aggressive data augmentation, we successfully developed a lightweight model capable of accurately identifying key objects from this unique and challenging viewpoint.
📝 Table of Contents

    Problem Statement

    Our Solution

    Methodology

        Model Selection

        Dataset Creation & Augmentation

        Data Structure

    Getting Started

        Prerequisites

        Training the Model

        Running Detection

    Results & Performance

    Conclusion & Future Work

1. Problem Statement

The primary goal was to perform object detection on video frames to identify key classes: man, tree, bush, building, and vehicle. The core challenge stemmed from the video footage, which was recorded by a drone at a low, inclined angle. Standard object detection models are typically pre-trained on ground-level or high-altitude, top-down imagery and thus perform poorly on this intermediate, angled perspective. This gap necessitated the creation and training of a custom model tailored specifically to our data.
2. Our Solution

To address this challenge, we developed a custom object detection model using YOLOv8. A significant portion of the project involved building a bespoke dataset from video frames. Through an iterative process of manual annotation, semi-automated labeling with Roboflow, and extensive data augmentation, we trained a lightweight YOLOv8n model capable of accurately identifying objects from this unique viewpoint.
3. Methodology
3.1. Model Selection

Due to hardware constraints and the potential for real-time, on-device deployment, the YOLOv8n (nano) model was the ideal choice. It offers an excellent trade-off between inference speed and accuracy, making it perfectly suited for applications on resource-constrained devices like drones.
3.2. Dataset Creation & Augmentation

The unavailability of a suitable public dataset was a major hurdle. Our dataset creation process was iterative and methodical:

    Initial Dataset (V1):

        Extracted frames from the source videos.

        Manually annotated approximately 250 images in Roboflow.

        Used the initial model as a "Roboflow bot" to semi-automate the annotation of the remaining images, ensuring a diverse and balanced dataset.

    Augmented Dataset (V2):

        To combat overfitting and improve generalization, we applied a suite of data augmentation techniques, including tilt, rotation, translation, noise, and blur.

        We also annotated additional images to further enrich the dataset. This more robust, augmented dataset was used for final model training.

3.3. Data Structure

The dataset was organized in the standard YOLO format to ensure compatibility with the training pipeline. The project was managed using a data.yaml file to define class names and directory paths.

/dataset
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
4. Getting Started
4.1. Prerequisites

Ensure you have Python 3.8+ and the following libraries installed:

    ultralytics

    opencv-python

You can install them using pip: pip install ultralytics opencv-python
4.2. Training the Model

The train.py script handles the training of the custom YOLOv8 model. It loads a pre-trained yolov8n.pt checkpoint and fine-tunes it on the custom dataset defined in data.yaml.

# train.py
from ultralytics import YOLO

# --- 1. Load a Pre-trained Model ---
# Load the YOLOv8 nano model, which is lightweight and fast.
model = YOLO('yolov8n.pt')

# --- 2. Train the Model on a Custom Dataset ---
# The 'data.yaml' file contains paths to the training/validation data and class names.
results = model.train(
   data='data.yaml',
   epochs=100,
   imgsz=640,
   batch=16,
   name='yolov8_custom',
   workers=4
)

print("✅ Training complete.")

4.3. Running Detection

The detect.py script uses the best-performing trained weights to perform inference on a given video file (input.mp4). It generates an output video with bounding box overlays (output_video.mp4) and a detailed detections_log.txt file.

# detect.py
import cv2
from ultralytics import YOLO

# Load the best performing model weights
model = YOLO("runs/detect/yolov8_custom/weights/best.pt")
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Video writer and log file setup
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
log_file = open("detections_log.txt", "w")

print("⏳ Starting detection on video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    detections = []
    
    # Process and log detections
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy)
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detections.append(f"{label} ({conf:.2f})")

    # Write to log file
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    log_entry = f"Frame {frame_number}, Time {timestamp:.2f}s: "
    log_entry += ", ".join(detections) if detections else "No objects detected"
    log_file.write(log_entry + "\n")

    # Write frame to output video
    out.write(annotated_frame)
    
    # Display the output
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("✅ Detection and logging complete.")
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

5. Results & Performance

The final model, trained on our augmented V2 dataset, achieved a peak mean Average Precision (mAP50) of 48%. This confirms that the approach is viable and that the model successfully learned to identify key objects from the challenging inclined-angle perspective.
![Fig:1](Images/confusion_matrix_normalized.png)
Figure 1: Normalized Confusion Matrix

The confusion matrix below highlights the model's performance on the validation set. The diagonal values represent correct predictions. Off-diagonal values reveal areas for improvement, such as the model sometimes misidentifying 'tree' as 'bush'. This provides clear, actionable guidance for future dataset enhancements.

![Fig:2](Images/train_batch2720.jpg)
Figure 2: Sample Detections

Example output from the trained YOLOv8n model on test images, demonstrating successful detections of vehicles, people, and vegetation.


6. Conclusion & Future Work

This project successfully establishes a complete workflow and a crucial performance baseline for object detection on inclined-angle drone footage. The lightweight YOLOv8n model proved to be an effective proof-of-concept, though its performance was ultimately constrained by its limited parameter count.

Future work will focus on:

    Training Larger Models: Migrating to more powerful models like YOLOv8m or YOLOv8l on high-performance GPUs.

    Continued Dataset Expansion: Further enriching the dataset with more varied examples and addressing class confusion identified in the confusion matrix.

This data-driven path is projected to significantly increase detection accuracy and lead to a production-ready system for real-world drone applications.
