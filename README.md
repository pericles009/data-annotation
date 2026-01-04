[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/PericlesRodrigues01/player-detector)

# ⚽ Player and Ball Detection with YOLOv8

This Computer Vision project leverages Deep Learning to detect players, balls, and sports equipment in match footage.

The primary challenge of this project was dealing with a **restricted dataset** (a low volume of images). Consequently, the development focused heavily on **Data Engineering** and advanced **Data Augmentation** strategies to mitigate overfitting and ensure the model could generalize effectively.

## 🎯 The Data Annotation Process

The foundation of supervised learning lies in the quality of the annotations. For this project, the dataset was manually labeled using **[CVAT (Computer Vision Annotation Tool)](https://github.com/cvat-ai/cvat)**, an industry-standard open-source tool.

20 images were carefully annotated by drawing *Bounding Boxes* around each object of interest and exporting annotations directly in YOLO format.

The defined classes were:
1.  **Person** (Players)
2.  **Ball** (Football/tennis ball)
3.  **Equipment** (Tennis racket)

<img width="909" alt="Exemple of the annotation process" src="https://github.com/user-attachments/assets/6f02758a-90d8-48d6-9612-b6545c46fda5" />

*Figure 1: Visual example of the manual annotation process performed on the dataset.*

### How the Model "Reads" the Annotations
The YOLO model does not "see" the image as we do. It interprets text files (`.txt`) associated with each image. Each line represents an object in the following format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

For example, a line reading 0 0.552 0.586 0.117 0.636 teaches the model that:
0: It is a person (Class ID 0).
0.552 / 0.586: The center of the player is approximately in the middle of the screen (normalized coordinates from 0 to 1).



## 🚀 Data Augmentation Strategy
Due to the scarcity of original data, the initial model suffered from Overfitting (memorizing training images and failing on new ones). To solve this, a hybrid data augmentation strategy was implemented:

1. Offline Augmentation (Physical)
A custom Python script was developed to manipulate the raw files before training, effectively tripling the dataset size:

Horizontal Flip: In football, the visual tactics remain consistent even if the field is mirrored. This doubled the data while maintaining semantic integrity.

Contrast Adjustment: Simulated different lighting conditions (e.g., sunny vs. cloudy matches).

Note: The script mathematically recalculated the bounding box coordinates (labels) to ensure they perfectly matched the transformed images.

2. Online Augmentation (On-the-fly)
During training, YOLO was configured to apply random transformations in memory at every epoch:

Mosaic: Combines 4 images into a single training sample (forces the model to detect objects in complex contexts).

Rotation & Scale: Slight rotations and zooming to simulate varying camera distances and angles.

## 📊 Results & Post-Processing
The final model successfully overcame the "null detection" issue present in earlier iterations. To refine the output, NMS (Non-Maximum Suppression) was applied with an IoU=0.4 threshold, eliminating duplicate bounding boxes and ensuring each player is detected only once with the highest confidence score.
0.117 / 0.636: These are the proportional dimensions of the box relative to the total image size.

<img width="909" alt="Result example" src="https://github.com/user-attachments/assets/02b344b6-ebf0-4ad1-95ee-b7f537598d6a" />

*Figure 2: Result example showing detections on a test image.*

## 🛠 Technologies Used
Python

CVAT (Computer Vision Annotation Tool)

YOLOv8 (Ultralytics)

OpenCV (Image processing and dataset manipulation)

Google Colab (GPU training environment)
