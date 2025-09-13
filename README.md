# Real-Time Fight Detection System

A Python-based real-time solution for automatic detection and tracking of fights or violent incidents in surveillance video feeds. Utilizes YOLO for deep learning-based person detection, centroid tracking, and spatial analysis for robust detection in diverse scenarios.

## Features

- **YOLO Person Detection**: Fast, accurate detection in real-time streams using a deep learning-based object detector.
- **Centroid Tracking**: Maintains consistent object identities and tracks movement.
- **Fight State Logic**: Flags "fighting" based on proximity and duration, minimizing false positives.
- **Flexible Input**: Webcam (`c`) and video file (`v`) modes with playback controls.
- **Live Visual Overlays**: Bounding boxes, identities, "FIGHTING"/"NON-FIGHTING" status, and path trails.

## Technologies Used

- Python 3.x
- OpenCV
- YOLO (object detection)
- NumPy
- SciPy
