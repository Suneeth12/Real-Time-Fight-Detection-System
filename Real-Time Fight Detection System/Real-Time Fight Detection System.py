import cv2
import numpy as np
from scipy.spatial import distance as dist

# -------------------------------
# Centroid Tracker Class
# -------------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        self.nextObjectID = 0
        self.objects = {}       # objectID -> current centroid
        self.disappeared = {}   # objectID -> consecutive frames missing
        self.tracks = {}        # objectID -> list of centroids (path)
        self.boxes = {}         # objectID -> current bounding box (startX, startY, endX, endY)
        self.fighting = {}      # objectID -> persistent fighting state (True/False)
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, box):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.tracks[self.nextObjectID] = [centroid]
        self.boxes[self.nextObjectID] = box
        self.fighting[self.nextObjectID] = False
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.tracks[objectID]
        del self.boxes[objectID]
        del self.fighting[objectID]

    def update(self, rects):
        """
        rects: list of bounding boxes as (startX, startY, endX, endY)
        """
        if len(rects) == 0:
            # Mark all objects as disappeared.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute centroids for each bounding box.
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If no objects are tracked, register each detection.
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Find the smallest distance pairings.
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.tracks[objectID].append(inputCentroids[col])
                self.boxes[objectID] = rects[col]
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Mark unmatched existing objects as disappeared.
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])
        return self.objects

# -------------------------------
# Process a Frame with Detection Interval, Persistent Fighting State, & Average Distance
# -------------------------------
def process_frame(frame, net, output_layers, ct, classes,
                  start_fighting_threshold=100, persist_fighting_threshold=300,
                  run_detection=True, prev_rects=None):
    height, width = frame.shape[:2]

    # Run YOLO detection if flagged; otherwise, reuse previous detections.
    if run_detection:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        rects = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                rects.append((x, y, x + w, y + h))
        last_rects = rects  # Save current detections for future frames.
    else:
        if prev_rects is None:
            return process_frame(frame, net, output_layers, ct, classes,
                                 start_fighting_threshold, persist_fighting_threshold,
                                 run_detection=True)
        rects = prev_rects
        last_rects = prev_rects

    # Update tracker.
    objects = ct.update(rects)
    objectIDs = list(objects.keys())
    centroids = list(objects.values())

    # --- Update Fighting State ---
    # Mark objects as fighting if any two are very close.
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance_ij = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
            if distance_ij < start_fighting_threshold:
                ct.fighting[objectIDs[i]] = True
                ct.fighting[objectIDs[j]] = True

    # For objects already flagged as fighting, only unflag them
    # if they are far (beyond the persist threshold) from every other fighting object.
    for idx, objectID in enumerate(objectIDs):
        if ct.fighting[objectID]:
            still_close = False
            for jdx, otherID in enumerate(objectIDs):
                if otherID != objectID and ct.fighting[otherID]:
                    if np.linalg.norm(np.array(centroids[idx]) - np.array(centroids[jdx])) < persist_fighting_threshold:
                        still_close = True
                        break
            if not still_close:
                ct.fighting[objectID] = False

    # --- Calculate Average Distance between all Persons ---
    avg_distance = 0.0
    if len(centroids) > 1:
        centroids_arr = np.array(centroids)
        dists = dist.cdist(centroids_arr, centroids_arr)
        # Only consider the upper triangle (excluding the diagonal) to avoid duplicate distances.
        triu_indices = np.triu_indices(len(centroids_arr), k=1)
        if len(triu_indices[0]) > 0:
            avg_distance = np.mean(dists[triu_indices])

    # --- Draw Overlays ---
    for idx, objectID in enumerate(objectIDs):
        (startX, startY, endX, endY) = ct.boxes[objectID]
        color = (0, 0, 255) if ct.fighting[objectID] else (0, 255, 0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cX, cY = objects[objectID]
        cv2.circle(frame, (cX, cY), 4, color, -1)
        label = "FIGHTING" if ct.fighting[objectID] else "NON-FIGHTING"
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pts = ct.tracks[objectID]
        for k in range(1, len(pts)):
            cv2.line(frame, tuple(pts[k - 1]), tuple(pts[k]), color, 2)

    # Display the average distance on the frame.
    cv2.putText(frame, f"Avg Pixel Distance: {avg_distance:.2f}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame, last_rects

# -------------------------------
# Main Function with Video Controls & Detection Interval
# -------------------------------
def main():
    # Update file paths as needed.
    config_path = r"C:\Users\saiku\OneDrive\Desktop\DV Lab\Project\yolov3.cfg"
    weights_path = r"C:\Users\saiku\OneDrive\Desktop\DV Lab\Project\yolov3.weights"
    classes_file = r"C:\Users\saiku\OneDrive\Desktop\DV Lab\Project\coco.names"
    video_path = r"C:\Users\saiku\OneDrive\Desktop\DV Lab\Project\videos\OWN dataset.mp4"

    # Load class labels.
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load YOLOv3 network.
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # Enable GPU acceleration if available.
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    # Get output layer names.
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Choose camera or video file.
    choice = input("Enter 'c' for camera or 'v' for video file: ").strip().lower()
    if choice == 'c':
        cap = cv2.VideoCapture(0)
        videoControls = False
        start_fighting_threshold = 150
    else:
        cap = cv2.VideoCapture(video_path)
        videoControls = True
        start_fighting_threshold = 100
        paused = False
        skipFrames = 30  # Frames to skip on forward/backward

    ct = CentroidTracker(maxDisappeared=40)
    detection_interval = 3  # Run detection every 3 frames.
    frame_count = 0
    prev_rects = None

    print("Video Controls:")
    if videoControls:
        print("  Space: Pause/Play")
        print("  f    : Forward")
        print("  b    : Backward")
    print("  ESC  : Exit")

    while True:
        if not videoControls or not paused:
            ret, frame = cap.read()
            if not ret:
                break
            run_detection = (frame_count % detection_interval == 0)
            frame, prev_rects = process_frame(frame, net, output_layers, ct, classes,
                                                start_fighting_threshold=start_fighting_threshold,
                                                persist_fighting_threshold=300,
                                                run_detection=run_detection, prev_rects=prev_rects)
            cv2.imshow("Fight Detection", frame)
            frame_count += 1

        if videoControls:
            key = cv2.waitKey(1) & 0xFF  # minimal delay for natural pace
            if key == 27:  # ESC to exit.
                break
            elif key == ord(' '):  # Toggle pause/play.
                paused = not paused
            elif key == ord('f'):  # Skip forward.
                currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame + skipFramecs)
                if paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame, prev_rects = process_frame(frame, net, output_layers, ct, classes,
                                                        start_fighting_threshold=start_fighting_threshold,
                                                        persist_fighting_threshold=300,
                                                        run_detection=True)
                    cv2.imshow("Fight Detection", frame)
            elif key == ord('b'):  # Skip backward.
                currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                newPos = max(currentFrame - skipFrames, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, newPos)
                if paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame, prev_rects = process_frame(frame, net, output_layers, ct, classes,
                                                        start_fighting_threshold=start_fighting_threshold,
                                                        persist_fighting_threshold=300,
                                                        run_detection=True)
                    cv2.imshow("Fight Detection", frame)
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
