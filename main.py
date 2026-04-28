import cv2
import argparse
import supervision as sv
import json
import csv
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict


# Helper Functions
def check_unattended_objects(unattended_objects, class_name, timestamp):
    if class_name in unattended_objects:
        for obj in unattended_objects[class_name]:
            duration = (timestamp - obj['last_seen']).seconds
            if duration >= 10:
                return True
    return False


def get_priority(unattended_objects):
    if len(unattended_objects) == 0:
        return None

    last_seen_times = [
        obj['last_seen']
        for objs in unattended_objects.values()
        for obj in objs
    ]

    if not last_seen_times:
        return None

    last_seen = max(last_seen_times)
    time_difference = (datetime.now() - last_seen).seconds

    if time_difference >= 30:
        return "P1"
    elif time_difference >= 20:
        return "P2"
    elif time_difference >= 10:
        return "P3"

    return None


def process_frame(frame, model, box_annotator):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Draw bounding boxes (no labels param to avoid error)
    annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    # Manually draw labels (works in all versions)
    for box, class_id, confidence in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.model.names[class_id]} {confidence:.2f}"

        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return annotated_frame, detections


def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    return None


# Main Function
def main(model):
    output_data = []
    class_counts = defaultdict(dict)
    unattended_objects = defaultdict(list)
    terminal_output = []
    cap = None

    try:
        args = parse_arguments()
        config = read_config(args.config)

        camera = config['cameras'][0]
        rtsp_url = camera['rtsp_url']

        print("Opening camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("Setting up annotator...")
        box_annotator = sv.BoxAnnotator(thickness=2)

        UNATTENDED_CLASSES = list(range(14, 56)) + list(range(62, 80))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            timestamp = datetime.now()

            annotated_frame, detections = process_frame(
                frame, model, box_annotator
            )

            cv2.imshow("YOLOv9 Detection", annotated_frame)

            data = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "detections": []
            }

            for class_id, confidence in zip(detections.class_id, detections.confidence):
                class_name = model.model.names[class_id]

                data["detections"].append({
                    "class_name": class_name,
                    "confidence": float(confidence)
                })

                ts = data["timestamp"]
                if ts not in class_counts:
                    class_counts[ts] = {}

                class_counts[ts][class_name] = class_counts[ts].get(class_name, 0) + 1

                # Unattended logic
                if class_id in UNATTENDED_CLASSES:
                    unattended = check_unattended_objects(
                        unattended_objects,
                        class_name,
                        timestamp
                    )

                    if not unattended:
                        unattended_objects[class_name].append({
                            "last_seen": timestamp
                        })

            output_data.append(data)

            # Remove unattended if person detected
            detected_classes = [d["class_name"] for d in data["detections"]]

            if "person" in detected_classes:
                for class_name in unattended_objects:
                    unattended_objects[class_name] = [
                        obj for obj in unattended_objects[class_name]
                        if (timestamp - obj["last_seen"]).seconds < 10
                    ]

            # Priority
            priority = get_priority(unattended_objects)
            if priority:
                msg = f"{timestamp} - Priority Alert: {priority}"
                print(msg)
                terminal_output.append(msg)

            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Safe release
        if cap is not None and cap.isOpened():
            cap.release()

        cv2.destroyAllWindows()

        # Save logs
        try:
            with open("terminal_output.txt", "w") as f:
                f.write("\n".join(terminal_output))
        except:
            print("Error saving terminal output")

        try:
            unattended_with_priority = {
                class_name: {
                    "objects": objs,
                    "priority": get_priority({class_name: objs})
                }
                for class_name, objs in unattended_objects.items()
            }

            with open("unattended_objects.json", "w") as f:
                json.dump(unattended_with_priority, f, indent=4, default=default_serializer)
        except:
            print("Error saving unattended objects")

        try:
            with open("output.json", "w") as f:
                json.dump(output_data, f, indent=4)
        except:
            print("Error saving output data")

        try:
            with open("class_counts.csv", "w", newline="") as csvfile:
                fieldnames = ["Timestamp", "Class Name", "Count"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for ts, counts in class_counts.items():
                    for cname, count in counts.items():
                        writer.writerow({
                            "Timestamp": ts,
                            "Class Name": cname,
                            "Count": count
                        })
        except:
            print("Error saving CSV")


# Config & Args
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv9 Live Detection")
    parser.add_argument("--config", default="config.json")
    return parser.parse_args()


def read_config(path):
    with open(path, "r") as f:
        return json.load(f)


# Entry Point
if __name__ == "__main__":
    model = YOLO("yolov9e.pt")
    main(model)