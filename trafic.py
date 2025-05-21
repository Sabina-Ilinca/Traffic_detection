import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import os

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pixel_to_meter = 0.05

    model = YOLO('yolov8n.pt')
    track_history = defaultdict(lambda: {'positions': [], 'speed': 0, 'zone': 'Unknown', 'class': -1})

    green_normal_sec = 10
    green_extended_sec = 15

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[0, 2, 3, 5, 7],  # person, car, motorcycle, bus, truck
            tracker="bytetrack.yaml"
        )

        if not results:
            continue

        annotated_frame = results[0].plot()

        zone_counts = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        zone_speeds = {'N': [], 'S': [], 'E': [], 'W': []}
        people = []
        vehicles = []
        motorcyclists = []

        pedestrian_near_crossing = False
        pedestrian_crossing = False

        if results[0].boxes.id is not None:
            for box, obj_id, cls in zip(
                results[0].boxes.xywh.cpu(),
                results[0].boxes.id.int().cpu().tolist(),
                results[0].boxes.cls.int().cpu().tolist()
            ):
                x, y, w, h = box
                center = (float(x), float(y))

                if x < width / 2 and y < height / 2:
                    zone = 'N'
                elif x >= width / 2 and y < height / 2:
                    zone = 'E'
                elif x < width / 2 and y >= height / 2:
                    zone = 'W'
                else:
                    zone = 'S'

                track = track_history[obj_id]
                track['positions'].append(center)
                track['zone'] = zone
                track['class'] = cls
                if len(track['positions']) > 10:
                    track['positions'].pop(0)

                if len(track['positions']) >= 2:
                    dist = np.linalg.norm(
                        np.array(track['positions'][-1]) -
                        np.array(track['positions'][-2])
                    )
                    speed_kph = dist * pixel_to_meter * fps * 3.6
                    track['speed'] = speed_kph
                    zone_speeds[zone].append(speed_kph)

                if cls == 0:  # Person
                    people.append((center, obj_id))
                    if y > height * 0.85:
                        pedestrian_near_crossing = True

                        if len(track['positions']) >= 3:
                            dy = track['positions'][-1][1] - track['positions'][-3][1]
                            if dy < -5:  # Moving up
                                pedestrian_crossing = True
                elif cls == 3:
                    motorcyclists.append(center)
                else:
                    vehicles.append(center)
                    zone_counts[zone] += 1

                # Speed text
                label = "Person" if cls == 0 else "Motorcycle" if cls == 3 else "Vehicle"
                cv2.putText(
                    annotated_frame,
                    f"{label} {track['speed']:.1f} km/h",
                    (int(x - w / 2), int(y - h / 2 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        # Red/Yellow/green - car light
        for zone in ['N', 'S', 'E', 'W']:
            count = zone_counts[zone]
            speeds = zone_speeds[zone]
            avg_speed = np.mean(speeds) if speeds else 0

            if pedestrian_crossing:
                recommendation = f"{zone}: ROSU - Pieton traverseaza"
                color = (0, 0, 255)
            elif pedestrian_near_crossing:
                recommendation = f"{zone}: ROSU - Pieton asteapta"
                color = (0, 0, 200)
            elif count > 5 or avg_speed < 10:
                recommendation = f"{zone}: Prelungire verde ({green_extended_sec}s)"
                color = (0, 255, 0)
            else:
                recommendation = f"{zone}: Durata normala ({green_normal_sec}s)"
                color = (255, 255, 0)

            y_offset = {'N': 30, 'E': 60, 'S': 90, 'W': 120}[zone]
            cv2.putText(
                annotated_frame,
                recommendation,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Alert people near cars
        for person, pid in people:
            for veh in vehicles:
                dist = np.linalg.norm(np.array(person) - np.array(veh))
                if dist < 50:
                    cv2.putText(
                        annotated_frame,
                        f"OPEL ACCIDENT [{pid}]",
                        (int(person[0]), int(person[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3
                    )

        # Close cars alert
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                dist = np.linalg.norm(np.array(vehicles[i]) - np.array(vehicles[j]))
                if dist < 40:
                    mid = (
                        int((vehicles[i][0] + vehicles[j][0]) / 2),
                        int((vehicles[i][1] + vehicles[j][1]) / 2)
                    )
                    cv2.putText(
                        annotated_frame,
                        "HOPA ACCIDENT",
                        mid,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3
                    )

        cv2.imshow("Analiza Trafic AI", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video_2.mp4" #modify path with video_1.mp4 to use the other video
    if not os.path.exists(video_path):
        print("The video path doesn't exist.")
        exit(1)

    process_video(video_path)
