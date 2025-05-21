import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import yt_dlp
import os

def download_youtube_video(url, output_path='video_1.mp4'):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pixel_to_meter = 0.05

    vehicle_model = YOLO('yolov8n.pt')
    sign_model = YOLO('yolov8s.pt')

    track_history = defaultdict(lambda: {'positions': [], 'speed': 0})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        vehicle_results = vehicle_model.track(
            frame,
            persist=True,
            classes=[2, 3, 5, 7],
            tracker="bytetrack.yaml"
        )

        sign_results = sign_model(frame, classes=[9, 11])

        annotated_frame = vehicle_results[0].plot()

        if vehicle_results[0].boxes.id is not None:
            for box, obj_id in zip(vehicle_results[0].boxes.xywh.cpu(),
                                   vehicle_results[0].boxes.id.int().cpu().tolist()):
                x, y, w, h = box
                center = (float(x), float(y))

                track = track_history[obj_id]
                track['positions'].append(center)
                if len(track['positions']) > 10:
                    track['positions'].pop(0)

                if len(track['positions']) >= 2:
                    dist_pixels = np.linalg.norm(
                        np.array(track['positions'][-1]) - np.array(track['positions'][-2])
                    )
                    speed_kph = dist_pixels * pixel_to_meter * fps * 3.6
                    track['speed'] = speed_kph

                    cv2.putText(
                        annotated_frame,
                        f"{speed_kph:.1f} km/h",
                        (int(x - w / 2), int(y - h / 2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

        for sign in sign_results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = map(int, sign[:6])
            label = sign_model.names[int(cls)]
            color = (0, 255, 0) if "green" in label.lower() else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("YouTube Traffic Analysis", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=MxUFsufAGoc"
    local_video_path = "video_1.mp4"

    if not os.path.exists(local_video_path):
        print("Downloading video...")
        download_youtube_video(youtube_url, output_path=local_video_path)
        print("Download complete.")

    process_video(local_video_path)
