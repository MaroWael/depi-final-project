import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient
from collections import defaultdict
import json
import os
from datetime import datetime

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = r"videos\8cd1babd-1a63-4a6f-bf28-0ed4a9a9b8a6.mp4"
MODEL_ID = "saldjs-eodej/1"
API_KEY = "f50xHu5kMJ54A1ERJdnX"

FRAME_SKIP = 5
DISPLAY_VIDEO = True
RESIZE_DIM = (640, 640)
REPORT_JSON_PATH = "report.json"
REPORT_TXT_PATH = "report.txt"


def generate_report(behavior_stats, frame_count):
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "video_path": VIDEO_PATH,
        "total_frames_processed": frame_count,
        "model_id": MODEL_ID,
        "behaviors": {}
    }

    print("\n===== FINAL BEHAVIOR STATS =====")
    summary_lines = []
    summary_lines.append("===== VIDEO ANALYSIS REPORT =====")
    summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Video: {VIDEO_PATH}")
    summary_lines.append(f"Total Frames Processed: {frame_count}")
    summary_lines.append("-" * 30)

    for behavior, stats in behavior_stats.items():
        count = stats["count"]
        avg_conf = stats["confidence_sum"] / count if count > 0 else 0.0
        
        # Determine boolean status based on count
        is_present = count >= 1
        
        report_data["behaviors"][behavior] = {
            "count": count,
            "average_confidence": round(avg_conf, 3),
            "detected": is_present
        }

        line = f"{behavior}: count = {count}, detected = {is_present}"
        print(line)
        summary_lines.append(line)

    # Save JSON
    try:
        with open(REPORT_JSON_PATH, 'w') as json_file:
            json.dump(report_data, json_file, indent=4)
        print(f"JSON report saved to {REPORT_JSON_PATH}")
    except Exception as e:
        print("Error saving JSON:", e)

    # Save TXT
    try:
        with open(REPORT_TXT_PATH, 'w') as txt_file:
            txt_file.write("\n".join(summary_lines))
        print(f"Text report saved to {REPORT_TXT_PATH}")
    except Exception as e:
        print("Error saving TXT:", e)


def main():
    # -----------------------------
    # INITIALIZE ROBFLOW CLIENT
    # -----------------------------
    try:
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",   # FIXED URL
            api_key=API_KEY
        )
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    # -----------------------------
    # LOAD VIDEO
    # -----------------------------
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    frame_count = 0
    behavior_stats = defaultdict(lambda: {"count": 0, "confidence_sum": 0.0})

    print(f"Starting analysis on {VIDEO_PATH}...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                continue

            # Resize
            resized = cv2.resize(frame, RESIZE_DIM)
            frame_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

            # -----------------------------
            # RUN INFERENCE
            # -----------------------------
            try:
                result = CLIENT.infer(frame_pil, model_id=MODEL_ID)
                predictions = result.get("predictions", [])
            except Exception as e:
                print(f"ERROR at frame {frame_count}: {e}")
                continue

            # Collect stats
            for pred in predictions:
                label = pred.get("class")
                conf = pred.get("confidence", 0)
                if label:
                    behavior_stats[label]["count"] += 1
                    behavior_stats[label]["confidence_sum"] += conf

            # -----------------------------
            # DISPLAY VIDEO
            # -----------------------------
            if DISPLAY_VIDEO:
                for pred in predictions:
                    x = int(pred["x"])
                    y = int(pred["y"])
                    w = int(pred["width"])
                    h = int(pred["height"])
                    label = pred["class"]
                    conf = pred["confidence"]

                    x1 = int(x - w/2)
                    y1 = int(y - h/2)
                    x2 = int(x + w/2)
                    y2 = int(y + h/2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,255,0), 2)

                cv2.imshow("Video Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Stopped manually.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\nGenerating final report...")
        generate_report(behavior_stats, frame_count)


if __name__ == "__main__":
    main()
