import cv2
import logging
from pathlib import Path
from Monitor_Classes import AlertConfig, DetectionMonitor
from Yolo_Algorithm import DetectionRecorder, ModelConfig, MultiYOLODetector
import Config
# Configure logging
logging.basicConfig(level=logging.INFO)

def detect_from_camera(video_source):
    

    fire_smoke_alert_config = AlertConfig(
        enabled=True,
        time_window=2,  # 5 seconds window
        threshold_ratio=0.6,  # 60% detection ratio
        cooldown_period=4,  # 30 seconds between alerts
    )


    # Initialize detection monitor
    detection_monitor = DetectionMonitor(
         fire_smoke_alert_config=fire_smoke_alert_config,
     )
    
    # Define model configurations
    model_configs = [
        ModelConfig(
            model_path="model/best.onnx",
            class_names=["FIRE", "SMOKE"],
            confidence_threshold=0.5,
            iou_threshold=0.7,
            color_mapping={
                "FIRE": (0, 0, 255),    # Red
                "SMOKE": (128, 128, 128) # Gray
            }
        )
    ]
    
    # Initialize multi-model detector
    detector = MultiYOLODetector(
        model_configs=model_configs,
        device='GPU'
    )
    
    # Initialize detection recorder
    recorder = DetectionRecorder(save_path="detections/")
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get combined detections from all models
            boxes, scores, labels = detector.detect_all(frame)
            
            # Update detection monitor
            detection_monitor.update(labels,frame)
            
            # Draw combined detections
            combined_img = detector.draw_combined_detections(
                frame, boxes, scores, labels,
                draw_scores=True
            )
            
            # Assuming combined_img is your image
            # Calculate the new width while maintaining the aspect ratio
            height = 600
            width = 900

            # Resize the image
            resized_img = cv2.resize(combined_img, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Multi-Model Detection", resized_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:  # Exit on 'esc' or 'q'
                break
            elif key == ord('s'):  # Save frame on 's'
                saved_path = recorder.save_frame(combined_img, labels)
                logging.info(f"Frame saved: {saved_path}")
                
    except Exception as e:
        logging.error(f"Error during detection: {str(e)}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

