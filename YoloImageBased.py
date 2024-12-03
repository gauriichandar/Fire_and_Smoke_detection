import cv2
import logging
from pathlib import Path
from Monitor_Classes import AlertConfig, DetectionMonitor
from Yolo_Algorithm import DetectionRecorder, ModelConfig, MultiYOLODetector

# Configure logging
logging.basicConfig(level=logging.INFO)

def detect(img_url):
    print(img_url)
    # Read image
    frame = cv2.imread(img_url)
     
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
    
    try:
        # Get combined detections from all models
        boxes, scores, labels = detector.detect_all(frame)

        print(labels)
        
        # Update detection monitor
        detection_monitor.update(labels,frame)
        
        # Draw combined detections
        combined_img = detector.draw_combined_detections(
            frame, boxes, scores, labels,
            draw_scores=True
        )
        
        # Resize image to 600px height while maintaining aspect ratio
        h, w = combined_img.shape[:2]
        aspect_ratio = w / h
        new_height = 600
        new_width = int(new_height * aspect_ratio)
        resized_img = cv2.resize(combined_img, (new_width, new_height))
        
        # Display result
        cv2.imshow("Multi-Model Detection", resized_img)  # Show resized image
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        if key in [27, ord('q')]:  # Exit on 'esc' or 'q'
            return
        elif key == ord('s'):  # Save frame on 's'
            saved_path = recorder.save_frame(resized_img, labels)  # Save resized image
            logging.info(f"Frame saved: {saved_path}")
                
    except Exception as e:
        logging.error(f"Error during detection: {str(e)}")
        
    finally:
        cv2.destroyAllWindows()


import cv2
import logging
from pathlib import Path
from Monitor_Classes import AlertConfig, DetectionMonitor
from Yolo_Algorithm import DetectionRecorder, ModelConfig, MultiYOLODetector
import Config

# Configure logging
logging.basicConfig(level=logging.INFO)

def detect_from_image(image_path):
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
    
    # Read the image from the given path
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            logging.error(f"Unable to read image: {image_path}")
            return
        
        # Get combined detections from all models
        boxes, scores, labels = detector.detect_all(frame)
        
        # Update detection monitor
        detection_monitor.update(labels, frame)
        
        # Draw combined detections
        combined_img = detector.draw_combined_detections(
            frame, boxes, scores, labels,
            draw_scores=True
        )
        
        # Resize the image for better visualization
        height = 600
        width = 900
        resized_img = cv2.resize(combined_img, (width, height), interpolation=cv2.INTER_AREA)
        
        # Display the image
        cv2.imshow("Image Detection", resized_img)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()
        
        # Save the resulting image
        saved_path = recorder.save_frame(combined_img, labels)
        logging.info(f"Processed image saved: {saved_path}")
        
    except Exception as e:
        logging.error(f"Error during detection: {str(e)}")

