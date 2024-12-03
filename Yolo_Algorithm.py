import time 
import cv2
import numpy as np
import onnxruntime
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class DetectionResult:
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    label: str
    frame: np.ndarray

@dataclass
class ModelConfig:
    model_path: str
    class_names: List[str]
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None

class YOLOv8:
    def __init__(
        self, 
        model_path: str, 
        confidence_threshold: float = 0.7, 
        iou_threshold: float = 0.5,
        device: str = 'CPU'
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.initialize_model(model_path)

    def initialize_model(self, model_path: str) -> None:
        """Initialize ONNX Runtime model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            # Set providers based on device selection
            providers = ['CPUExecutionProvider']
            if self.device == 'GPU':
                if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                else:
                    self.logger.warning("CUDA is not available. Falling back to CPU.")
            
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=providers
            )
            
            # Get model info
            self.get_input_details()
            self.get_output_details()
            
            self.logger.info(f"Model initialized successfully using {providers[0]}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def get_input_details(self) -> None:
        """Get model input details"""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self) -> None:
        """Get model output details"""
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wrapper for detect_objects"""
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in an image
        Args:
            image: Input image
        Returns:
            Tuple of boxes, scores, and class IDs
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        try:
            input_tensor = self.prepare_input(image)
            outputs = self.inference(input_tensor)
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)
            
            return self.boxes, self.scores, self.class_ids
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            raise

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare input image for inference"""
        self.image_height, self.image_width = image.shape[:2]

        try:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, (self.input_width, self.input_height))
            input_image = input_image / 255.0
            input_image = input_image.transpose(2, 0, 1)
            input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
            
            return input_tensor
            
        except Exception as e:
            self.logger.error(f"Input preparation failed: {str(e)}")
            raise

    def inference(self, input_tensor: np.ndarray) -> List:
        """Run inference on the input tensor"""
        try:
            start_time = time.perf_counter()
            outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
            inference_time = (time.perf_counter() - start_time) * 1000
            self.logger.debug(f"Inference time: {inference_time:.2f} ms")
            return outputs
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            raise

    def process_output(self, outputs: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process the raw output from the model"""
        predictions = np.squeeze(outputs[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression
        indices = self.apply_nms(boxes, scores, class_ids)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions: np.ndarray) -> np.ndarray:
        """Extract boxes from predictions"""
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = self.xywh_to_xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Rescale boxes to original image dimensions"""
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        return boxes

    @staticmethod
    def xywh_to_xyxy(x: np.ndarray) -> np.ndarray:
        """Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> List[int]:
        """Apply Non-Maximum Suppression"""
        indices = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = self.non_maximum_suppression(class_boxes, class_scores)
            indices.extend(np.where(class_mask)[0][class_indices])
        return indices

    def non_maximum_suppression(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """Apply Non-Maximum Suppression to boxes"""
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.iou_threshold
        )
        return indices.flatten().tolist() if len(indices) > 0 else []

class MultiYOLODetector:
    def __init__(self, model_configs: List[ModelConfig], device: str = 'CPU'):
        """
        Initialize multiple YOLO models
        Args:
            model_configs: List of ModelConfig objects containing paths and settings
            device: 'CPU' or 'GPU'
        """
        self.device = device
        self.models = []
        self.configs = model_configs
        self.logger = logging.getLogger(__name__)
        
        # Initialize each model
        for config in model_configs:
            model = YOLOv8(
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
                device=device
            )
            self.models.append(model)
            
            # Generate random colors for classes if not provided
            if config.color_mapping is None:
                config.color_mapping = {
                    class_name: tuple(map(int, np.random.randint(0, 255, 3)))
                    for class_name in config.class_names
                }

    def detect_all(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Run detection with all models
        Returns combined boxes, scores, and labels
        """
        combined_boxes = []
        combined_scores = []
        combined_labels = []
        
        for model, config in zip(self.models, self.configs):
            try:
                boxes, scores, class_ids = model(frame)
                
                # Convert class IDs to class names
                labels = [config.class_names[class_id] for class_id in class_ids]
                
                combined_boxes.extend(boxes)
                combined_scores.extend(scores)
                combined_labels.extend(labels)
                
            except Exception as e:
                self.logger.error(f"Error in model {config.model_path}: {str(e)}")
                continue
                
        return combined_boxes, combined_scores, combined_labels

    def draw_combined_detections(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        scores: List[np.ndarray],
        labels: List[str],
        draw_scores: bool = True
    ) -> np.ndarray:
        """Draw all detections from all models"""
        det_image = image.copy()
        
        if not boxes:
            return det_image
            
        try:
            img_height, img_width = image.shape[:2]
            font_size = min([img_height, img_width]) * 0.001
            text_thickness = int(min([img_height, img_width]) * 0.001)
            
            # Draw boxes and labels
            for box, score, label in zip(boxes, scores, labels):
                if 'normal' in label.lower():continue
                x1, y1, x2, y2 = box.astype(int)
                
                # Find the corresponding model config for this label
                config = next(
                    (cfg for cfg in self.configs if label in cfg.class_names),
                    self.configs[0]
                )
                
                # Get color for this label
                color = config.color_mapping.get(label, (255, 255, 255))
                
                # Draw box
                cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 2)
                
                if draw_scores:
                    # Draw label
                    caption = f'{label} {int(score * 100)}%'
                    text_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness)[0]
                    cv2.rectangle(det_image, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 5), color, -1)
                    cv2.putText(det_image, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness)
            
            return det_image
            
        except Exception as e:
            self.logger.error(f"Drawing detections failed: {str(e)}")
            return image

class DetectionRecorder:
    def __init__(self, save_path: str = "detections/"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
    def save_frame(self, frame: np.ndarray, labels: List[str]) -> str:
        """Save frame with timestamp and detected labels in filename"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        labels_str = '_'.join(set(labels)) if labels else 'no_detection'
        frame_path = self.save_path / f"{timestamp}_{labels_str}.jpg"
        cv2.imwrite(str(frame_path), frame)
        return str(frame_path)