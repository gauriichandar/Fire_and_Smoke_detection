import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional
import threading
from datetime import datetime
import requests
from enum import Enum
import cv2
import firebase_admin
from firebase_admin import credentials, storage

import Config

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import base64

class SimpleCrypto:
    def __init__(self, key):
        # Ensure key is 16 bytes (128 bits) for AES
        self.key = key.encode('utf-8').ljust(16)[:16]

    def encrypt(self, plaintext):
        # Create cipher
        cipher = Cipher(algorithms.AES(self.key), modes.ECB())
        encryptor = cipher.encryptor()

        # Create padder and pad the data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()

        # Encrypt the padded data
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        # Convert to base64
        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt(self, encrypted_data):
        # Convert from base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))

        # Create cipher
        cipher = Cipher(algorithms.AES(self.key), modes.ECB())
        decryptor = cipher.decryptor()

        # Decrypt the data
        decrypted_padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()

        # Create unpadder and unpad the data
        unpadder = padding.PKCS7(128).unpadder()
        decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

        return decrypted_data.decode('utf-8')



# Initialize Firebase Admin SDK with credentials
cred = credentials.Certificate("alert-app-d1368-firebase-adminsdk-qv2u6-75ed891123.json")  # Replace with your actual Firebase credentials file path
firebase_admin.initialize_app(cred, {
    'storageBucket': 'alert-app-d1368.appspot.com'  # Replace with your Firebase storage bucket name
})

@dataclass
class AlertConfig:
    enabled: bool = True
    time_window: float = 10.0  # seconds
    threshold_ratio: float = 0.8  # 80% detection ratio
    cooldown_period: float = 10.0  # seconds between alerts
    email_recipients: List[str] = None

class DetectionType(Enum):
    FIRE_SMOKE = "Fire/Smoke"

class DetectionHistory:
    def __init__(self, config: AlertConfig):
        self.history = deque(maxlen=int(config.time_window * 30))  # Assuming 30 FPS
        self.last_alert_time = 0
        self.config = config

    def update(self, detected: bool) -> None:
        self.history.append(detected)

    def should_alert(self, current_time: float) -> bool:
        if not self.config.enabled or not self.history:
            return False

        if (current_time - self.last_alert_time) < self.config.cooldown_period:
            return False

        detection_ratio = sum(self.history) / len(self.history)
        return detection_ratio >= self.config.threshold_ratio

    def clear_history(self) -> None:
        self.history.clear()

    def update_alert_time(self, current_time: float) -> None:
        self.last_alert_time = current_time

class DetectionMonitor:
    def __init__(
        self,
        fire_smoke_alert_config: AlertConfig,
    ):
        self.detection_histories = {
            DetectionType.FIRE_SMOKE: DetectionHistory(fire_smoke_alert_config),
        }
        
        self.logger = logging.getLogger(__name__)
        self.alert_thread_pool = []

        self.detection_criteria = {
            DetectionType.FIRE_SMOKE: lambda labels: any(label in ["FIRE", "SMOKE"] for label in labels),
        }

    def update(self, labels: List[str], frame) -> None:
        """Update detection histories and check for alerts, with frame upload to Firebase."""
        current_time = time.time()
        
        for detection_type, history in self.detection_histories.items():
            detected = self.detection_criteria[detection_type](labels)
            history.update(detected)
            
            if history.should_alert(current_time):
                history.update_alert_time(current_time)
                history.clear_history()
                self._trigger_alert(detection_type, frame)
        
        self._cleanup_alert_threads()

    def _trigger_alert(self, detection_type: DetectionType, frame) -> None:
        """Trigger alert for the specified detection type and upload frame to Firebase."""
        alert_thread = threading.Thread(
            target=self._send_alert_to_api_and_upload_frame,
            args=(detection_type.value.replace("/","_"), frame)
        )
        alert_thread.start()
        self.alert_thread_pool.append(alert_thread)

    def _cleanup_alert_threads(self) -> None:
        """Clean up completed alert threads."""
        self.alert_thread_pool = [thread for thread in self.alert_thread_pool if thread.is_alive()]

    def _send_alert_to_api_and_upload_frame(self, prediction: str, frame) -> None:
        """Send email alert using yagmail and upload frame to Firebase."""
        import yagmail
        
        # Initialize with a key
        key = "MySecretKey12345"  # must be <= 16 chars
        crypto = SimpleCrypto(key)
        
        # Save frame locally and upload to Firebase
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_path = f"alert_frames/{prediction}_{timestamp}.jpg"
        cv2.imwrite(frame_path, frame)

        try:
            # Upload to Firebase Storage
            bucket = storage.bucket()
            blob = bucket.blob(frame_path)
            blob.upload_from_filename(frame_path)
            blob.make_public()
            frame_url = blob.public_url
            print(f"Frame uploaded to Firebase Storage: {frame_url}")
        except Exception as e:
            print(f"Error uploading frame to Firebase: {e}")
            frame_url = ""

        # Email configuration
        try:
            # Get recipients from AlertConfig
            recipients = Config.email
            
            if not recipients:
                print("No email recipients configured")
                return

            # Yagmail setup (use app password)
            sender_email = "my.kot.app@gmail.com"
            sender_password = "vdzyjhfnlymwtzby"  # App password for Gmail
            
            # Create yagmail instance
            yag = yagmail.SMTP(sender_email, sender_password)

            # Email content
            subject = f"ALERT: {prediction} Detected"
            body = f"""
            ALERT NOTIFICATION

            Detection Type: {prediction}
            Timestamp: {timestamp}

            Frame URL: {frame_url}
            """

            # Send email with attachment
            yag.send(
                to=recipients,
                subject=subject,
                contents=body,
                attachments=frame_path
            )
            
            print("Email alert sent successfully")

        except Exception as e:
            print(f"Error sending email alert: {e}")