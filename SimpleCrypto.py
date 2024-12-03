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
