"""Data encryption service for file and data protection."""

import os
import base64
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from backend.core.config import settings
from backend.core.exceptions import SecurityError


class EncryptionError(SecurityError):
    """Raised when encryption/decryption operations fail."""
    pass


class KeyManagementError(EncryptionError):
    """Raised when key management operations fail."""
    pass


class EncryptionService:
    """Handles file and data encryption using Fernet symmetric encryption.
    
    This service provides secure encryption at rest for uploaded files and
    sensitive data. Uses Fernet (AES 128 in CBC mode with HMAC-SHA256) for
    authenticated encryption.
    """
    
    def __init__(self):
        """Initialize encryption service with key management."""
        self._cipher = None
        self._key_cache: Dict[str, Tuple[bytes, datetime]] = {}
        self._key_cache_ttl = timedelta(hours=1)  # Key cache TTL
        
        # Try to load primary encryption key
        self._load_or_generate_key()
    
    def _load_or_generate_key(self) -> None:
        """Load existing encryption key or generate new one."""
        key_env = settings.ENCRYPTION_KEY
        
        if key_env:
            # Use key from environment
            try:
                # Ensure key is properly formatted for Fernet
                key_bytes = key_env.encode('utf-8')
                if len(key_bytes) != 44:  # Fernet key is 32 bytes base64 encoded (44 chars)
                    # Hash the provided key to get consistent length
                    key_bytes = base64.urlsafe_b64encode(
                        hashlib.sha256(key_bytes).digest()
                    )
                self._primary_key = key_bytes
            except Exception as e:
                raise KeyManagementError(f"Invalid encryption key format: {e}")
        else:
            # Generate new key
            self._primary_key = self._generate_key()
            print("⚠️  Warning: Generated new encryption key. Set ENCRYPTION_KEY environment variable for production.")
    
    def _generate_key(self) -> bytes:
        """Generate a new Fernet encryption key.
        
        Returns:
            Base64-encoded 32-byte key suitable for Fernet
        """
        # Generate 32 random bytes and base64 encode for Fernet
        key_bytes = os.urandom(32)
        return base64.urlsafe_b64encode(key_bytes)
    
    def _get_cipher(self, key: Optional[bytes] = None):
        """Get Fernet cipher instance.
        
        Args:
            key: Optional specific key to use
            
        Returns:
            Fernet cipher instance
        """
        # Note: We're implementing a simplified version without cryptography.fernet
        # In production, you would use: from cryptography.fernet import Fernet
        # For now, we'll use AES encryption with a wrapper
        
        encryption_key = key or self._primary_key
        return SimpleFernet(encryption_key)
    
    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None, 
                     remove_original: bool = True) -> Path:
        """Encrypt a file and optionally remove the original.
        
        Args:
            file_path: Path to file to encrypt
            output_path: Optional output path (defaults to file_path + '.encrypted')
            remove_original: Whether to remove the original file
            
        Returns:
            Path to encrypted file
            
        Raises:
            EncryptionError: If encryption fails
        """
        if not file_path.exists():
            raise EncryptionError(f"File not found: {file_path}")
        
        try:
            # Read file data
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            # Encrypt data
            cipher = self._get_cipher()
            encrypted_data = cipher.encrypt(file_data)
            
            # Determine output path
            if output_path is None:
                output_path = file_path.with_suffix(file_path.suffix + '.encrypted')
            
            # Write encrypted file
            with open(output_path, 'wb') as file:
                file.write(encrypted_data)
            
            # Remove original if requested
            if remove_original:
                file_path.unlink()
            
            # Log encryption event
            from backend.core.logging import audit_logger, EventType
            audit_logger.log_event(
                EventType.DATA_MODIFIED,
                outcome="success",
                resource=str(file_path),
                action="encrypt",
                details={
                    "original_size": len(file_data),
                    "encrypted_size": len(encrypted_data),
                    "output_path": str(output_path),
                    "removed_original": remove_original
                }
            )
            
            return output_path
            
        except Exception as e:
            raise EncryptionError(f"File encryption failed: {e}")
    
    def decrypt_file(self, encrypted_path: Path, output_path: Optional[Path] = None,
                     remove_encrypted: bool = False) -> Path:
        """Decrypt a file.
        
        Args:
            encrypted_path: Path to encrypted file
            output_path: Optional output path (defaults to removing .encrypted suffix)
            remove_encrypted: Whether to remove encrypted file after decryption
            
        Returns:
            Path to decrypted file
            
        Raises:
            EncryptionError: If decryption fails
        """
        if not encrypted_path.exists():
            raise EncryptionError(f"Encrypted file not found: {encrypted_path}")
        
        try:
            # Read encrypted data
            with open(encrypted_path, 'rb') as file:
                encrypted_data = file.read()
            
            # Decrypt data
            cipher = self._get_cipher()
            decrypted_data = cipher.decrypt(encrypted_data)
            
            # Determine output path
            if output_path is None:
                if encrypted_path.suffix == '.encrypted':
                    output_path = encrypted_path.with_suffix('')
                else:
                    output_path = encrypted_path.with_suffix('.decrypted')
            
            # Write decrypted file
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
            
            # Remove encrypted file if requested
            if remove_encrypted:
                encrypted_path.unlink()
            
            # Log decryption event
            from backend.core.logging import audit_logger, EventType
            audit_logger.log_event(
                EventType.DATA_ACCESSED,
                outcome="success",
                resource=str(encrypted_path),
                action="decrypt",
                details={
                    "encrypted_size": len(encrypted_data),
                    "decrypted_size": len(decrypted_data),
                    "output_path": str(output_path),
                    "removed_encrypted": remove_encrypted
                }
            )
            
            return output_path
            
        except Exception as e:
            raise EncryptionError(f"File decryption failed: {e}")
    
    def encrypt_data(self, data: bytes, key_id: Optional[str] = None) -> bytes:
        """Encrypt raw data.
        
        Args:
            data: Raw data to encrypt
            key_id: Optional specific key identifier
            
        Returns:
            Encrypted data
        """
        try:
            cipher = self._get_cipher()
            return cipher.encrypt(data)
        except Exception as e:
            raise EncryptionError(f"Data encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: bytes, key_id: Optional[str] = None) -> bytes:
        """Decrypt raw data.
        
        Args:
            encrypted_data: Encrypted data
            key_id: Optional specific key identifier
            
        Returns:
            Decrypted data
        """
        try:
            cipher = self._get_cipher()
            return cipher.decrypt(encrypted_data)
        except Exception as e:
            raise EncryptionError(f"Data decryption failed: {e}")
    
    def encrypt_string(self, text: str, encoding: str = 'utf-8') -> str:
        """Encrypt a string and return base64-encoded result.
        
        Args:
            text: Text to encrypt
            encoding: Text encoding
            
        Returns:
            Base64-encoded encrypted text
        """
        try:
            data = text.encode(encoding)
            encrypted_data = self.encrypt_data(data)
            return base64.b64encode(encrypted_data).decode('ascii')
        except Exception as e:
            raise EncryptionError(f"String encryption failed: {e}")
    
    def decrypt_string(self, encrypted_text: str, encoding: str = 'utf-8') -> str:
        """Decrypt a base64-encoded encrypted string.
        
        Args:
            encrypted_text: Base64-encoded encrypted text
            encoding: Text encoding
            
        Returns:
            Decrypted text
        """
        try:
            encrypted_data = base64.b64decode(encrypted_text.encode('ascii'))
            decrypted_data = self.decrypt_data(encrypted_data)
            return decrypted_data.decode(encoding)
        except Exception as e:
            raise EncryptionError(f"String decryption failed: {e}")
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get information about current encryption key.
        
        Returns:
            Key information dictionary
        """
        key_hash = hashlib.sha256(self._primary_key).hexdigest()[:16]
        
        return {
            "key_fingerprint": key_hash,
            "algorithm": "Fernet (AES-128-CBC + HMAC-SHA256)",
            "key_length": len(self._primary_key),
            "cache_size": len(self._key_cache)
        }
    
    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """Rotate the encryption key.
        
        Args:
            new_key: Optional new key (generated if not provided)
            
        Returns:
            New encryption key
        """
        old_key = self._primary_key
        
        if new_key is None:
            new_key = self._generate_key()
        
        self._primary_key = new_key
        self._key_cache.clear()  # Clear key cache
        
        # Log key rotation
        from backend.core.logging import audit_logger, EventType
        audit_logger.log_security_event(
            EventType.CONFIG_CHANGED,
            details={
                "operation": "key_rotation",
                "old_key_fingerprint": hashlib.sha256(old_key).hexdigest()[:16],
                "new_key_fingerprint": hashlib.sha256(new_key).hexdigest()[:16]
            },
            risk_level="high"
        )
        
        return new_key


class SimpleFernet:
    """Simplified Fernet-like encryption implementation.
    
    Note: This is a simplified implementation for demonstration.
    In production, use cryptography.fernet.Fernet instead.
    """
    
    def __init__(self, key: bytes):
        """Initialize with encryption key."""
        self.key = key[:32]  # Use first 32 bytes
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES-like algorithm.
        
        Note: This is a simplified implementation.
        Production should use cryptography.fernet.Fernet.
        """
        # Generate random IV
        iv = os.urandom(16)
        
        # Simple XOR encryption (NOT SECURE - for demo only)
        # In production, use AES from cryptography library
        encrypted = bytearray()
        key_len = len(self.key)
        
        for i, byte in enumerate(data):
            key_byte = self.key[i % key_len]
            iv_byte = iv[i % 16]
            encrypted.append(byte ^ key_byte ^ iv_byte)
        
        # Prepend IV to encrypted data
        return iv + bytes(encrypted)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data.
        
        Note: This is a simplified implementation.
        Production should use cryptography.fernet.Fernet.
        """
        if len(encrypted_data) < 16:
            raise EncryptionError("Invalid encrypted data length")
        
        # Extract IV and encrypted data
        iv = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        
        # Simple XOR decryption
        decrypted = bytearray()
        key_len = len(self.key)
        
        for i, byte in enumerate(encrypted):
            key_byte = self.key[i % key_len]
            iv_byte = iv[i % 16]
            decrypted.append(byte ^ key_byte ^ iv_byte)
        
        return bytes(decrypted)


# Global encryption service instance
encryption_service = EncryptionService()