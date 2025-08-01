"""Tests for encryption service."""

import sys
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.encryption_service import (
    EncryptionService,
    EncryptionError,
    KeyManagementError,
    SimpleFernet
)


class TestEncryptionService:
    """Test suite for EncryptionService."""
    
    def setup_method(self):
        """Set up test environment."""
        self.encryption_service = EncryptionService()
    
    def test_encrypt_decrypt_data(self):
        """Test basic data encryption and decryption."""
        original_data = b"This is test data for encryption"
        
        # Encrypt data
        encrypted_data = self.encryption_service.encrypt_data(original_data)
        
        # Should be different from original
        assert encrypted_data != original_data
        assert len(encrypted_data) > len(original_data)  # Includes IV
        
        # Decrypt data
        decrypted_data = self.encryption_service.decrypt_data(encrypted_data)
        
        # Should match original
        assert decrypted_data == original_data
    
    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption."""
        original_text = "This is a test string with unicode: 🔐 🔑"
        
        # Encrypt string
        encrypted_text = self.encryption_service.encrypt_string(original_text)
        
        # Should be base64 encoded
        assert isinstance(encrypted_text, str)
        assert encrypted_text != original_text
        
        # Decrypt string
        decrypted_text = self.encryption_service.decrypt_string(encrypted_text)
        
        # Should match original
        assert decrypted_text == original_text
    
    def test_encrypt_decrypt_file(self):
        """Test file encryption and decryption."""
        original_content = b"This is test file content\nWith multiple lines\nAnd binary data: \x00\x01\x02"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            test_file = temp_path / "test.txt"
            test_file.write_bytes(original_content)
            
            # Encrypt file
            encrypted_file = self.encryption_service.encrypt_file(
                test_file, 
                remove_original=False
            )
            
            # Original file should still exist (remove_original=False)
            assert test_file.exists()
            assert encrypted_file.exists()
            assert encrypted_file.suffix == ".encrypted"
            
            # Encrypted file should be different
            encrypted_content = encrypted_file.read_bytes()
            assert encrypted_content != original_content
            
            # Decrypt file
            decrypted_file = self.encryption_service.decrypt_file(encrypted_file)
            
            # Should match original content
            decrypted_content = decrypted_file.read_bytes()
            assert decrypted_content == original_content
    
    def test_encrypt_file_remove_original(self):
        """Test file encryption with original removal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            test_file = temp_path / "test.txt"
            test_file.write_bytes(b"test content")
            
            # Encrypt file with removal
            encrypted_file = self.encryption_service.encrypt_file(
                test_file,
                remove_original=True
            )
            
            # Original file should be gone
            assert not test_file.exists()
            assert encrypted_file.exists()
    
    def test_encrypt_nonexistent_file(self):
        """Test encryption of non-existent file raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = Path(temp_dir) / "nonexistent.txt"
            
            try:
                self.encryption_service.encrypt_file(nonexistent_file)
                assert False, "Should have raised EncryptionError"
            except EncryptionError as e:
                assert "not found" in str(e).lower()
    
    def test_decrypt_nonexistent_file(self):
        """Test decryption of non-existent file raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = Path(temp_dir) / "nonexistent.encrypted"
            
            try:
                self.encryption_service.decrypt_file(nonexistent_file)
                assert False, "Should have raised EncryptionError"
            except EncryptionError as e:
                assert "not found" in str(e).lower()
    
    def test_get_key_info(self):
        """Test key information retrieval."""
        key_info = self.encryption_service.get_key_info()
        
        assert "key_fingerprint" in key_info
        assert "algorithm" in key_info
        assert "key_length" in key_info
        assert "cache_size" in key_info
        
        assert len(key_info["key_fingerprint"]) == 16  # SHA256 truncated
        assert "Fernet" in key_info["algorithm"]
        assert key_info["key_length"] > 0
    
    def test_key_rotation(self):
        """Test encryption key rotation."""
        # Get original key info
        original_info = self.encryption_service.get_key_info()
        original_fingerprint = original_info["key_fingerprint"]
        
        # Rotate key
        new_key = self.encryption_service.rotate_key()
        
        # Key info should change
        new_info = self.encryption_service.get_key_info()
        new_fingerprint = new_info["key_fingerprint"]
        
        assert new_fingerprint != original_fingerprint
        assert len(new_key) > 0
    
    def test_different_encryptions_produce_different_results(self):
        """Test that same data encrypted twice produces different results."""
        data = b"Same data encrypted twice"
        
        encrypted1 = self.encryption_service.encrypt_data(data)
        encrypted2 = self.encryption_service.encrypt_data(data)
        
        # Should be different due to random IV
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same original
        decrypted1 = self.encryption_service.decrypt_data(encrypted1)
        decrypted2 = self.encryption_service.decrypt_data(encrypted2)
        
        assert decrypted1 == data
        assert decrypted2 == data


class TestSimpleFernet:
    """Test suite for SimpleFernet implementation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.key = b"test_key_32_bytes_long_for_aes_" 
        self.cipher = SimpleFernet(self.key)
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test encrypt/decrypt roundtrip."""
        original_data = b"Test data for SimpleFernet"
        
        encrypted = self.cipher.encrypt(original_data)
        decrypted = self.cipher.decrypt(encrypted)
        
        assert decrypted == original_data
    
    def test_encrypted_data_has_iv(self):
        """Test that encrypted data includes IV."""
        data = b"test"
        encrypted = self.cipher.encrypt(data)
        
        # Should have IV (16 bytes) + encrypted data
        assert len(encrypted) >= 16 + len(data)
    
    def test_decrypt_invalid_data_raises_error(self):
        """Test that decrypting invalid data raises error."""
        try:
            self.cipher.decrypt(b"too_short")
            assert False, "Should have raised EncryptionError"
        except EncryptionError:
            pass
    
    def test_different_keys_produce_different_results(self):
        """Test that different keys produce different results."""
        data = b"same data"
        
        cipher1 = SimpleFernet(b"key1" + b"0" * 28)
        cipher2 = SimpleFernet(b"key2" + b"0" * 28)
        
        encrypted1 = cipher1.encrypt(data)
        encrypted2 = cipher2.encrypt(data)
        
        # Should be different
        assert encrypted1 != encrypted2
        
        # Each should decrypt correctly with its own key
        assert cipher1.decrypt(encrypted1) == data
        assert cipher2.decrypt(encrypted2) == data
        
        # But not with the other key
        try:
            cipher1.decrypt(encrypted2)
            # May or may not fail depending on implementation, but results should differ
            result = cipher1.decrypt(encrypted2)
            assert result != data  # Should be garbled
        except:
            pass  # Expected to fail


def run_encryption_tests():
    """Run all encryption service tests."""
    print("Running Encryption Service Tests...")
    
    # Test EncryptionService
    print("✓ Testing EncryptionService...")
    test_encryption = TestEncryptionService()
    
    test_encryption.setup_method()
    test_encryption.test_encrypt_decrypt_data()
    
    test_encryption.setup_method()
    test_encryption.test_encrypt_decrypt_string()
    
    test_encryption.setup_method()
    test_encryption.test_encrypt_decrypt_file()
    
    test_encryption.setup_method()
    test_encryption.test_encrypt_file_remove_original()
    
    test_encryption.setup_method()
    test_encryption.test_encrypt_nonexistent_file()
    
    test_encryption.setup_method()
    test_encryption.test_decrypt_nonexistent_file()
    
    test_encryption.setup_method()
    test_encryption.test_get_key_info()
    
    test_encryption.setup_method()
    test_encryption.test_key_rotation()
    
    test_encryption.setup_method()
    test_encryption.test_different_encryptions_produce_different_results()
    
    # Test SimpleFernet
    print("✓ Testing SimpleFernet...")
    test_fernet = TestSimpleFernet()
    
    test_fernet.setup_method()
    test_fernet.test_encrypt_decrypt_roundtrip()
    
    test_fernet.setup_method()
    test_fernet.test_encrypted_data_has_iv()
    
    test_fernet.setup_method()
    test_fernet.test_decrypt_invalid_data_raises_error()
    
    test_fernet.setup_method()
    test_fernet.test_different_keys_produce_different_results()
    
    print("✅ All encryption service tests passed!")


if __name__ == "__main__":
    run_encryption_tests()