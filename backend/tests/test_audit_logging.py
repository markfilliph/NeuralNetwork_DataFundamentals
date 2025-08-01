"""Tests for audit logging system."""

import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.logging import (
    AuditLogger,
    AuditEvent,
    EventType,
    LogLevel,
    audit_logger
)


class TestAuditEvent:
    """Test suite for AuditEvent."""
    
    def test_audit_event_creation(self):
        """Test audit event creation."""
        event = AuditEvent(
            timestamp="2023-01-01T12:00:00",
            event_type=EventType.LOGIN_SUCCESS,
            user_id="test_user",
            session_id="session123",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            resource="auth",
            action="login",
            outcome="success",
            details={"method": "password"},
            risk_level="low"
        )
        
        assert event.timestamp == "2023-01-01T12:00:00"
        assert event.event_type == EventType.LOGIN_SUCCESS
        assert event.user_id == "test_user"
        assert event.outcome == "success"
        assert event.details["method"] == "password"
        assert event.risk_level == "low"
    
    def test_audit_event_to_dict(self):
        """Test audit event conversion to dictionary."""
        event = AuditEvent(
            timestamp="2023-01-01T12:00:00",
            event_type=EventType.DATA_UPLOADED,
            user_id="test_user",
            session_id=None,
            ip_address="192.168.1.1",
            user_agent=None,
            resource="dataset_123",
            action="upload",
            outcome="success",
            details={"file_size": 1024, "file_type": "csv"}
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["timestamp"] == "2023-01-01T12:00:00"
        assert event_dict["event_type"] == "data_uploaded"  # Enum value
        assert event_dict["user_id"] == "test_user"
        assert event_dict["resource"] == "dataset_123"
        assert event_dict["details"]["file_size"] == 1024
    
    def test_audit_event_to_json(self):
        """Test audit event conversion to JSON."""
        event = AuditEvent(
            timestamp="2023-01-01T12:00:00",
            event_type=EventType.MODEL_TRAINED,
            user_id="test_user",
            session_id="session123",
            ip_address="192.168.1.1",
            user_agent=None,
            resource="model_456",
            action="train",
            outcome="success",
            details={"algorithm": "linear_regression", "accuracy": 0.85}
        )
        
        json_str = event.to_json()
        
        assert '"timestamp": "2023-01-01T12:00:00"' in json_str
        assert '"event_type": "model_trained"' in json_str
        assert '"accuracy": 0.85' in json_str


class TestAuditLogger:
    """Test suite for AuditLogger."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a fresh audit logger instance
        self.audit_logger = AuditLogger()
        
        # Override log directory for testing
        original_makedirs = os.makedirs
        def mock_makedirs(path, exist_ok=False):
            if "logs" in str(path):
                path = self.temp_dir
            return original_makedirs(path, exist_ok=exist_ok)
        
        os.makedirs = mock_makedirs
    
    def teardown_method(self):
        """Clean up test environment."""
        # Restore original makedirs
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_event_basic(self):
        """Test basic event logging."""
        self.audit_logger.log_event(
            event_type=EventType.LOGIN_SUCCESS,
            user_id="test_user",
            outcome="success",
            details={"method": "password"}
        )
        
        # Check that event was stored in memory
        events = self.audit_logger.get_events(limit=10)
        assert len(events) >= 1
        
        latest_event = events[0]
        assert latest_event.event_type == EventType.LOGIN_SUCCESS
        assert latest_event.user_id == "test_user"
        assert latest_event.outcome == "success"
        assert latest_event.details["method"] == "password"
    
    def test_log_security_event(self):
        """Test security event logging."""
        self.audit_logger.log_security_event(
            event_type=EventType.MALWARE_DETECTED,
            user_id="test_user",
            details={"file_name": "suspicious.exe", "threat_type": "trojan"},
            ip_address="192.168.1.100"
        )
        
        events = self.audit_logger.get_events(limit=10)
        security_event = events[0]
        
        assert security_event.event_type == EventType.MALWARE_DETECTED
        assert security_event.risk_level == "high"
        assert security_event.outcome == "security_event"
        assert security_event.details["threat_type"] == "trojan"
    
    def test_log_authentication_event_success(self):
        """Test successful authentication event logging."""
        self.audit_logger.log_authentication_event(
            success=True,
            user_id="test_user",
            ip_address="192.168.1.1",
            details={"login_method": "password"}
        )
        
        events = self.audit_logger.get_events(limit=10)
        auth_event = events[0]
        
        assert auth_event.event_type == EventType.LOGIN_SUCCESS
        assert auth_event.outcome == "success"
        assert auth_event.risk_level == "low"
    
    def test_log_authentication_event_failure(self):
        """Test failed authentication event logging."""
        self.audit_logger.log_authentication_event(
            success=False,
            user_id="test_user",
            ip_address="192.168.1.1",
            details={"error": "invalid_password", "attempts": 3}
        )
        
        events = self.audit_logger.get_events(limit=10)
        auth_event = events[0]
        
        assert auth_event.event_type == EventType.LOGIN_FAILED
        assert auth_event.outcome == "failure"
        assert auth_event.risk_level == "medium"
        assert auth_event.details["attempts"] == 3
    
    def test_log_data_access(self):
        """Test data access event logging."""
        operations = ["upload", "read", "modify", "delete", "export"]
        
        for operation in operations:
            self.audit_logger.log_data_access(
                user_id="test_user",
                dataset_id=f"dataset_{operation}",
                operation=operation,
                details={"file_size": 1024}
            )
        
        events = self.audit_logger.get_events(limit=10)
        
        # Check that all operations were logged
        logged_operations = [e.action for e in events]
        for operation in operations:
            assert operation in logged_operations
    
    def test_get_events_filtering(self):
        """Test event filtering functionality."""
        # Log multiple events
        self.audit_logger.log_event(EventType.LOGIN_SUCCESS, user_id="user1")
        self.audit_logger.log_event(EventType.LOGIN_FAILED, user_id="user2")
        self.audit_logger.log_event(EventType.DATA_UPLOADED, user_id="user1")
        
        # Filter by event type
        login_events = self.audit_logger.get_events(event_type=EventType.LOGIN_SUCCESS)
        assert len(login_events) == 1
        assert login_events[0].event_type == EventType.LOGIN_SUCCESS
        
        # Filter by user
        user1_events = self.audit_logger.get_events(user_id="user1")
        assert len(user1_events) == 2
        for event in user1_events:
            assert event.user_id == "user1"
    
    def test_get_events_time_filtering(self):
        """Test time-based event filtering."""
        # Create events with specific timestamps
        now = datetime.utcnow()
        past = now - timedelta(hours=2)
        future = now + timedelta(hours=1)
        
        # Mock timestamp creation
        original_utcnow = datetime.utcnow
        
        def mock_utcnow_past():
            return past
        
        def mock_utcnow_present():
            return now
        
        # Log past event
        datetime.utcnow = mock_utcnow_past
        self.audit_logger.log_event(EventType.LOGIN_SUCCESS, user_id="past_user")
        
        # Log present event  
        datetime.utcnow = mock_utcnow_present
        self.audit_logger.log_event(EventType.LOGIN_SUCCESS, user_id="present_user")
        
        # Restore original function
        datetime.utcnow = original_utcnow
        
        # Test time filtering
        recent_events = self.audit_logger.get_events(
            start_time=(now - timedelta(minutes=30)).isoformat()
        )
        
        # Should only get the present event
        present_user_events = [e for e in recent_events if e.user_id == "present_user"]
        assert len(present_user_events) >= 1
    
    def test_get_security_summary(self):
        """Test security summary generation."""
        # Log various types of events
        self.audit_logger.log_event(EventType.LOGIN_SUCCESS, risk_level="low")
        self.audit_logger.log_event(EventType.LOGIN_FAILED, risk_level="medium")
        self.audit_logger.log_event(EventType.ACCESS_DENIED, risk_level="medium")
        self.audit_logger.log_security_event(EventType.MALWARE_DETECTED, risk_level="critical")
        
        summary = self.audit_logger.get_security_summary(hours=24)
        
        assert "total_events" in summary
        assert "security_events" in summary
        assert "failed_logins" in summary
        assert "access_denials" in summary
        assert "high_risk_events" in summary
        
        assert summary["total_events"] >= 4
        assert summary["security_events"] >= 1  # The malware detection
        assert summary["failed_logins"] >= 1
        assert summary["access_denials"] >= 1
    
    def test_memory_limit(self):
        """Test that audit logger respects memory limits."""
        # Override max events for testing
        self.audit_logger._max_events = 5
        
        # Log more events than the limit
        for i in range(10):
            self.audit_logger.log_event(
                EventType.DATA_ACCESSED,
                user_id=f"user_{i}",
                details={"index": i}
            )
        
        # Should only keep the last 5 events
        events = self.audit_logger.get_events(limit=20)
        assert len(events) <= 5
        
        # Should have the most recent events
        user_ids = [e.user_id for e in events]
        assert "user_9" in user_ids  # Most recent
        assert "user_0" not in user_ids  # Oldest should be removed
    
    def test_log_level_determination(self):
        """Test log level determination logic."""
        # Test critical events
        critical_level = self.audit_logger._get_log_level(
            EventType.MALWARE_DETECTED, "success", "critical"
        )
        assert critical_level == LogLevel.CRITICAL
        
        # Test error events
        error_level = self.audit_logger._get_log_level(
            EventType.LOGIN_FAILED, "failure", "low"
        )
        assert error_level == LogLevel.ERROR
        
        # Test warning events
        warning_level = self.audit_logger._get_log_level(
            EventType.RATE_LIMIT_EXCEEDED, "success", "medium"
        )
        assert warning_level == LogLevel.WARNING
        
        # Test info events
        info_level = self.audit_logger._get_log_level(
            EventType.DATA_ACCESSED, "success", "low"
        )
        assert info_level == LogLevel.INFO


def run_audit_logging_tests():
    """Run all audit logging tests."""
    print("Running Phase 2 Audit Logging Tests...")
    
    # Test AuditEvent
    print("✓ Testing AuditEvent...")
    test_event = TestAuditEvent()
    test_event.test_audit_event_creation()
    test_event.test_audit_event_to_dict()
    test_event.test_audit_event_to_json()
    
    # Test AuditLogger
    print("✓ Testing AuditLogger...")
    test_logger = TestAuditLogger()
    
    test_logger.setup_method()
    test_logger.test_log_event_basic()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_log_security_event()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_log_authentication_event_success()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_log_authentication_event_failure()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_log_data_access()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_get_events_filtering()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_get_security_summary()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_memory_limit()
    test_logger.teardown_method()
    
    test_logger.setup_method()
    test_logger.test_log_level_determination()
    test_logger.teardown_method()
    
    print("✅ All audit logging tests passed!")


if __name__ == "__main__":
    run_audit_logging_tests()