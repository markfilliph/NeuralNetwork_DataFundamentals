"""Tests for database models and repositories."""

import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.database import (
    DatabaseManager,
    UserRepository,
    SessionRepository,
    AuditLogRepository
)


class TestDatabaseManager:
    """Test suite for DatabaseManager."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test environment."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_database_initialization(self):
        """Test database initialization creates required tables."""
        # Test that we can connect and query
        tables = self.db_manager.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        
        table_names = [table['name'] for table in tables]
        
        expected_tables = [
            'users', 'user_sessions', 'datasets', 
            'models', 'audit_logs', 'api_keys'
        ]
        
        for expected_table in expected_tables:
            assert expected_table in table_names
    
    def test_execute_query(self):
        """Test query execution."""
        # Insert test data
        self.db_manager.execute_update(
            "INSERT INTO users (user_id, username, email, password_hash, password_salt, role) VALUES (?, ?, ?, ?, ?, ?)",
            ("test_user", "testuser", "test@example.com", "hash", "salt", "analyst")
        )
        
        # Query data
        results = self.db_manager.execute_query(
            "SELECT * FROM users WHERE user_id = ?",
            ("test_user",)
        )
        
        assert len(results) == 1
        assert results[0]['username'] == "testuser"
        assert results[0]['email'] == "test@example.com"
    
    def test_execute_update(self):
        """Test update execution."""
        # Insert data
        affected = self.db_manager.execute_update(
            "INSERT INTO users (user_id, username, email, password_hash, password_salt, role) VALUES (?, ?, ?, ?, ?, ?)",
            ("test_user2", "testuser2", "test2@example.com", "hash", "salt", "analyst")
        )
        
        assert affected == 1
        
        # Update data
        affected = self.db_manager.execute_update(
            "UPDATE users SET role = ? WHERE user_id = ?",
            ("admin", "test_user2")
        )
        
        assert affected == 1


class TestUserRepository:
    """Test suite for UserRepository."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_manager = DatabaseManager(self.temp_db.name)
        self.user_repo = UserRepository(self.db_manager)
    
    def teardown_method(self):
        """Clean up test environment."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_create_user(self):
        """Test user creation."""
        success = self.user_repo.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            password_hash="test_hash",
            password_salt="test_salt",
            role="analyst",
            metadata={"created_by": "test"}
        )
        
        assert success is True
        
        # Verify user was created
        user = self.user_repo.get_user("test_user")
        assert user is not None
        assert user['username'] == "testuser"
        assert user['email'] == "test@example.com"
        assert user['role'] == "analyst"
        assert user['metadata']['created_by'] == "test"
    
    def test_get_user_by_username(self):
        """Test getting user by username."""
        # Create user
        self.user_repo.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            password_hash="test_hash",
            password_salt="test_salt",
            role="analyst"
        )
        
        # Get by username
        user = self.user_repo.get_user_by_username("testuser")
        assert user is not None
        assert user['user_id'] == "test_user"
        assert user['email'] == "test@example.com"
        
        # Non-existent user
        user = self.user_repo.get_user_by_username("nonexistent")
        assert user is None
    
    def test_update_user(self):
        """Test user updates."""
        # Create user
        self.user_repo.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            password_hash="test_hash",
            password_salt="test_salt",
            role="analyst"
        )
        
        # Update user
        success = self.user_repo.update_user(
            "test_user",
            email="updated@example.com",
            role="admin",
            metadata={"updated": True}
        )
        
        assert success is True
        
        # Verify updates
        user = self.user_repo.get_user("test_user")
        assert user['email'] == "updated@example.com"
        assert user['role'] == "admin"
        assert user['metadata']['updated'] is True
    
    def test_list_users(self):
        """Test user listing with pagination."""
        # Create multiple users
        for i in range(5):
            self.user_repo.create_user(
                user_id=f"user_{i}",
                username=f"user{i}",
                email=f"user{i}@example.com",
                password_hash="hash",
                password_salt="salt",
                role="analyst"
            )
        
        # List all users
        users = self.user_repo.list_users(limit=10)
        assert len(users) == 5
        
        # Test pagination
        users_page1 = self.user_repo.list_users(limit=2, offset=0)
        users_page2 = self.user_repo.list_users(limit=2, offset=2)
        
        assert len(users_page1) == 2
        assert len(users_page2) == 2
        
        # Should be different users
        page1_ids = {u['user_id'] for u in users_page1}
        page2_ids = {u['user_id'] for u in users_page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    def test_deactivate_user(self):
        """Test user deactivation."""
        # Create user
        self.user_repo.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            password_hash="test_hash",
            password_salt="test_salt",
            role="analyst"
        )
        
        # User should be active initially
        user = self.user_repo.get_user("test_user")
        assert user['is_active'] == 1
        
        # Deactivate user
        success = self.user_repo.deactivate_user("test_user")
        assert success is True
        
        # User should be inactive
        user = self.user_repo.get_user("test_user")
        assert user['is_active'] == 0


class TestSessionRepository:
    """Test suite for SessionRepository."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_manager = DatabaseManager(self.temp_db.name)
        self.session_repo = SessionRepository(self.db_manager)
        
        # Create a test user first
        self.user_repo = UserRepository(self.db_manager)
        self.user_repo.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            password_hash="hash",
            password_salt="salt",
            role="analyst"
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_create_session(self):
        """Test session creation."""
        success = self.session_repo.create_session(
            session_id="test_session",
            user_id="test_user",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        
        assert success is True
        
        # Verify session was created
        session = self.session_repo.get_session("test_session")
        assert session is not None
        assert session['user_id'] == "test_user"
        assert session['ip_address'] == "192.168.1.1"
        assert session['is_active'] == 1
    
    def test_destroy_session(self):
        """Test session destruction."""
        # Create session
        self.session_repo.create_session("test_session", "test_user")
        
        # Session should exist
        session = self.session_repo.get_session("test_session")
        assert session is not None
        
        # Destroy session
        success = self.session_repo.destroy_session("test_session")
        assert success is True
        
        # Session should be inactive
        session = self.session_repo.get_session("test_session")
        assert session is None  # get_session filters by is_active=1
    
    def test_destroy_user_sessions(self):
        """Test destroying all sessions for a user."""
        # Create multiple sessions for user
        for i in range(3):
            self.session_repo.create_session(f"session_{i}", "test_user")
        
        # All sessions should exist
        for i in range(3):
            session = self.session_repo.get_session(f"session_{i}")
            assert session is not None
        
        # Destroy all user sessions
        destroyed = self.session_repo.destroy_user_sessions("test_user")
        assert destroyed == 3
        
        # No sessions should be active
        for i in range(3):
            session = self.session_repo.get_session(f"session_{i}")
            assert session is None


class TestAuditLogRepository:
    """Test suite for AuditLogRepository."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db_manager = DatabaseManager(self.temp_db.name)
        self.audit_repo = AuditLogRepository(self.db_manager)
    
    def teardown_method(self):
        """Clean up test environment."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_create_log(self):
        """Test audit log creation."""
        success = self.audit_repo.create_log(
            event_type="login_success",
            outcome="success",
            user_id="test_user",
            ip_address="192.168.1.1",
            resource="auth",
            action="login",
            risk_level="low",
            details={"method": "password"}
        )
        
        assert success is True
        
        # Verify log was created
        logs = self.audit_repo.get_logs(limit=1)
        assert len(logs) == 1
        
        log = logs[0]
        assert log['event_type'] == "login_success"
        assert log['user_id'] == "test_user"
        assert log['details']['method'] == "password"
    
    def test_get_logs_filtering(self):
        """Test audit log filtering."""
        # Create multiple logs
        events = [
            ("login_success", "user1"),
            ("login_failed", "user2"),
            ("data_uploaded", "user1"),
            ("data_accessed", "user3")
        ]
        
        for event_type, user_id in events:
            self.audit_repo.create_log(
                event_type=event_type,
                outcome="success",
                user_id=user_id
            )
        
        # Test filtering by event type
        login_logs = self.audit_repo.get_logs(event_type="login_success")
        assert len(login_logs) == 1
        assert login_logs[0]['event_type'] == "login_success"
        
        # Test filtering by user
        user1_logs = self.audit_repo.get_logs(user_id="user1")
        assert len(user1_logs) == 2
        for log in user1_logs:
            assert log['user_id'] == "user1"
        
        # Test combined filtering
        user1_login_logs = self.audit_repo.get_logs(
            event_type="login_success",
            user_id="user1"
        )
        assert len(user1_login_logs) == 1
        assert user1_login_logs[0]['event_type'] == "login_success"
        assert user1_login_logs[0]['user_id'] == "user1"
    
    def test_get_logs_pagination(self):
        """Test audit log pagination."""
        # Create multiple logs
        for i in range(10):
            self.audit_repo.create_log(
                event_type="test_event",
                outcome="success",
                user_id=f"user_{i}"
            )
        
        # Test pagination
        page1 = self.audit_repo.get_logs(limit=3, offset=0)
        page2 = self.audit_repo.get_logs(limit=3, offset=3)
        
        assert len(page1) == 3
        assert len(page2) == 3
        
        # Should be different logs
        page1_ids = {log['log_id'] for log in page1}
        page2_ids = {log['log_id'] for log in page2}
        assert page1_ids.isdisjoint(page2_ids)


def run_database_tests():
    """Run all database tests."""
    print("Running Database Tests...")
    
    # Test DatabaseManager
    print("✓ Testing DatabaseManager...")
    test_db = TestDatabaseManager()
    test_db.setup_method()
    test_db.test_database_initialization()
    test_db.test_execute_query()
    test_db.test_execute_update()
    test_db.teardown_method()
    
    # Test UserRepository  
    print("✓ Testing UserRepository...")
    test_user = TestUserRepository()
    
    test_user.setup_method()
    test_user.test_create_user()
    test_user.teardown_method()
    
    test_user.setup_method()
    test_user.test_get_user_by_username()
    test_user.teardown_method()
    
    test_user.setup_method()
    test_user.test_update_user()
    test_user.teardown_method()
    
    test_user.setup_method()
    test_user.test_list_users()
    test_user.teardown_method()
    
    test_user.setup_method()
    test_user.test_deactivate_user()
    test_user.teardown_method()
    
    # Test SessionRepository
    print("✓ Testing SessionRepository...")
    test_session = TestSessionRepository()
    
    test_session.setup_method()
    test_session.test_create_session()
    test_session.teardown_method()
    
    test_session.setup_method()
    test_session.test_destroy_session()
    test_session.teardown_method()
    
    test_session.setup_method()
    test_session.test_destroy_user_sessions()
    test_session.teardown_method()
    
    # Test AuditLogRepository
    print("✓ Testing AuditLogRepository...")
    test_audit = TestAuditLogRepository()
    
    test_audit.setup_method()
    test_audit.test_create_log()
    test_audit.teardown_method()
    
    test_audit.setup_method()
    test_audit.test_get_logs_filtering()
    test_audit.teardown_method()
    
    test_audit.setup_method()
    test_audit.test_get_logs_pagination()
    test_audit.teardown_method()
    
    print("✅ All database tests passed!")


if __name__ == "__main__":
    run_database_tests()