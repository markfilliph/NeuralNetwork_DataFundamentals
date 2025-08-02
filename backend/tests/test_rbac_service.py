"""Tests for Role-Based Access Control (RBAC) service."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.rbac_service_db import (
    DatabaseRBACService, 
    Permission, 
    Role, 
    User,
    db_rbac_service
)
from backend.core.exceptions import SecurityError


class TestRBACService:
    """Test suite for DatabaseRBACService."""
    
    def setup_method(self):
        """Set up test environment."""
        self.rbac = DatabaseRBACService()
    
    def test_create_user(self):
        """Test user creation."""
        user = self.rbac.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            role=Role.ANALYST
        )
        
        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == Role.ANALYST
        assert user.is_active is True
    
    def test_create_duplicate_user(self):
        """Test creating duplicate user raises error."""
        self.rbac.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            role=Role.ANALYST
        )
        
        try:
            self.rbac.create_user(
                user_id="test_user",  # Same ID
                username="another",
                email="another@example.com",
                role=Role.VIEWER
            )
            assert False, "Should have raised SecurityError"
        except SecurityError as e:
            assert "already exists" in str(e)
    
    def test_get_user(self):
        """Test user retrieval."""
        original_user = self.rbac.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            role=Role.ANALYST
        )
        
        retrieved_user = self.rbac.get_user("test_user")
        
        assert retrieved_user is not None
        assert retrieved_user.user_id == original_user.user_id
        assert retrieved_user.username == original_user.username
        
        # Non-existent user
        assert self.rbac.get_user("nonexistent") is None
    
    def test_create_and_get_session(self):
        """Test session management."""
        user = self.rbac.create_user(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            role=Role.ANALYST
        )
        
        session_id = "session123"
        
        # Create session
        self.rbac.create_session(user.user_id, session_id)
        
        # Retrieve user by session
        retrieved_user = self.rbac.get_user_by_session(session_id)
        assert retrieved_user is not None
        assert retrieved_user.user_id == user.user_id
        
        # Destroy session
        self.rbac.destroy_session(session_id)
        assert self.rbac.get_user_by_session(session_id) is None
    
    def test_role_permissions(self):
        """Test role permission mappings."""
        # Test viewer permissions
        viewer = self.rbac.create_user("viewer", "viewer", "viewer@test.com", Role.VIEWER)
        assert self.rbac.has_permission(viewer, Permission.READ_DATA)
        assert self.rbac.has_permission(viewer, Permission.VIEW_MODELS)
        assert not self.rbac.has_permission(viewer, Permission.UPLOAD_DATA)
        assert not self.rbac.has_permission(viewer, Permission.TRAIN_MODELS)
        
        # Test analyst permissions
        analyst = self.rbac.create_user("analyst", "analyst", "analyst@test.com", Role.ANALYST)
        assert self.rbac.has_permission(analyst, Permission.READ_DATA)
        assert self.rbac.has_permission(analyst, Permission.UPLOAD_DATA)
        assert self.rbac.has_permission(analyst, Permission.TRAIN_MODELS)
        assert not self.rbac.has_permission(analyst, Permission.MANAGE_USERS)
        
        # Test admin permissions
        admin = self.rbac.create_user("admin", "admin", "admin@test.com", Role.ADMIN)
        assert self.rbac.has_permission(admin, Permission.READ_DATA)
        assert self.rbac.has_permission(admin, Permission.MANAGE_USERS)
        assert self.rbac.has_permission(admin, Permission.DELETE_DATA)
        
        # Test super admin permissions (should have all)
        super_admin = self.rbac.create_user("superadmin", "superadmin", "sa@test.com", Role.SUPER_ADMIN)
        for permission in Permission:
            assert self.rbac.has_permission(super_admin, permission)
    
    def test_inactive_user_permissions(self):
        """Test that inactive users have no permissions."""
        user = self.rbac.create_user("inactive", "inactive", "inactive@test.com", Role.ADMIN)
        
        # User should have permissions when active
        assert self.rbac.has_permission(user, Permission.MANAGE_USERS)
        
        # Deactivate user
        user.is_active = False
        
        # User should have no permissions when inactive
        assert not self.rbac.has_permission(user, Permission.MANAGE_USERS)
        assert not self.rbac.has_permission(user, Permission.READ_DATA)
    
    def test_get_user_permissions(self):
        """Test getting all user permissions."""
        analyst = self.rbac.create_user("analyst", "analyst", "analyst@test.com", Role.ANALYST)
        
        permissions = self.rbac.get_user_permissions(analyst)
        
        expected_permissions = {
            Permission.READ_DATA,
            Permission.UPLOAD_DATA,
            Permission.EXPORT_DATA,
            Permission.VIEW_MODELS,
            Permission.TRAIN_MODELS,
        }
        
        assert permissions == expected_permissions
    
    def test_can_access_resource(self):
        """Test resource access checking."""
        analyst = self.rbac.create_user("analyst", "analyst", "analyst@test.com", Role.ANALYST)
        viewer = self.rbac.create_user("viewer", "viewer", "viewer@test.com", Role.VIEWER)
        
        # Analyst should be able to upload data
        assert self.rbac.can_access_resource(analyst, "data", "upload")
        assert self.rbac.can_access_resource(analyst, "data", "read")
        assert self.rbac.can_access_resource(analyst, "model", "train")
        
        # Viewer should not be able to upload data
        assert not self.rbac.can_access_resource(viewer, "data", "upload")
        assert self.rbac.can_access_resource(viewer, "data", "read")
        assert not self.rbac.can_access_resource(viewer, "model", "train")
        
        # Unknown resource/action should return False
        assert not self.rbac.can_access_resource(analyst, "unknown", "action")
    
    def test_list_users_admin_only(self):
        """Test that only admins can list users."""
        admin = self.rbac.create_user("admin", "admin", "admin@test.com", Role.ADMIN)
        analyst = self.rbac.create_user("analyst", "analyst", "analyst@test.com", Role.ANALYST)
        viewer = self.rbac.create_user("viewer", "viewer", "viewer@test.com", Role.VIEWER)
        
        # Admin should be able to list users
        users = self.rbac.list_users(admin)
        assert len(users) == 3
        user_ids = [u.user_id for u in users]
        assert "admin" in user_ids
        assert "analyst" in user_ids
        assert "viewer" in user_ids
        
        # Non-admin should not be able to list users
        try:
            self.rbac.list_users(analyst)
            assert False, "Should have raised SecurityError"
        except SecurityError as e:
            assert "Permission denied" in str(e)
    
    def test_update_user_role(self):
        """Test updating user roles."""
        admin = self.rbac.create_user("admin", "admin", "admin@test.com", Role.ADMIN)
        user = self.rbac.create_user("user", "user", "user@test.com", Role.VIEWER)
        
        # Admin should be able to update user role
        updated_user = self.rbac.update_user_role(admin, "user", Role.ANALYST)
        assert updated_user.role == Role.ANALYST
        
        # Check that the change persisted
        retrieved_user = self.rbac.get_user("user")
        assert retrieved_user.role == Role.ANALYST
        
        # Non-admin should not be able to update roles
        try:
            self.rbac.update_user_role(user, "admin", Role.VIEWER)
            assert False, "Should have raised SecurityError"
        except SecurityError as e:
            assert "Permission denied" in str(e)
    
    def test_super_admin_restrictions(self):
        """Test super admin creation restrictions."""
        admin = self.rbac.create_user("admin", "admin", "admin@test.com", Role.ADMIN)
        super_admin = self.rbac.create_user("superadmin", "superadmin", "sa@test.com", Role.SUPER_ADMIN)
        user = self.rbac.create_user("user", "user", "user@test.com", Role.VIEWER)
        
        # Only super-admin can create super-admin users
        try:
            self.rbac.update_user_role(admin, "user", Role.SUPER_ADMIN)
            assert False, "Should have raised SecurityError"
        except SecurityError as e:
            assert "Only super-admins can create super-admin users" in str(e)
        
        # Super-admin should be able to create super-admin users
        updated_user = self.rbac.update_user_role(super_admin, "user", Role.SUPER_ADMIN)
        assert updated_user.role == Role.SUPER_ADMIN
    
    def test_deactivate_user(self):
        """Test user deactivation."""
        admin = self.rbac.create_user("admin", "admin", "admin@test.com", Role.ADMIN)
        user = self.rbac.create_user("user", "user", "user@test.com", Role.ANALYST)
        
        # Create a session for the user
        self.rbac.create_session(user.user_id, "session123")
        assert self.rbac.get_user_by_session("session123") is not None
        
        # Deactivate user
        deactivated_user = self.rbac.deactivate_user(admin, "user")
        assert deactivated_user.is_active is False
        
        # Session should be destroyed
        assert self.rbac.get_user_by_session("session123") is None
    
    def test_deactivate_super_admin_restrictions(self):
        """Test super admin deactivation restrictions."""
        admin = self.rbac.create_user("admin", "admin", "admin@test.com", Role.ADMIN)
        super_admin = self.rbac.create_user("superadmin", "superadmin", "sa@test.com", Role.SUPER_ADMIN)
        
        # Regular admin cannot deactivate super-admin
        try:
            self.rbac.deactivate_user(admin, "superadmin")
            assert False, "Should have raised SecurityError"
        except SecurityError as e:
            assert "Only super-admins can deactivate super-admin users" in str(e)
        
        # Super-admin can deactivate themselves (edge case)
        self.rbac.deactivate_user(super_admin, "superadmin")
        updated_super_admin = self.rbac.get_user("superadmin")
        assert updated_super_admin.is_active is False


def run_rbac_tests():
    """Run all RBAC tests."""
    print("Running Phase 2 RBAC Tests...")
    
    test_rbac = TestRBACService()
    
    # Test user management
    print("✓ Testing user management...")
    test_rbac.setup_method()
    test_rbac.test_create_user()
    
    test_rbac.setup_method()
    test_rbac.test_create_duplicate_user()
    
    test_rbac.setup_method()
    test_rbac.test_get_user()
    
    test_rbac.setup_method()
    test_rbac.test_create_and_get_session()
    
    # Test permissions
    print("✓ Testing permissions...")
    test_rbac.setup_method()
    test_rbac.test_role_permissions()
    
    test_rbac.setup_method()
    test_rbac.test_inactive_user_permissions()
    
    test_rbac.setup_method()
    test_rbac.test_get_user_permissions()
    
    test_rbac.setup_method()
    test_rbac.test_can_access_resource()
    
    # Test admin functions
    print("✓ Testing admin functions...")
    test_rbac.setup_method()
    test_rbac.test_list_users_admin_only()
    
    test_rbac.setup_method()
    test_rbac.test_update_user_role()
    
    test_rbac.setup_method()
    test_rbac.test_super_admin_restrictions()
    
    test_rbac.setup_method()
    test_rbac.test_deactivate_user()
    
    test_rbac.setup_method()
    test_rbac.test_deactivate_super_admin_restrictions()
    
    print("✅ All RBAC tests passed!")


if __name__ == "__main__":
    run_rbac_tests()