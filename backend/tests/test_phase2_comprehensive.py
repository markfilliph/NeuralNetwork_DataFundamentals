"""Comprehensive test suite for Phase 2: Authentication & Authorization."""

import sys
from pathlib import Path

# Add backend to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all test modules
from backend.tests.test_auth_service import run_auth_tests
from backend.tests.test_rbac_service import run_rbac_tests
from backend.tests.test_middleware import run_middleware_tests
from backend.tests.test_audit_logging import run_audit_logging_tests


def run_integration_tests():
    """Run integration tests that test multiple components together."""
    print("Running Phase 2 Integration Tests...")
    
    from backend.services.auth_service import AuthenticationService
    from backend.services.rbac_service import rbac_service, Role, Permission
    from backend.api.middleware import require_permission, require_role
    from backend.core.logging import audit_logger, EventType
    
    # Test 1: Complete authentication flow
    print("✓ Testing complete authentication flow...")
    
    # Create user
    user = rbac_service.create_user(
        user_id="integration_user",
        username="integrationuser", 
        email="integration@test.com",
        role=Role.ANALYST
    )
    
    # Create token
    auth_service = AuthenticationService()
    token = auth_service.create_access_token({
        "user_id": "integration_user",
        "username": "integrationuser"
    })
    
    # Test protected function
    @require_permission(Permission.TRAIN_MODELS)
    def protected_function(**kwargs):
        user_obj = kwargs.get('current_user_obj')
        audit_logger.log_event(
            EventType.MODEL_TRAINED,
            user_id=user_obj.user_id,
            outcome="success",
            details={"model_type": "integration_test"}
        )
        return f"Success: {user_obj.username}"
    
    result = protected_function(token=token)
    assert "Success: integrationuser" == result
    
    # Test 2: Role hierarchy enforcement
    print("✓ Testing role hierarchy...")
    
    # Create users with different roles
    viewer = rbac_service.create_user("viewer", "viewer", "viewer@test.com", Role.VIEWER)
    admin = rbac_service.create_user("admin", "admin", "admin@test.com", Role.ADMIN)
    
    @require_role(Role.ADMIN)
    def admin_function(**kwargs):
        return "Admin operation"
    
    # Admin should work
    admin_token = auth_service.create_access_token({"user_id": "admin"})
    result = admin_function(token=admin_token)
    assert result == "Admin operation"
    
    # Viewer should fail
    viewer_token = auth_service.create_access_token({"user_id": "viewer"})
    try:
        admin_function(token=viewer_token)
        assert False, "Should have failed"
    except Exception:
        pass  # Expected
    
    # Test 3: Audit logging integration
    print("✓ Testing audit logging integration...")
    
    # Log authentication event
    audit_logger.log_authentication_event(
        success=True,
        user_id="integration_user",
        ip_address="192.168.1.1"
    )
    
    # Check that event was logged
    events = audit_logger.get_events(user_id="integration_user", limit=10)
    auth_events = [e for e in events if e.event_type == EventType.LOGIN_SUCCESS]
    assert len(auth_events) >= 1
    
    print("✅ All integration tests passed!")


def run_security_tests():
    """Run security-focused tests."""
    print("Running Phase 2 Security Tests...")
    
    from backend.services.auth_service import AuthenticationService, InvalidTokenError
    from backend.services.rbac_service import rbac_service, Role, Permission
    from backend.core.logging import audit_logger, EventType
    
    # Test 1: Token tampering detection
    print("✓ Testing token tampering detection...")
    auth_service = AuthenticationService()
    token = auth_service.create_access_token({"user_id": "test"})
    
    # Tamper with token
    tampered_token = token[:-5] + "XXXXX"
    
    try:
        auth_service.verify_token(tampered_token)
        assert False, "Should have detected tampering"
    except InvalidTokenError:
        pass  # Expected
    
    # Test 2: Inactive user cannot access resources
    print("✓ Testing inactive user access denial...")
    user = rbac_service.create_user("inactive_test", "inactive", "inactive@test.com", Role.ADMIN)
    user.is_active = False
    
    # Even admin permissions shouldn't work for inactive user
    assert not rbac_service.has_permission(user, Permission.MANAGE_USERS)
    
    # Test 3: Session management security
    print("✓ Testing session management...")
    active_user = rbac_service.create_user("session_test", "session", "session@test.com", Role.ANALYST)
    admin_user = rbac_service.create_user("session_admin", "admin", "admin@test.com", Role.ADMIN)
    
    # Create session
    rbac_service.create_session(active_user.user_id, "session123")
    assert rbac_service.get_user_by_session("session123") is not None
    
    # Deactivate user should destroy sessions (admin can deactivate others)
    rbac_service.deactivate_user(admin_user, active_user.user_id)
    assert rbac_service.get_user_by_session("session123") is None
    
    # Test 4: Security event logging
    print("✓ Testing security event logging...")
    audit_logger.log_security_event(
        EventType.SUSPICIOUS_ACTIVITY,
        user_id="security_test",
        details={"activity": "multiple_failed_logins", "count": 5},
        risk_level="high"
    )
    
    # Verify security events are properly categorized
    summary = audit_logger.get_security_summary(hours=1)
    assert summary["security_events"] >= 1
    
    print("✅ All security tests passed!")


def run_performance_tests():
    """Run performance-related tests."""
    print("Running Phase 2 Performance Tests...")
    
    import time
    from backend.services.auth_service import AuthenticationService
    from backend.services.rbac_service import rbac_service, Role
    from backend.core.logging import audit_logger, EventType
    
    # Test 1: Token creation/verification performance
    print("✓ Testing token performance...")
    auth_service = AuthenticationService()
    
    start_time = time.time()
    
    # Create and verify 100 tokens
    for i in range(100):
        token = auth_service.create_access_token({"user_id": f"perf_user_{i}"})
        payload = auth_service.verify_token(token)
        assert payload["user_id"] == f"perf_user_{i}"
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should complete in reasonable time (less than 1 second for 100 operations)
    assert duration < 1.0, f"Token operations too slow: {duration}s"
    
    # Test 2: User creation performance
    print("✓ Testing user creation performance...")
    rbac = rbac_service
    
    start_time = time.time()
    
    # Create 50 users
    for i in range(50):
        rbac.create_user(f"perf_user_{i}", f"user{i}", f"user{i}@test.com", Role.ANALYST)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should complete quickly
    assert duration < 0.5, f"User creation too slow: {duration}s"
    
    # Test 3: Audit logging performance
    print("✓ Testing audit logging performance...")
    
    start_time = time.time()
    
    # Log 200 events
    for i in range(200):
        audit_logger.log_event(
            EventType.DATA_ACCESSED,
            user_id=f"perf_user_{i % 10}",
            details={"index": i}
        )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should complete quickly
    assert duration < 1.0, f"Audit logging too slow: {duration}s"
    
    print("✅ All performance tests passed!")


def run_all_phase2_tests():
    """Run all Phase 2 tests."""
    print("=" * 60)
    print("PHASE 2: AUTHENTICATION & AUTHORIZATION - COMPREHENSIVE TESTS")
    print("=" * 60)
    
    try:
        # Core component tests
        run_auth_tests()
        print()
        
        run_rbac_tests() 
        print()
        
        run_middleware_tests()
        print()
        
        run_audit_logging_tests()
        print()
        
        # Integration tests
        run_integration_tests()
        print()
        
        # Security tests
        run_security_tests()
        print()
        
        # Performance tests
        run_performance_tests()
        print()
        
        print("=" * 60)
        print("🎉 ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Phase 2 Implementation Summary:")
        print("✅ JWT-based Authentication System")
        print("✅ Role-Based Access Control (RBAC)")
        print("✅ Authentication Middleware & Decorators")
        print("✅ Comprehensive Audit Logging")
        print("✅ Security Event Monitoring")
        print("✅ Rate Limiting")
        print("✅ Session Management")
        print("✅ Integration & Security Tests")
        print("✅ Performance Validation")
        print()
        print("Ready for Phase 3: Data Encryption!")
        
        return True
        
    except Exception as e:
        print(f"❌ PHASE 2 TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_phase2_tests()
    sys.exit(0 if success else 1)