"""Database-backed Role-Based Access Control (RBAC) service."""

from enum import Enum
from typing import Dict, List, Set, Any, Optional
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.core.exceptions import SecurityError
from backend.services.auth_service import AuthenticationError, AuthenticationService
from backend.models.database import user_repository, session_repository
from backend.core.logging import audit_logger, EventType
from backend.services.cache_service import cache_service


class Permission(Enum):
    """System permissions."""
    
    # Data permissions
    READ_DATA = "read_data"
    UPLOAD_DATA = "upload_data"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"
    
    # Model permissions
    VIEW_MODELS = "view_models"
    TRAIN_MODELS = "train_models"
    DELETE_MODELS = "delete_models"
    
    # System permissions
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    SYSTEM_CONFIG = "system_config"


class Role(Enum):
    """System roles with their permissions."""
    
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class User:
    """User data class."""
    
    user_id: str
    username: str
    email: str
    role: Role
    is_active: bool = True
    created_at: Optional[str] = None
    last_login: Optional[str] = None
    metadata: Optional[Dict] = None
    
    @classmethod
    def from_db_dict(cls, db_user: Dict) -> 'User':
        """Create User from database dictionary."""
        return cls(
            user_id=db_user['user_id'],
            username=db_user['username'],
            email=db_user['email'],
            role=Role(db_user['role']),
            is_active=bool(db_user['is_active']),
            created_at=db_user['created_at'],
            last_login=db_user['last_login'],
            metadata=db_user.get('metadata', {})
        )


class DatabaseRBACService:
    """Database-backed Role-Based Access Control service."""
    
    # Define role permissions mapping
    ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
        Role.VIEWER: {
            Permission.READ_DATA,
            Permission.VIEW_MODELS,
        },
        Role.ANALYST: {
            Permission.READ_DATA,
            Permission.UPLOAD_DATA,
            Permission.EXPORT_DATA,
            Permission.VIEW_MODELS,
            Permission.TRAIN_MODELS,
        },
        Role.ADMIN: {
            Permission.READ_DATA,
            Permission.UPLOAD_DATA,
            Permission.DELETE_DATA,
            Permission.EXPORT_DATA,
            Permission.VIEW_MODELS,
            Permission.TRAIN_MODELS,
            Permission.DELETE_MODELS,
            Permission.MANAGE_USERS,
            Permission.VIEW_AUDIT_LOGS,
        },
        Role.SUPER_ADMIN: set(Permission),  # All permissions
    }
    
    def __init__(self):
        """Initialize database RBAC service."""
        self.auth_service = AuthenticationService()
    
    def create_user(self, user_id: str, username: str, email: str, 
                   password: str, role: Role, metadata: Optional[Dict] = None) -> User:
        """Create a new user with password.
        
        Args:
            user_id: Unique user identifier
            username: Username
            email: User email
            password: Plain text password
            role: User role
            metadata: Optional user metadata
            
        Returns:
            Created user object
            
        Raises:
            SecurityError: If user creation fails
        """
        # Check if user already exists
        existing_user = self.get_user(user_id)
        if existing_user:
            raise SecurityError(f"User {user_id} already exists")
        
        # Check username uniqueness
        existing_username = user_repository.get_user_by_username(username)
        if existing_username:
            raise SecurityError(f"Username {username} already exists")
        
        try:
            # Hash password
            password_hash, password_salt = self.auth_service.hash_password(password)
            
            # Create user in database
            success = user_repository.create_user(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                password_salt=password_salt,
                role=role.value,
                metadata=metadata
            )
            
            if not success:
                raise SecurityError("Failed to create user in database")
            
            # Log user creation
            audit_logger.log_event(
                EventType.USER_CREATED,
                outcome="success",
                resource=user_id,
                details={
                    "username": username,
                    "email": email,
                    "role": role.value
                }
            )
            
            # Return user object
            return User(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                metadata=metadata or {}
            )
            
        except Exception as e:
            raise SecurityError(f"Failed to create user: {e}")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User object or None if not found
        """
        db_user = user_repository.get_user(user_id)
        if db_user:
            return User.from_db_dict(db_user)
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User object or None if not found
        """
        db_user = user_repository.get_user_by_username(username)
        if db_user:
            return User.from_db_dict(db_user)
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            # Get user from database
            db_user = user_repository.get_user_by_username(username)
            if not db_user:
                audit_logger.log_authentication_event(
                    success=False,
                    user_id=username,
                    details={"error": "user_not_found"}
                )
                return None
            
            # Check if user is active
            if not db_user['is_active']:
                audit_logger.log_authentication_event(
                    success=False,
                    user_id=db_user['user_id'],
                    details={"error": "user_inactive"}
                )
                return None
            
            # Verify password
            password_valid = self.auth_service.verify_password(
                password, 
                db_user['password_hash'], 
                db_user['password_salt']
            )
            
            if password_valid:
                # Update last login
                user_repository.update_user(
                    db_user['user_id'],
                    last_login=datetime.utcnow().isoformat()
                )
                
                # Log successful authentication
                audit_logger.log_authentication_event(
                    success=True,
                    user_id=db_user['user_id'],
                    details={"username": username}
                )
                
                return User.from_db_dict(db_user)
            else:
                # Log failed authentication
                audit_logger.log_authentication_event(
                    success=False,
                    user_id=db_user['user_id'],
                    details={"error": "invalid_password", "username": username}
                )
                return None
                
        except Exception as e:
            audit_logger.log_authentication_event(
                success=False,
                user_id=username,
                details={"error": "authentication_exception", "exception": str(e)}
            )
            return None
    
    def create_session(self, user_id: str, session_id: str, 
                      expires_at: Optional[datetime] = None,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> bool:
        """Create a user session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            expires_at: Session expiration time
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if session created successfully
        """
        return session_repository.create_session(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def get_user_by_session(self, session_id: str) -> Optional[User]:
        """Get user by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            User object or None if not found
        """
        session = session_repository.get_session(session_id)
        if session:
            return self.get_user(session['user_id'])
        return None
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a user session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session destroyed successfully
        """
        return session_repository.destroy_session(session_id)
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission.
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        if not user.is_active:
            return False
        
        user_permissions = self.ROLE_PERMISSIONS.get(user.role, set())
        return permission in user_permissions
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for a user with caching.
        
        Args:
            user: User object
            
        Returns:
            Set of permissions
        """
        if not user.is_active:
            return set()
        
        # Try to get from cache first
        cache_key = f"user_permissions:{user.user_id}:{user.role.value}"
        cached_permissions = cache_service.get(cache_key)
        
        if cached_permissions is not None:
            # Convert back to Permission enum set
            return {Permission(perm) for perm in cached_permissions}
        
        # Get permissions from role mapping
        permissions = self.ROLE_PERMISSIONS.get(user.role, set())
        
        # Cache the permissions (convert enum values to strings for JSON serialization)
        permission_strings = [perm.value for perm in permissions]
        cache_service.set(cache_key, permission_strings, ttl=1800)  # 30 minutes
        
        return permissions
    
    def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate cached data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if cache was invalidated
        """
        success = True
        
        # Invalidate permissions cache for all roles (we don't know current role)
        for role in Role:
            cache_key = f"user_permissions:{user_id}:{role.value}"
            if not cache_service.delete(cache_key):
                success = False
        
        # Invalidate session cache
        if not cache_service.invalidate_user_session(user_id):
            success = False
        
        return success
    
    def can_access_resource(self, user: User, resource_type: str, action: str) -> bool:
        """Check if user can access a specific resource.
        
        Args:
            user: User object
            resource_type: Type of resource (data, model, user, etc.)
            action: Action to perform (read, write, delete, etc.)
            
        Returns:
            True if access allowed, False otherwise
        """
        # Map resource/action combinations to permissions
        permission_map = {
            ("data", "read"): Permission.READ_DATA,
            ("data", "upload"): Permission.UPLOAD_DATA,
            ("data", "delete"): Permission.DELETE_DATA,
            ("data", "export"): Permission.EXPORT_DATA,
            ("model", "view"): Permission.VIEW_MODELS,
            ("model", "train"): Permission.TRAIN_MODELS,
            ("model", "delete"): Permission.DELETE_MODELS,
            ("user", "manage"): Permission.MANAGE_USERS,
            ("audit", "view"): Permission.VIEW_AUDIT_LOGS,
            ("system", "config"): Permission.SYSTEM_CONFIG,
        }
        
        required_permission = permission_map.get((resource_type, action))
        if required_permission is None:
            return False
        
        return self.has_permission(user, required_permission)
    
    def list_users(self, requesting_user: User, limit: int = 100, offset: int = 0) -> List[User]:
        """List all users (admin only).
        
        Args:
            requesting_user: User making the request
            limit: Maximum number of users to return
            offset: Number of users to skip
            
        Returns:
            List of users
            
        Raises:
            SecurityError: If user lacks permission
        """
        if not self.has_permission(requesting_user, Permission.MANAGE_USERS):
            raise SecurityError("Permission denied: cannot list users")
        
        db_users = user_repository.list_users(limit=limit, offset=offset)
        return [User.from_db_dict(db_user) for db_user in db_users]
    
    def update_user_role(self, requesting_user: User, target_user_id: str, new_role: Role) -> User:
        """Update a user's role (admin only).
        
        Args:
            requesting_user: User making the request
            target_user_id: ID of user to update
            new_role: New role to assign
            
        Returns:
            Updated user object
            
        Raises:
            SecurityError: If user lacks permission or target not found
        """
        if not self.has_permission(requesting_user, Permission.MANAGE_USERS):
            raise SecurityError("Permission denied: cannot update user roles")
        
        target_user = self.get_user(target_user_id)
        if not target_user:
            raise SecurityError(f"User {target_user_id} not found")
        
        # Prevent non-super-admins from creating super-admins
        if new_role == Role.SUPER_ADMIN and requesting_user.role != Role.SUPER_ADMIN:
            raise SecurityError("Only super-admins can create super-admin users")
        
        # Update in database
        success = user_repository.update_user(target_user_id, role=new_role.value)
        if not success:
            raise SecurityError("Failed to update user role")
        
        # Log role change
        audit_logger.log_event(
            EventType.USER_MODIFIED,
            user_id=requesting_user.user_id,
            outcome="success",
            resource=target_user_id,
            action="role_change",
            details={
                "old_role": target_user.role.value,
                "new_role": new_role.value,
                "target_username": target_user.username
            }
        )
        
        # Return updated user
        target_user.role = new_role
        return target_user
    
    def deactivate_user(self, requesting_user: User, target_user_id: str) -> User:
        """Deactivate a user (admin only).
        
        Args:
            requesting_user: User making the request
            target_user_id: ID of user to deactivate
            
        Returns:
            Updated user object
            
        Raises:
            SecurityError: If user lacks permission or target not found
        """
        if not self.has_permission(requesting_user, Permission.MANAGE_USERS):
            raise SecurityError("Permission denied: cannot deactivate users")
        
        target_user = self.get_user(target_user_id)
        if not target_user:
            raise SecurityError(f"User {target_user_id} not found")
        
        # Prevent deactivating super-admins (unless by another super-admin)
        if target_user.role == Role.SUPER_ADMIN and requesting_user.role != Role.SUPER_ADMIN:
            raise SecurityError("Only super-admins can deactivate super-admin users")
        
        # Deactivate in database
        success = user_repository.deactivate_user(target_user_id)
        if not success:
            raise SecurityError("Failed to deactivate user")
        
        # Destroy all user sessions
        session_repository.destroy_user_sessions(target_user_id)
        
        # Log user deactivation
        audit_logger.log_event(
            EventType.USER_DEACTIVATED,
            user_id=requesting_user.user_id,
            outcome="success",
            resource=target_user_id,
            action="deactivate",
            details={
                "target_username": target_user.username,
                "target_role": target_user.role.value
            }
        )
        
        target_user.is_active = False
        return target_user
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        return session_repository.cleanup_expired_sessions()


# Global database-backed RBAC service instance
db_rbac_service = DatabaseRBACService()