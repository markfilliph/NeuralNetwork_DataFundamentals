"""Role-Based Access Control (RBAC) service."""

from enum import Enum
from typing import Dict, List, Set, Any, Optional
from functools import wraps
from dataclasses import dataclass

from backend.core.exceptions import SecurityError
from backend.services.auth_service import AuthenticationError


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


class RBACService:
    """Role-Based Access Control service."""
    
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
        """Initialize RBAC service."""
        self._users: Dict[str, User] = {}
        self._user_sessions: Dict[str, str] = {}  # session_id -> user_id
    
    def create_user(self, user_id: str, username: str, email: str, role: Role) -> User:
        """Create a new user.
        
        Args:
            user_id: Unique user identifier
            username: Username
            email: User email
            role: User role
            
        Returns:
            Created user object
        """
        if user_id in self._users:
            raise SecurityError(f"User {user_id} already exists")
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role
        )
        
        self._users[user_id] = user
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User object or None if not found
        """
        return self._users.get(user_id)
    
    def get_user_by_session(self, session_id: str) -> Optional[User]:
        """Get user by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            User object or None if not found
        """
        user_id = self._user_sessions.get(session_id)
        if user_id:
            return self.get_user(user_id)
        return None
    
    def create_session(self, user_id: str, session_id: str) -> None:
        """Create a user session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        if user_id in self._users:
            self._user_sessions[session_id] = user_id
    
    def destroy_session(self, session_id: str) -> None:
        """Destroy a user session.
        
        Args:
            session_id: Session identifier
        """
        self._user_sessions.pop(session_id, None)
    
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
        """Get all permissions for a user.
        
        Args:
            user: User object
            
        Returns:
            Set of permissions
        """
        if not user.is_active:
            return set()
        
        return self.ROLE_PERMISSIONS.get(user.role, set())
    
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
    
    def require_permission(self, permission: Permission):
        """Decorator to require a specific permission.
        
        Args:
            permission: Permission required
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user from kwargs (set by auth decorator)
                current_user_data = kwargs.get('current_user', {})
                if not current_user_data:
                    raise AuthenticationError("Authentication required")
                
                # Get user object
                user_id = current_user_data.get('user_id')
                if not user_id:
                    raise AuthenticationError("Invalid user token")
                
                user = self.get_user(user_id)
                if not user:
                    raise SecurityError("User not found")
                
                # Check permission
                if not self.has_permission(user, permission):
                    raise SecurityError(f"Insufficient permissions: {permission.value} required")
                
                # Add user object to kwargs
                kwargs['current_user_obj'] = user
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def require_role(self, required_role: Role):
        """Decorator to require a specific role or higher.
        
        Args:
            required_role: Minimum role required
            
        Returns:
            Decorator function
        """
        # Define role hierarchy
        role_hierarchy = {
            Role.VIEWER: 1,
            Role.ANALYST: 2,
            Role.ADMIN: 3,
            Role.SUPER_ADMIN: 4,
        }
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user from kwargs
                current_user_data = kwargs.get('current_user', {})
                if not current_user_data:
                    raise AuthenticationError("Authentication required")
                
                user_id = current_user_data.get('user_id')
                if not user_id:
                    raise AuthenticationError("Invalid user token")
                
                user = self.get_user(user_id)
                if not user:
                    raise SecurityError("User not found")
                
                # Check role hierarchy
                user_level = role_hierarchy.get(user.role, 0)
                required_level = role_hierarchy.get(required_role, 5)
                
                if user_level < required_level:
                    raise SecurityError(f"Insufficient role: {required_role.value} or higher required")
                
                kwargs['current_user_obj'] = user
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def list_users(self, requesting_user: User) -> List[User]:
        """List all users (admin only).
        
        Args:
            requesting_user: User making the request
            
        Returns:
            List of users
            
        Raises:
            SecurityError: If user lacks permission
        """
        if not self.has_permission(requesting_user, Permission.MANAGE_USERS):
            raise SecurityError("Permission denied: cannot list users")
        
        return list(self._users.values())
    
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
        
        target_user.is_active = False
        
        # Remove all sessions for deactivated user
        sessions_to_remove = [
            sid for sid, uid in self._user_sessions.items() 
            if uid == target_user_id
        ]
        for session_id in sessions_to_remove:
            self.destroy_session(session_id)
        
        return target_user


# Global RBAC service instance
rbac_service = RBACService()