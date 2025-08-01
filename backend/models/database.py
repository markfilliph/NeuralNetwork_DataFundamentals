"""Database models and connection management."""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager

from backend.core.config import settings
from backend.core.exceptions import SecurityError
from backend.utils.cache_utils import cache_query, cache_user_data, CachedRepositoryMixin


class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or "data/app.db"
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self) -> None:
        """Ensure database directory exists."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self) -> None:
        """Initialize database with required tables."""
        with self.get_connection() as conn:
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            # Sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Datasets table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    file_path TEXT,
                    file_size INTEGER,
                    file_hash TEXT,
                    owner_id TEXT NOT NULL,
                    is_encrypted BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (owner_id) REFERENCES users (user_id)
                )
            ''')
            
            # Models table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    model_data TEXT,
                    metrics TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'trained',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id),
                    FOREIGN KEY (owner_id) REFERENCES users (user_id)
                )
            ''')
            
            # Audit logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    resource TEXT,
                    action TEXT,
                    outcome TEXT NOT NULL,
                    risk_level TEXT DEFAULT 'low',
                    details TEXT DEFAULT '{}'
                )
            ''')
            
            # Create optimized indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs (timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs (user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_logs (event_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_outcome ON audit_logs (outcome)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_risk ON audit_logs (risk_level)')
            
            # Composite indexes for common query patterns
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_user_timestamp ON audit_logs (user_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_event_timestamp ON audit_logs (event_type, timestamp)')
            
            # User table indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_role ON users (role)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_active ON users (is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_users_created ON users (created_at)')
            
            # Session table indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions (user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions (expires_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_active ON user_sessions (is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON user_sessions (user_id, is_active)')
            
            # Dataset table indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_datasets_owner ON datasets (owner_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets (created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_datasets_hash ON datasets (file_hash)')
            
            # Model table indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_dataset ON models (dataset_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_owner ON models (owner_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models (model_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_status ON models (status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_models_created ON models (created_at)')
            
            # API keys table indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_apikeys_user ON api_keys (user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_apikeys_active ON api_keys (is_active)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_apikeys_expires ON api_keys (expires_at)')
            
            # API keys table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    name TEXT,
                    permissions TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling and optimizations."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Performance optimizations
            conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
            conn.execute('PRAGMA synchronous=NORMAL')  # Balance safety and speed
            conn.execute('PRAGMA cache_size=10000')  # Increase cache size
            conn.execute('PRAGMA temp_store=MEMORY')  # Use memory for temp tables
            
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount


class UserRepository(CachedRepositoryMixin):
    """Repository for user data operations with caching support."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize user repository."""
        self.db = db_manager
    
    def create_user(self, user_id: str, username: str, email: str, 
                   password_hash: str, password_salt: str, role: str,
                   metadata: Optional[Dict] = None) -> bool:
        """Create a new user.
        
        Args:
            user_id: Unique user identifier
            username: Username
            email: User email
            password_hash: Hashed password
            password_salt: Password salt
            role: User role
            metadata: Optional user metadata
            
        Returns:
            True if user created successfully
            
        Raises:
            DatabaseError: If user creation fails
        """
        try:
            metadata_json = json.dumps(metadata or {})
            
            affected = self.db.execute_update(
                '''INSERT INTO users 
                   (user_id, username, email, password_hash, password_salt, role, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (user_id, username, email, password_hash, password_salt, role, metadata_json)
            )
            
            return affected > 0
            
        except Exception as e:
            raise DatabaseError(f"Failed to create user: {e}")
    
    @cache_user_data(ttl=1800)  # Cache for 30 minutes
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID with optimized query and caching.
        
        Args:
            user_id: User identifier
            
        Returns:
            User dictionary or None if not found
        """
        # Optimized query with LIMIT for single result
        results = self.db.execute_query(
            'SELECT * FROM users WHERE user_id = ? LIMIT 1',
            (user_id,)
        )
        
        if results:
            user = results[0]
            user['metadata'] = json.loads(user['metadata'] or '{}')
            return user
        
        return None
    
    @cache_query(ttl=1800, key_prefix="user_by_username")
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username with optimized query and caching.
        
        Args:
            username: Username
            
        Returns:
            User dictionary or None if not found
        """
        # Optimized query with LIMIT for single result
        results = self.db.execute_query(
            'SELECT * FROM users WHERE username = ? AND is_active = 1 LIMIT 1',
            (username,)
        )
        
        if results:
            user = results[0]
            user['metadata'] = json.loads(user['metadata'] or '{}')
            return user
        
        return None
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """Update user fields.
        
        Args:
            user_id: User identifier
            **kwargs: Fields to update
            
        Returns:
            True if user updated successfully
        """
        if not kwargs:
            return False
        
        # Handle metadata serialization
        if 'metadata' in kwargs:
            kwargs['metadata'] = json.dumps(kwargs['metadata'])
        
        set_clause = ', '.join(f"{key} = ?" for key in kwargs.keys())
        values = list(kwargs.values()) + [user_id]
        
        affected = self.db.execute_update(
            f'UPDATE users SET {set_clause} WHERE user_id = ?',
            tuple(values)
        )
        
        return affected > 0
    
    @cache_query(ttl=300, key_prefix="user_list")  # Cache for 5 minutes
    def list_users(self, limit: int = 100, offset: int = 0, active_only: bool = True) -> List[Dict]:
        """List users with optimized pagination and caching.
        
        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            active_only: Only return active users
            
        Returns:
            List of user dictionaries
        """
        where_clause = "WHERE is_active = 1" if active_only else ""
        
        # Only select needed columns for listing
        results = self.db.execute_query(
            f'''SELECT user_id, username, email, role, is_active, created_at, last_login 
               FROM users {where_clause} 
               ORDER BY created_at DESC 
               LIMIT ? OFFSET ?''',
            (limit, offset)
        )
        
        return results
    
    def get_users_count(self, active_only: bool = True) -> int:
        """Get total count of users for pagination.
        
        Args:
            active_only: Only count active users
            
        Returns:
            Total number of users
        """
        where_clause = "WHERE is_active = 1" if active_only else ""
        
        results = self.db.execute_query(
            f'SELECT COUNT(*) as count FROM users {where_clause}'
        )
        
        return results[0]['count'] if results else 0
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user deactivated successfully
        """
        affected = self.db.execute_update(
            'UPDATE users SET is_active = 0 WHERE user_id = ?',
            (user_id,)
        )
        
        return affected > 0


class SessionRepository:
    """Repository for session data operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize session repository."""
        self.db = db_manager
    
    def create_session(self, session_id: str, user_id: str, 
                      expires_at: Optional[datetime] = None,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> bool:
        """Create a new session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            expires_at: Session expiration time
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if session created successfully
        """
        affected = self.db.execute_update(
            '''INSERT INTO user_sessions 
               (session_id, user_id, expires_at, ip_address, user_agent)
               VALUES (?, ?, ?, ?, ?)''',
            (session_id, user_id, expires_at, ip_address, user_agent)
        )
        
        return affected > 0
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID with optimized query and expiration check.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session dictionary or None if not found/expired
        """
        # Optimized query that checks both active status and expiration
        results = self.db.execute_query(
            '''SELECT * FROM user_sessions 
               WHERE session_id = ? 
               AND is_active = 1 
               AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
               LIMIT 1''',
            (session_id,)
        )
        
        return results[0] if results else None
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session destroyed successfully
        """
        affected = self.db.execute_update(
            'UPDATE user_sessions SET is_active = 0 WHERE session_id = ?',
            (session_id,)
        )
        
        return affected > 0
    
    def destroy_user_sessions(self, user_id: str) -> int:
        """Destroy all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of sessions destroyed
        """
        affected = self.db.execute_update(
            'UPDATE user_sessions SET is_active = 0 WHERE user_id = ?',
            (user_id,)
        )
        
        return affected
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        affected = self.db.execute_update(
            '''UPDATE user_sessions 
               SET is_active = 0 
               WHERE expires_at < CURRENT_TIMESTAMP AND is_active = 1'''
        )
        
        return affected


class AuditLogRepository:
    """Repository for audit log operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize audit log repository."""
        self.db = db_manager
    
    def create_log(self, event_type: str, outcome: str, 
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   ip_address: Optional[str] = None,
                   user_agent: Optional[str] = None,
                   resource: Optional[str] = None,
                   action: Optional[str] = None,
                   risk_level: str = 'low',
                   details: Optional[Dict] = None) -> bool:
        """Create audit log entry.
        
        Args:
            event_type: Type of event
            outcome: Event outcome
            user_id: User identifier
            session_id: Session identifier
            ip_address: Client IP address
            user_agent: Client user agent
            resource: Resource accessed
            action: Action performed
            risk_level: Risk level
            details: Additional event details
            
        Returns:
            True if log created successfully
        """
        details_json = json.dumps(details or {})
        
        affected = self.db.execute_update(
            '''INSERT INTO audit_logs 
               (event_type, user_id, session_id, ip_address, user_agent, 
                resource, action, outcome, risk_level, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (event_type, user_id, session_id, ip_address, user_agent,
             resource, action, outcome, risk_level, details_json)
        )
        
        return affected > 0
    
    def get_logs(self, limit: int = 100, offset: int = 0,
                event_type: Optional[str] = None,
                user_id: Optional[str] = None,
                start_time: Optional[str] = None,
                end_time: Optional[str] = None,
                risk_level: Optional[str] = None) -> List[Dict]:
        """Get audit logs with optimized filtering and indexing.
        
        Args:
            limit: Maximum number of logs to return
            offset: Number of logs to skip
            event_type: Filter by event type
            user_id: Filter by user ID
            start_time: Start time filter
            end_time: End time filter
            risk_level: Filter by risk level
            
        Returns:
            List of audit log dictionaries
        """
        where_conditions = []
        params = []
        
        # Order conditions by selectivity for better index usage
        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)
        
        if event_type:
            where_conditions.append("event_type = ?")
            params.append(event_type)
        
        if risk_level:
            where_conditions.append("risk_level = ?")
            params.append(risk_level)
        
        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)
        
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        params.extend([limit, offset])
        
        # Use optimized query with proper index hints
        query = f'''
            SELECT log_id, timestamp, event_type, user_id, session_id, ip_address, 
                   user_agent, resource, action, outcome, risk_level, details
            FROM audit_logs 
            {where_clause}
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        '''
        
        results = self.db.execute_query(query, tuple(params))
        
        for log in results:
            log['details'] = json.loads(log['details'] or '{}')
        
        return results
    
    def get_logs_count(self, event_type: Optional[str] = None,
                      user_id: Optional[str] = None,
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None,
                      risk_level: Optional[str] = None) -> int:
        """Get count of audit logs matching filters (optimized for pagination).
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_time: Start time filter
            end_time: End time filter
            risk_level: Filter by risk level
            
        Returns:
            Count of matching logs
        """
        where_conditions = []
        params = []
        
        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)
        
        if event_type:
            where_conditions.append("event_type = ?")
            params.append(event_type)
        
        if risk_level:
            where_conditions.append("risk_level = ?")
            params.append(risk_level)
        
        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)
        
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        query = f'SELECT COUNT(*) as count FROM audit_logs {where_clause}'
        
        results = self.db.execute_query(query, tuple(params))
        return results[0]['count'] if results else 0


class DatabaseOptimizer:
    """Database optimization utilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize database optimizer."""
        self.db = db_manager
    
    def analyze_database(self) -> Dict[str, Any]:
        """Analyze database performance and provide recommendations.
        
        Returns:
            Dictionary with analysis results and recommendations
        """
        analysis = {}
        
        with self.db.get_connection() as conn:
            # Table sizes
            cursor = conn.execute(
                "SELECT name, COUNT(*) as count FROM sqlite_master WHERE type='table' GROUP BY name"
            )
            table_info = cursor.fetchall()
            
            analysis['tables'] = {}
            for table in table_info:
                table_name = table['name']
                count_cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                row_count = count_cursor.fetchone()['count']
                analysis['tables'][table_name] = {'row_count': row_count}
            
            # Index usage statistics (SQLite STAT tables)
            try:
                cursor = conn.execute("SELECT * FROM sqlite_stat1")
                stat_info = cursor.fetchall()
                analysis['index_stats'] = [dict(row) for row in stat_info]
            except sqlite3.Error:
                analysis['index_stats'] = []
        
        return analysis
    
    def vacuum_database(self) -> bool:
        """Vacuum database to reclaim space and optimize.
        
        Returns:
            True if vacuum completed successfully
        """
        try:
            with self.db.get_connection() as conn:
                conn.execute('VACUUM')
                conn.execute('ANALYZE')
            return True
        except Exception:
            return False
    
    def get_slow_queries(self) -> List[str]:
        """Get recommendations for query optimization.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze table sizes and suggest optimizations
        analysis = self.analyze_database()
        
        for table_name, info in analysis.get('tables', {}).items():
            if info['row_count'] > 10000:
                recommendations.append(
                    f"Consider partitioning {table_name} table (has {info['row_count']} rows)"
                )
        
        if analysis.get('tables', {}).get('audit_logs', {}).get('row_count', 0) > 100000:
            recommendations.append(
                "Consider implementing audit log archiving for better performance"
            )
        
        return recommendations


# Global database instances
db_manager = DatabaseManager()
user_repository = UserRepository(db_manager)
session_repository = SessionRepository(db_manager)
audit_log_repository = AuditLogRepository(db_manager)
db_optimizer = DatabaseOptimizer(db_manager)