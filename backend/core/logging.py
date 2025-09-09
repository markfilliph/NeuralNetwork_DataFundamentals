"""Comprehensive logging and audit system."""

import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from backend.core.config import settings


class LogLevel(Enum):
    """Log levels for audit events."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Types of audit events."""
    
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TOKEN_EXPIRED = "token_expired"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION = "permission_escalation"
    
    # Data events
    DATA_UPLOADED = "data_uploaded"
    DATA_ACCESSED = "data_accessed"
    DATA_ACCESS = "data_access"  # Alias for DATA_ACCESSED
    DATA_ANALYZED = "data_analyzed"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    
    # Model events
    MODEL_TRAINED = "model_trained"
    MODEL_ACCESSED = "model_accessed"
    MODEL_DELETED = "model_deleted"
    PREDICTION_MADE = "prediction_made"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    MALWARE_DETECTED = "malware_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_DEACTIVATED = "user_deactivated"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    
    timestamp: str
    event_type: EventType
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    risk_level: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        return result
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.setup_logging()
        self._events: List[AuditEvent] = []
        self._max_events = 10000  # Keep last 10k events in memory
    
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure audit logger
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        audit_handler = logging.FileHandler(f"{log_dir}/audit.log")
        audit_handler.setLevel(logging.INFO)
        
        # Security event handler (separate file)
        security_handler = logging.FileHandler(f"{log_dir}/security.log")
        security_handler.setLevel(logging.WARNING)
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
        )
        audit_handler.setFormatter(formatter)
        security_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.audit_logger.handlers:
            self.audit_logger.addHandler(audit_handler)
            self.audit_logger.addHandler(security_handler)
        
        # Configure application logger
        self.app_logger = logging.getLogger("app")
        self.app_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        
        # Console handler for application logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        if not self.app_logger.handlers:
            self.app_logger.addHandler(console_handler)
    
    def log_event(self, 
                  event_type: EventType,
                  outcome: str = "success",
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  resource: Optional[str] = None,
                  action: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  risk_level: str = "low") -> None:
        """Log an audit event.
        
        Args:
            event_type: Type of event
            outcome: Event outcome (success, failure, error)
            user_id: User identifier
            session_id: Session identifier
            ip_address: Client IP address
            user_agent: Client user agent
            resource: Resource being accessed
            action: Action being performed
            details: Additional event details
            risk_level: Risk level of the event
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            risk_level=risk_level
        )
        
        # Add to memory store
        self._events.append(event)
        
        # Trim events if necessary
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        
        # Log to file
        log_level = self._get_log_level(event_type, outcome, risk_level)
        
        if log_level == LogLevel.CRITICAL or risk_level == "critical":
            self.audit_logger.critical(event.to_json())
        elif log_level == LogLevel.ERROR or outcome == "error":
            self.audit_logger.error(event.to_json())
        elif log_level == LogLevel.WARNING or risk_level in ["medium", "high"]:
            self.audit_logger.warning(event.to_json())
        else:
            self.audit_logger.info(event.to_json())
        
        # Also persist to database
        self._persist_to_database(event)
    
    def _get_log_level(self, event_type: EventType, outcome: str, risk_level: str) -> LogLevel:
        """Determine log level based on event characteristics.
        
        Args:
            event_type: Type of event
            outcome: Event outcome
            risk_level: Risk level
            
        Returns:
            Log level
        """
        # Critical events
        if risk_level == "critical" or event_type in [
            EventType.MALWARE_DETECTED,
            EventType.SECURITY_VIOLATION
        ]:
            return LogLevel.CRITICAL
        
        # Error events
        if outcome == "error" or event_type in [
            EventType.LOGIN_FAILED,
            EventType.ACCESS_DENIED
        ]:
            return LogLevel.ERROR
        
        # Warning events
        if risk_level in ["medium", "high"] or event_type in [
            EventType.TOKEN_EXPIRED,
            EventType.RATE_LIMIT_EXCEEDED,
            EventType.SUSPICIOUS_ACTIVITY
        ]:
            return LogLevel.WARNING
        
        return LogLevel.INFO
    
    def _persist_to_database(self, event: AuditEvent) -> None:
        """Persist audit event to database.
        
        Args:
            event: Audit event to persist
        """
        try:
            # Import here to avoid circular imports
            import backend.models.database as db_module
            if hasattr(db_module, 'audit_log_repository'):
                audit_log_repository = db_module.audit_log_repository
                audit_log_repository.create_log(
                    event_type=event.event_type.value,
                    outcome=event.outcome,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    resource=event.resource,
                    action=event.action,
                    risk_level=event.risk_level,
                    details=event.details
                )
        except Exception as e:
            # Don't let database errors break audit logging
            self.app_logger.error(f"Failed to persist audit log to database: {e}")
    
    def log_security_event(self, 
                          event_type: EventType,
                          user_id: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None,
                          ip_address: Optional[str] = None,
                          risk_level: str = "high") -> None:
        """Log a security-related event.
        
        Args:
            event_type: Security event type
            user_id: User identifier
            details: Event details
            ip_address: Client IP address
            risk_level: Risk level
        """
        self.log_event(
            event_type=event_type,
            outcome="security_event",
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            risk_level=risk_level
        )
    
    def log_authentication_event(self,
                                success: bool,
                                user_id: Optional[str] = None,
                                ip_address: Optional[str] = None,
                                details: Optional[Dict[str, Any]] = None) -> None:
        """Log authentication event.
        
        Args:
            success: Whether authentication succeeded
            user_id: User identifier
            ip_address: Client IP address
            details: Additional details
        """
        event_type = EventType.LOGIN_SUCCESS if success else EventType.LOGIN_FAILED
        outcome = "success" if success else "failure"
        risk_level = "low" if success else "medium"
        
        self.log_event(
            event_type=event_type,
            outcome=outcome,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            risk_level=risk_level
        )
    
    def log_data_access(self,
                       user_id: str,
                       dataset_id: str,
                       operation: str,
                       outcome: str = "success",
                       details: Optional[Dict[str, Any]] = None) -> None:
        """Log data access event.
        
        Args:
            user_id: User identifier
            dataset_id: Dataset identifier
            operation: Operation performed
            outcome: Operation outcome
            details: Additional details
        """
        event_type_map = {
            "upload": EventType.DATA_UPLOADED,
            "read": EventType.DATA_ACCESSED,
            "modify": EventType.DATA_MODIFIED,
            "delete": EventType.DATA_DELETED,
            "export": EventType.DATA_EXPORTED
        }
        
        event_type = event_type_map.get(operation, EventType.DATA_ACCESSED)
        
        self.log_event(
            event_type=event_type,
            outcome=outcome,
            user_id=user_id,
            resource=dataset_id,
            action=operation,
            details=details
        )
    
    def get_events(self, 
                   event_type: Optional[EventType] = None,
                   user_id: Optional[str] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   limit: int = 100) -> List[AuditEvent]:
        """Get audit events with filtering.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_time: Start time filter (ISO format)
            end_time: End time filter (ISO format)
            limit: Maximum number of events to return
            
        Returns:
            List of audit events
        """
        filtered_events = self._events
        
        # Apply filters
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security event summary for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Security summary
        """
        from datetime import timedelta
        
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        recent_events = self.get_events(start_time=cutoff_time, limit=1000)
        
        # Categorize events
        security_events = [e for e in recent_events if e.risk_level in ["high", "critical"]]
        failed_logins = [e for e in recent_events if e.event_type == EventType.LOGIN_FAILED]
        access_denials = [e for e in recent_events if e.event_type == EventType.ACCESS_DENIED]
        
        return {
            "total_events": len(recent_events),
            "security_events": len(security_events),
            "failed_logins": len(failed_logins),
            "access_denials": len(access_denials),
            "high_risk_events": [e.to_dict() for e in security_events[:10]],
            "summary_period_hours": hours
        }


# Global audit logger instance
audit_logger = AuditLogger()