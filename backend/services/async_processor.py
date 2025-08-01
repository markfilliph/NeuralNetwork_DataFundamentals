"""Asynchronous processing service with Celery integration."""

import json
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

try:
    from celery import Celery, Task
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

from backend.core.config import settings
from backend.core.logging import audit_logger, EventType
from backend.services.cache_service import cache_service


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None


class AsyncProcessor:
    """Asynchronous task processor with Celery backend."""
    
    def __init__(self):
        """Initialize async processor."""
        self.celery_app = None
        self.fallback_tasks: Dict[str, Dict] = {}
        self.task_results: Dict[str, TaskResult] = {}
        
        if CELERY_AVAILABLE:
            self._setup_celery()
        else:
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="warning",
                details={
                    "message": "Celery not available, using fallback async processing",
                    "component": "async_processor"
                }
            )
    
    def _setup_celery(self):
        """Setup Celery application."""
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            
            self.celery_app = Celery(
                'dapp_async',
                broker=redis_url,
                backend=redis_url,
                include=['backend.services.async_processor']
            )
            
            # Configure Celery
            self.celery_app.conf.update(
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,
                task_track_started=True,
                task_time_limit=30 * 60,  # 30 minutes
                task_soft_time_limit=25 * 60,  # 25 minutes
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_disable_rate_limits=False,
                task_default_retry_delay=60,  # 1 minute
                task_max_retries=3,
            )
            
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="success",
                details={
                    "message": "Celery configured successfully",
                    "broker": redis_url,
                    "component": "async_processor"
                }
            )
            
        except Exception as e:
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="error",
                details={
                    "message": f"Failed to setup Celery: {e}",
                    "component": "async_processor"
                },
                risk_level="medium"
            )
            self.celery_app = None
    
    async def submit_task(self, task_name: str, *args, **kwargs) -> str:
        """Submit a task for asynchronous execution.
        
        Args:
            task_name: Name of the task to execute
            *args: Task arguments
            **kwargs: Task keyword arguments
            
        Returns:
            Task ID
        """
        task_id = f"task_{datetime.utcnow().timestamp()}_{hash(task_name)}"
        
        if self.celery_app:
            try:
                # Submit to Celery
                result = self.celery_app.send_task(
                    task_name, 
                    args=args, 
                    kwargs=kwargs,
                    task_id=task_id
                )
                
                # Cache task info
                task_info = {
                    'task_id': task_id,
                    'task_name': task_name,
                    'status': TaskStatus.PENDING.value,
                    'submitted_at': datetime.utcnow().isoformat(),
                    'args': str(args),
                    'kwargs': json.dumps(kwargs, default=str)
                }
                
                cache_service.set(f"task:{task_id}", task_info, ttl=3600)
                
                audit_logger.log_event(
                    EventType.DATA_ACCESS,
                    outcome="success",
                    details={
                        "action": "task_submitted",
                        "task_id": task_id,
                        "task_name": task_name
                    }
                )
                
                return task_id
                
            except Exception as e:
                audit_logger.log_event(
                    EventType.DATA_ACCESS,
                    outcome="error",
                    details={
                        "action": "task_submission_failed",
                        "task_name": task_name,
                        "error": str(e)
                    }
                )
                # Fall back to local processing
                return await self._submit_fallback_task(task_name, *args, **kwargs)
        else:
            # Use fallback processing
            return await self._submit_fallback_task(task_name, *args, **kwargs)
    
    async def _submit_fallback_task(self, task_name: str, *args, **kwargs) -> str:
        """Submit task for fallback processing.
        
        Args:
            task_name: Name of the task
            *args: Task arguments
            **kwargs: Task keyword arguments
            
        Returns:
            Task ID
        """
        task_id = f"fallback_task_{datetime.utcnow().timestamp()}_{hash(task_name)}"
        
        task_info = {
            'task_id': task_id,
            'task_name': task_name,
            'args': args,
            'kwargs': kwargs,
            'status': TaskStatus.PENDING.value,
            'submitted_at': datetime.utcnow(),
            'processor': 'fallback'
        }\n        \n        self.fallback_tasks[task_id] = task_info\n        \n        # Schedule task execution\n        asyncio.create_task(self._execute_fallback_task(task_id))\n        \n        return task_id\n    \n    async def _execute_fallback_task(self, task_id: str):\n        \"\"\"Execute a fallback task.\n        \n        Args:\n            task_id: Task identifier\n        \"\"\"\n        if task_id not in self.fallback_tasks:\n            return\n        \n        task_info = self.fallback_tasks[task_id]\n        task_name = task_info['task_name']\n        \n        try:\n            # Update status to started\n            task_info['status'] = TaskStatus.STARTED.value\n            task_info['started_at'] = datetime.utcnow()\n            \n            # Execute task based on task name\n            result = await self._execute_task_by_name(\n                task_name, \n                *task_info['args'], \n                **task_info['kwargs']\n            )\n            \n            # Update with success\n            task_info['status'] = TaskStatus.SUCCESS.value\n            task_info['result'] = result\n            task_info['completed_at'] = datetime.utcnow()\n            \n            # Create task result\n            task_result = TaskResult(\n                task_id=task_id,\n                status=TaskStatus.SUCCESS,\n                result=result,\n                started_at=task_info['started_at'],\n                completed_at=task_info['completed_at'],\n                duration=(task_info['completed_at'] - task_info['started_at']).total_seconds()\n            )\n            \n            self.task_results[task_id] = task_result\n            \n            audit_logger.log_event(\n                EventType.DATA_ACCESS,\n                outcome=\"success\",\n                details={\n                    \"action\": \"task_completed\",\n                    \"task_id\": task_id,\n                    \"task_name\": task_name,\n                    \"duration\": task_result.duration\n                }\n            )\n            \n        except Exception as e:\n            # Update with failure\n            task_info['status'] = TaskStatus.FAILURE.value\n            task_info['error'] = str(e)\n            task_info['completed_at'] = datetime.utcnow()\n            \n            task_result = TaskResult(\n                task_id=task_id,\n                status=TaskStatus.FAILURE,\n                error=str(e),\n                started_at=task_info.get('started_at'),\n                completed_at=task_info['completed_at']\n            )\n            \n            self.task_results[task_id] = task_result\n            \n            audit_logger.log_event(\n                EventType.DATA_ACCESS,\n                outcome=\"error\",\n                details={\n                    \"action\": \"task_failed\",\n                    \"task_id\": task_id,\n                    \"task_name\": task_name,\n                    \"error\": str(e)\n                }\n            )\n    \n    async def _execute_task_by_name(self, task_name: str, *args, **kwargs) -> Any:\n        \"\"\"Execute a task by name.\n        \n        Args:\n            task_name: Name of the task to execute\n            *args: Task arguments\n            **kwargs: Task keyword arguments\n            \n        Returns:\n            Task result\n        \"\"\"\n        # Task registry - add more tasks as needed\n        task_registry = {\n            'process_file': self._process_file_task,\n            'generate_report': self._generate_report_task,\n            'cleanup_data': self._cleanup_data_task,\n            'send_notification': self._send_notification_task,\n            'backup_database': self._backup_database_task\n        }\n        \n        if task_name in task_registry:\n            return await task_registry[task_name](*args, **kwargs)\n        else:\n            raise ValueError(f\"Unknown task: {task_name}\")\n    \n    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:\n        \"\"\"Get status of a task.\n        \n        Args:\n            task_id: Task identifier\n            \n        Returns:\n            Task result or None if not found\n        \"\"\"\n        # Check local results first\n        if task_id in self.task_results:\n            return self.task_results[task_id]\n        \n        # Check Celery if available\n        if self.celery_app and not task_id.startswith('fallback_'):\n            try:\n                result = AsyncResult(task_id, app=self.celery_app)\n                \n                status_map = {\n                    'PENDING': TaskStatus.PENDING,\n                    'STARTED': TaskStatus.STARTED,\n                    'SUCCESS': TaskStatus.SUCCESS,\n                    'FAILURE': TaskStatus.FAILURE,\n                    'RETRY': TaskStatus.RETRY,\n                    'REVOKED': TaskStatus.REVOKED\n                }\n                \n                task_result = TaskResult(\n                    task_id=task_id,\n                    status=status_map.get(result.status, TaskStatus.PENDING),\n                    result=result.result if result.successful() else None,\n                    error=str(result.result) if result.failed() else None\n                )\n                \n                return task_result\n                \n            except Exception as e:\n                audit_logger.log_event(\n                    EventType.DATA_ACCESS,\n                    outcome=\"error\",\n                    details={\n                        \"action\": \"get_task_status_failed\",\n                        \"task_id\": task_id,\n                        \"error\": str(e)\n                    }\n                )\n        \n        # Check fallback tasks\n        if task_id in self.fallback_tasks:\n            task_info = self.fallback_tasks[task_id]\n            return TaskResult(\n                task_id=task_id,\n                status=TaskStatus(task_info['status']),\n                result=task_info.get('result'),\n                error=task_info.get('error'),\n                started_at=task_info.get('started_at'),\n                completed_at=task_info.get('completed_at')\n            )\n        \n        return None\n    \n    async def cancel_task(self, task_id: str) -> bool:\n        \"\"\"Cancel a running task.\n        \n        Args:\n            task_id: Task identifier\n            \n        Returns:\n            True if cancelled successfully\n        \"\"\"\n        try:\n            if self.celery_app and not task_id.startswith('fallback_'):\n                self.celery_app.control.revoke(task_id, terminate=True)\n            \n            # Remove from fallback tasks\n            if task_id in self.fallback_tasks:\n                self.fallback_tasks[task_id]['status'] = TaskStatus.REVOKED.value\n            \n            audit_logger.log_event(\n                EventType.DATA_ACCESS,\n                outcome=\"success\",\n                details={\n                    \"action\": \"task_cancelled\",\n                    \"task_id\": task_id\n                }\n            )\n            \n            return True\n            \n        except Exception as e:\n            audit_logger.log_event(\n                EventType.DATA_ACCESS,\n                outcome=\"error\",\n                details={\n                    \"action\": \"task_cancellation_failed\",\n                    \"task_id\": task_id,\n                    \"error\": str(e)\n                }\n            )\n            return False\n    \n    async def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[TaskResult]:\n        \"\"\"List all tasks with optional status filter.\n        \n        Args:\n            status_filter: Optional status to filter by\n            \n        Returns:\n            List of task results\n        \"\"\"\n        tasks = []\n        \n        # Add local task results\n        for task_result in self.task_results.values():\n            if status_filter is None or task_result.status == status_filter:\n                tasks.append(task_result)\n        \n        # Add fallback tasks\n        for task_id, task_info in self.fallback_tasks.items():\n            if task_id not in self.task_results:  # Avoid duplicates\n                task_status = TaskStatus(task_info['status'])\n                if status_filter is None or task_status == status_filter:\n                    task_result = TaskResult(\n                        task_id=task_id,\n                        status=task_status,\n                        result=task_info.get('result'),\n                        error=task_info.get('error'),\n                        started_at=task_info.get('started_at'),\n                        completed_at=task_info.get('completed_at')\n                    )\n                    tasks.append(task_result)\n        \n        return tasks\n    \n    def get_processor_stats(self) -> Dict[str, Any]:\n        \"\"\"Get async processor statistics.\n        \n        Returns:\n            Dictionary with processor statistics\n        \"\"\"\n        total_tasks = len(self.task_results) + len(self.fallback_tasks)\n        completed_tasks = sum(1 for r in self.task_results.values() \n                            if r.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE])\n        \n        success_tasks = sum(1 for r in self.task_results.values() \n                          if r.status == TaskStatus.SUCCESS)\n        \n        failed_tasks = sum(1 for r in self.task_results.values() \n                         if r.status == TaskStatus.FAILURE)\n        \n        pending_tasks = total_tasks - completed_tasks\n        \n        return {\n            'celery_available': CELERY_AVAILABLE and self.celery_app is not None,\n            'total_tasks': total_tasks,\n            'completed_tasks': completed_tasks,\n            'success_tasks': success_tasks,\n            'failed_tasks': failed_tasks,\n            'pending_tasks': pending_tasks,\n            'success_rate': round((success_tasks / completed_tasks) * 100, 2) if completed_tasks > 0 else 0,\n            'fallback_tasks': len(self.fallback_tasks),\n            'processor_status': 'healthy' if total_tasks == 0 or (success_tasks / completed_tasks) > 0.8 else 'degraded'\n        }\n    \n    # Example task implementations\n    async def _process_file_task(self, file_path: str, **kwargs) -> Dict[str, Any]:\n        \"\"\"Process a file asynchronously.\"\"\"\n        # Simulate file processing\n        await asyncio.sleep(2)  # Simulate processing time\n        return {\n            'file_path': file_path,\n            'processed_at': datetime.utcnow().isoformat(),\n            'status': 'processed'\n        }\n    \n    async def _generate_report_task(self, report_type: str, **kwargs) -> Dict[str, Any]:\n        \"\"\"Generate a report asynchronously.\"\"\"\n        await asyncio.sleep(5)  # Simulate report generation\n        return {\n            'report_type': report_type,\n            'generated_at': datetime.utcnow().isoformat(),\n            'status': 'completed'\n        }\n    \n    async def _cleanup_data_task(self, **kwargs) -> Dict[str, Any]:\n        \"\"\"Cleanup old data asynchronously.\"\"\"\n        await asyncio.sleep(1)\n        return {\n            'cleaned_items': 42,\n            'cleaned_at': datetime.utcnow().isoformat()\n        }\n    \n    async def _send_notification_task(self, message: str, recipient: str, **kwargs) -> Dict[str, Any]:\n        \"\"\"Send notification asynchronously.\"\"\"\n        await asyncio.sleep(0.5)\n        return {\n            'message': message,\n            'recipient': recipient,\n            'sent_at': datetime.utcnow().isoformat(),\n            'status': 'sent'\n        }\n    \n    async def _backup_database_task(self, **kwargs) -> Dict[str, Any]:\n        \"\"\"Backup database asynchronously.\"\"\"\n        await asyncio.sleep(10)  # Simulate backup time\n        return {\n            'backup_file': f'backup_{datetime.utcnow().strftime(\"%Y%m%d_%H%M%S\")}.db',\n            'backed_up_at': datetime.utcnow().isoformat(),\n            'status': 'completed'\n        }\n\n\n# Global async processor instance\nasync_processor = AsyncProcessor()