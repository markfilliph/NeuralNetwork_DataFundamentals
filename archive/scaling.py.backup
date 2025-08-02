"""Horizontal scaling architecture components."""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from backend.core.config import settings
from backend.core.logging import audit_logger, EventType
from backend.services.cache_service import cache_service


@dataclass
class ServiceInstance:
    """Represents a service instance in the cluster."""
    instance_id: str
    host: str
    port: int
    status: str
    last_heartbeat: datetime
    load_score: float
    metadata: Dict[str, Any]


class LoadBalancer:
    """Simple load balancer for horizontal scaling."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.instances: Dict[str, ServiceInstance] = {}
        self.algorithms = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted_round_robin': self._weighted_round_robin
        }
        self.current_algorithm = 'round_robin'
        self.round_robin_index = 0
    
    def register_instance(self, instance: ServiceInstance) -> bool:
        """Register a service instance.
        
        Args:
            instance: Service instance to register
            
        Returns:
            True if registered successfully
        """
        try:
            self.instances[instance.instance_id] = instance
            
            # Cache instance info
            cache_service.set(
                f"instance:{instance.instance_id}",
                {
                    'host': instance.host,
                    'port': instance.port,
                    'status': instance.status,
                    'last_heartbeat': instance.last_heartbeat.isoformat(),
                    'load_score': instance.load_score
                },
                ttl=300  # 5 minutes
            )
            
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="success",
                details={
                    "action": "instance_registered",
                    "instance_id": instance.instance_id,
                    "host": instance.host,
                    "port": instance.port
                }
            )
            
            return True
            
        except Exception as e:
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="error",
                details={
                    "action": "instance_registration_failed",
                    "error": str(e)
                }
            )
            return False
    
    def deregister_instance(self, instance_id: str) -> bool:
        """Deregister a service instance.
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            True if deregistered successfully
        """
        try:
            if instance_id in self.instances:
                del self.instances[instance_id]
                cache_service.delete(f"instance:{instance_id}")
                
                audit_logger.log_event(
                    EventType.SYSTEM_START,
                    outcome="success",
                    details={
                        "action": "instance_deregistered",
                        "instance_id": instance_id
                    }
                )
                
                return True
            return False
            
        except Exception as e:
            audit_logger.log_event(
                EventType.SYSTEM_START,
                outcome="error",
                details={
                    "action": "instance_deregistration_failed",
                    "error": str(e)
                }
            )
            return False
    
    def get_next_instance(self) -> Optional[ServiceInstance]:
        """Get next available instance using configured algorithm.
        
        Returns:
            Selected service instance or None if none available
        """
        healthy_instances = self._get_healthy_instances()
        
        if not healthy_instances:
            return None
        
        algorithm = self.algorithms.get(self.current_algorithm, self._round_robin)
        return algorithm(healthy_instances)
    
    def _get_healthy_instances(self) -> List[ServiceInstance]:
        """Get list of healthy instances.
        
        Returns:
            List of healthy service instances
        """
        healthy = []
        current_time = datetime.utcnow()
        
        for instance in self.instances.values():
            # Consider instance healthy if heartbeat is within last 30 seconds
            if (current_time - instance.last_heartbeat).total_seconds() < 30:
                if instance.status == 'healthy':
                    healthy.append(instance)
        
        return healthy
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin load balancing algorithm."""
        if not instances:
            return None
        
        instance = instances[self.round_robin_index % len(instances)]
        self.round_robin_index = (self.round_robin_index + 1) % len(instances)
        return instance
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections load balancing algorithm."""
        if not instances:
            return None
        
        # For simplicity, use load_score as connection count proxy
        return min(instances, key=lambda x: x.load_score)
    
    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin based on instance capacity."""
        if not instances:
            return None
        
        # Weight by inverse of load_score (lower load = higher weight)
        total_weight = sum(1.0 / max(instance.load_score, 0.1) for instance in instances)
        target = (self.round_robin_index % 100) / 100.0 * total_weight
        
        current_weight = 0
        for instance in instances:
            current_weight += 1.0 / max(instance.load_score, 0.1)
            if current_weight >= target:
                self.round_robin_index += 1
                return instance
        
        return instances[0]  # Fallback
    
    def update_instance_health(self, instance_id: str, load_score: float, 
                              status: str = 'healthy') -> bool:
        """Update instance health information.
        
        Args:
            instance_id: Instance identifier
            load_score: Current load score
            status: Instance status
            
        Returns:
            True if updated successfully
        """
        if instance_id in self.instances:
            self.instances[instance_id].load_score = load_score
            self.instances[instance_id].status = status
            self.instances[instance_id].last_heartbeat = datetime.utcnow()
            
            # Update cache
            cache_service.set(
                f"instance:{instance_id}",
                {
                    'host': self.instances[instance_id].host,
                    'port': self.instances[instance_id].port,
                    'status': status,
                    'last_heartbeat': self.instances[instance_id].last_heartbeat.isoformat(),
                    'load_score': load_score
                },
                ttl=300
            )
            
            return True
        return False
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics.
        
        Returns:
            Dictionary with cluster statistics
        """
        healthy_instances = self._get_healthy_instances()
        total_instances = len(self.instances)
        
        if healthy_instances:
            avg_load = sum(instance.load_score for instance in healthy_instances) / len(healthy_instances)
            max_load = max(instance.load_score for instance in healthy_instances)
            min_load = min(instance.load_score for instance in healthy_instances)
        else:
            avg_load = max_load = min_load = 0
        
        return {
            'total_instances': total_instances,
            'healthy_instances': len(healthy_instances),
            'unhealthy_instances': total_instances - len(healthy_instances),
            'load_balancing_algorithm': self.current_algorithm,
            'average_load': round(avg_load, 2),
            'max_load': round(max_load, 2),
            'min_load': round(min_load, 2),
            'cluster_health': 'healthy' if len(healthy_instances) > 0 else 'unhealthy'
        }


class SessionAffinity:
    """Session affinity management for sticky sessions."""
    
    def __init__(self, load_balancer: LoadBalancer):
        """Initialize session affinity manager.
        
        Args:
            load_balancer: Load balancer instance
        """
        self.load_balancer = load_balancer
        self.session_mappings: Dict[str, str] = {}  # session_id -> instance_id
    
    def get_instance_for_session(self, session_id: str) -> Optional[ServiceInstance]:
        """Get instance for a specific session (sticky session).
        
        Args:
            session_id: Session identifier
            
        Returns:
            Service instance for the session
        """
        # Check if session is already mapped to an instance
        if session_id in self.session_mappings:
            instance_id = self.session_mappings[session_id]
            if instance_id in self.load_balancer.instances:
                instance = self.load_balancer.instances[instance_id]
                # Verify instance is still healthy
                if (datetime.utcnow() - instance.last_heartbeat).total_seconds() < 30:
                    return instance
                else:
                    # Remove stale mapping
                    del self.session_mappings[session_id]
        
        # Get new instance and create mapping
        instance = self.load_balancer.get_next_instance()
        if instance:
            self.session_mappings[session_id] = instance.instance_id
            
            # Cache session mapping
            cache_service.set(
                f"session_affinity:{session_id}",
                instance.instance_id,
                ttl=3600  # 1 hour
            )
        
        return instance
    
    def remove_session_mapping(self, session_id: str) -> bool:
        """Remove session mapping.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if mapping removed
        """
        if session_id in self.session_mappings:
            del self.session_mappings[session_id]
            cache_service.delete(f"session_affinity:{session_id}")
            return True
        return False


class HealthChecker:
    """Health checker for service instances."""
    
    def __init__(self, load_balancer: LoadBalancer):
        """Initialize health checker.
        
        Args:
            load_balancer: Load balancer instance
        """
        self.load_balancer = load_balancer
    
    def check_instance_health(self, instance_id: str) -> Dict[str, Any]:
        """Check health of a specific instance.
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Health check results
        """
        if instance_id not in self.load_balancer.instances:
            return {'status': 'not_found', 'healthy': False}
        
        instance = self.load_balancer.instances[instance_id]
        current_time = datetime.utcnow()
        last_heartbeat_age = (current_time - instance.last_heartbeat).total_seconds()
        
        # Instance is healthy if heartbeat is recent and status is good
        is_healthy = (
            last_heartbeat_age < 30 and 
            instance.status == 'healthy' and
            instance.load_score < 100  # Arbitrary threshold
        )
        
        return {
            'instance_id': instance_id,
            'status': instance.status,
            'healthy': is_healthy,
            'last_heartbeat_age': last_heartbeat_age,
            'load_score': instance.load_score,
            'host': instance.host,
            'port': instance.port
        }
    
    def check_cluster_health(self) -> Dict[str, Any]:
        """Check health of entire cluster.
        
        Returns:
            Cluster health status
        """
        health_results = {}
        healthy_count = 0
        total_count = len(self.load_balancer.instances)
        
        for instance_id in self.load_balancer.instances:
            result = self.check_instance_health(instance_id)
            health_results[instance_id] = result
            if result['healthy']:
                healthy_count += 1
        
        cluster_healthy = healthy_count > 0 and (healthy_count / total_count) >= 0.5
        
        return {
            'cluster_healthy': cluster_healthy,
            'healthy_instances': healthy_count,
            'total_instances': total_count,
            'health_percentage': round((healthy_count / total_count) * 100, 2) if total_count > 0 else 0,
            'instances': health_results,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instances
load_balancer = LoadBalancer()
session_affinity = SessionAffinity(load_balancer)
health_checker = HealthChecker(load_balancer)