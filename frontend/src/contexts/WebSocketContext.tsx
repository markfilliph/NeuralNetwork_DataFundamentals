/**
 * WebSocket context for real-time updates
 * Manages connection to FastAPI WebSocket endpoint
 */

'use client';

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuth } from './AuthContext';

interface WebSocketContextType {
  socket: Socket | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastActivity: string | null;
  subscribe: (event: string, callback: (data: any) => void) => void;
  unsubscribe: (event: string) => void;
  emit: (event: string, data: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastActivity, setLastActivity] = useState<string | null>(null);
  const { user, token } = useAuth();

  useEffect(() => {
    if (user && token) {
      // Create WebSocket connection
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
      const newSocket = io(wsUrl, {
        auth: {
          token: token,
        },
        transports: ['websocket'],
        upgrade: false,
      });

      // Connection event handlers
      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        setLastActivity(new Date().toISOString());
      });

      newSocket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setConnectionStatus('disconnected');
      });

      newSocket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setConnectionStatus('error');
      });

      // Real-time event handlers
      newSocket.on('dataset_processed', (data) => {
        console.log('Dataset processed:', data);
        setLastActivity(new Date().toISOString());
        // Trigger notifications or UI updates
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('Dataset Ready', {
            body: `Dataset "${data.filename}" has been processed successfully`,
            icon: '/favicon.ico',
          });
        }
      });

      newSocket.on('model_training_complete', (data) => {
        console.log('Model training complete:', data);
        setLastActivity(new Date().toISOString());
        if ('Notification' in window && Notification.permission === 'granted') {
          new Notification('Model Ready', {
            body: `Model training completed with R² score: ${data.metrics?.r2_score?.toFixed(3) || 'N/A'}`,
            icon: '/favicon.ico',
          });
        }
      });

      newSocket.on('system_alert', (data) => {
        console.log('System alert:', data);
        setLastActivity(new Date().toISOString());
        // Handle system alerts (high memory usage, errors, etc.)
      });

      // User activity tracking
      newSocket.on('user_activity', (data) => {
        setLastActivity(data.timestamp);
      });

      setSocket(newSocket);
      setConnectionStatus('connecting');

      // Cleanup on unmount
      return () => {
        newSocket.close();
        setSocket(null);
        setConnectionStatus('disconnected');
      };
    } else {
      // Clean up connection when user logs out
      if (socket) {
        socket.close();
        setSocket(null);
        setConnectionStatus('disconnected');
      }
    }
  }, [user, token]);

  // Request notification permission
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const subscribe = (event: string, callback: (data: any) => void) => {
    if (socket) {
      socket.on(event, callback);
    }
  };

  const unsubscribe = (event: string) => {
    if (socket) {
      socket.off(event);
    }
  };

  const emit = (event: string, data: any) => {
    if (socket && connectionStatus === 'connected') {
      socket.emit(event, data);
    }
  };

  const value: WebSocketContextType = {
    socket,
    connectionStatus,
    lastActivity,
    subscribe,
    unsubscribe,
    emit,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket(): WebSocketContextType {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}

// Custom hooks for specific WebSocket events
export function useDatasetUpdates() {
  const { subscribe, unsubscribe } = useWebSocket();
  
  useEffect(() => {
    const handleDatasetUpdate = (data: any) => {
      // Trigger React Query cache invalidation
      // This would be handled by the specific components
      console.log('Dataset update:', data);
    };

    subscribe('dataset_processed', handleDatasetUpdate);
    subscribe('dataset_error', handleDatasetUpdate);

    return () => {
      unsubscribe('dataset_processed');
      unsubscribe('dataset_error');
    };
  }, [subscribe, unsubscribe]);
}

export function useModelUpdates() {
  const { subscribe, unsubscribe } = useWebSocket();
  
  useEffect(() => {
    const handleModelUpdate = (data: any) => {
      console.log('Model update:', data);
    };

    subscribe('model_training_complete', handleModelUpdate);
    subscribe('model_training_failed', handleModelUpdate);
    subscribe('model_training_progress', handleModelUpdate);

    return () => {
      unsubscribe('model_training_complete');
      unsubscribe('model_training_failed');
      unsubscribe('model_training_progress');
    };
  }, [subscribe, unsubscribe]);
}

export function useSystemUpdates() {
  const { subscribe, unsubscribe } = useWebSocket();
  
  useEffect(() => {
    const handleSystemUpdate = (data: any) => {
      console.log('System update:', data);
    };

    subscribe('system_alert', handleSystemUpdate);
    subscribe('system_maintenance', handleSystemUpdate);

    return () => {
      unsubscribe('system_alert');
      unsubscribe('system_maintenance');
    };
  }, [subscribe, unsubscribe]);
}