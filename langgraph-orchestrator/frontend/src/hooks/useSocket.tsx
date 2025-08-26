import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import toast from 'react-hot-toast';

interface SocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  emit: (event: string, data?: any) => void;
  subscribe: (topics: string[]) => void;
  unsubscribe: (topics: string[]) => void;
}

const SocketContext = createContext<SocketContextType | undefined>(undefined);

interface SocketProviderProps {
  children: React.ReactNode;
}

export function SocketProvider({ children }: SocketProviderProps) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');

  useEffect(() => {
    const socketURL = import.meta.env.VITE_WS_URL || 'ws://localhost:3001';
    
    // Create socket connection
    const newSocket = io(socketURL, {
      autoConnect: true,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      timeout: 20000,
    });

    // Connection event handlers
    newSocket.on('connect', () => {
      console.log('✅ Socket connected:', newSocket.id);
      setIsConnected(true);
      setConnectionStatus('connected');
      toast.success('Connected to LangGraph Orchestrator');
    });

    newSocket.on('disconnect', (reason) => {
      console.log('❌ Socket disconnected:', reason);
      setIsConnected(false);
      setConnectionStatus('disconnected');
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, reconnect manually
        newSocket.connect();
      }
      
      toast.error('Connection lost. Attempting to reconnect...');
    });

    newSocket.on('connect_error', (error) => {
      console.error('🔴 Socket connection error:', error);
      setConnectionStatus('error');
      toast.error('Failed to connect to server');
    });

    newSocket.on('reconnect', (attemptNumber) => {
      console.log(`🔄 Socket reconnected after ${attemptNumber} attempts`);
      setIsConnected(true);
      setConnectionStatus('connected');
      toast.success('Reconnected successfully');
    });

    newSocket.on('reconnecting', (attemptNumber) => {
      console.log(`🔄 Socket reconnecting (attempt ${attemptNumber})`);
      setConnectionStatus('connecting');
    });

    newSocket.on('reconnect_error', (error) => {
      console.error('🔴 Socket reconnection error:', error);
      setConnectionStatus('error');
    });

    newSocket.on('reconnect_failed', () => {
      console.error('🔴 Socket reconnection failed');
      setConnectionStatus('error');
      toast.error('Failed to reconnect. Please refresh the page.');
    });

    // Application-specific event handlers
    newSocket.on('clusterStatus', (data) => {
      console.log('📊 Cluster status update:', data);
      // Handle cluster status updates
    });

    newSocket.on('workflowUpdate', (data) => {
      console.log('⚡ Workflow update:', data);
      // Handle workflow updates
      if (data.type === 'execution_update') {
        if (data.status === 'completed') {
          toast.success(`Workflow execution completed`);
        } else if (data.status === 'failed') {
          toast.error(`Workflow execution failed`);
        }
      }
    });

    newSocket.on('alert', (data) => {
      console.log('🚨 System alert:', data);
      // Handle system alerts
      if (data.data.type === 'error') {
        toast.error(data.data.message);
      } else if (data.data.type === 'warning') {
        toast.error(data.data.message, { icon: '⚠️' });
      } else {
        toast(data.data.message, { icon: 'ℹ️' });
      }
    });

    newSocket.on('nodeUpdate', (data) => {
      console.log('🖥️ Node update:', data);
      // Handle node status updates
    });

    newSocket.on('serviceUpdate', (data) => {
      console.log('🔧 Service update:', data);
      // Handle service status updates
    });

    newSocket.on('systemLog', (data) => {
      console.log('📝 System log:', data);
      // Handle system logs
      if (data.data.level === 'error') {
        console.error('System error:', data.data.message);
      }
    });

    newSocket.on('serverShutdown', (data) => {
      console.log('🛑 Server shutdown:', data);
      toast.error('Server is shutting down');
      setIsConnected(false);
      setConnectionStatus('disconnected');
    });

    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      console.log('🧹 Cleaning up socket connection');
      newSocket.removeAllListeners();
      newSocket.disconnect();
    };
  }, []);

  const emit = useCallback((event: string, data?: any) => {
    if (socket && isConnected) {
      socket.emit(event, data);
    } else {
      console.warn('Cannot emit: socket not connected');
      toast.error('Cannot send request: not connected to server');
    }
  }, [socket, isConnected]);

  const subscribe = useCallback((topics: string[]) => {
    if (socket && isConnected) {
      socket.emit('subscribe', { topics });
    }
  }, [socket, isConnected]);

  const unsubscribe = useCallback((topics: string[]) => {
    if (socket && isConnected) {
      socket.emit('unsubscribe', { topics });
    }
  }, [socket, isConnected]);

  const value: SocketContextType = {
    socket,
    isConnected,
    connectionStatus,
    emit,
    subscribe,
    unsubscribe,
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
}

export function useSocket() {
  const context = useContext(SocketContext);
  if (context === undefined) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
}

// Custom hooks for specific socket functionality
export function useClusterStatus() {
  const { socket, isConnected, subscribe } = useSocket();
  const [clusterStatus, setClusterStatus] = useState(null);

  useEffect(() => {
    if (isConnected) {
      subscribe(['cluster']);
      
      // Request current status
      socket?.emit('requestClusterStatus');

      // Listen for status updates
      const handleStatusUpdate = (data: any) => {
        setClusterStatus(data);
      };

      socket?.on('clusterStatus', handleStatusUpdate);

      return () => {
        socket?.off('clusterStatus', handleStatusUpdate);
      };
    }
  }, [isConnected, socket, subscribe]);

  return clusterStatus;
}

export function useWorkflowUpdates(workflowId?: string) {
  const { socket, isConnected, subscribe } = useSocket();
  const [updates, setUpdates] = useState<any[]>([]);

  useEffect(() => {
    if (isConnected) {
      const topics = ['workflows'];
      if (workflowId) {
        topics.push(`workflow:${workflowId}`);
      }
      subscribe(topics);

      // Listen for workflow updates
      const handleWorkflowUpdate = (data: any) => {
        if (!workflowId || data.workflowId === workflowId) {
          setUpdates(prev => [data, ...prev.slice(0, 99)]); // Keep last 100 updates
        }
      };

      socket?.on('workflowUpdate', handleWorkflowUpdate);

      return () => {
        socket?.off('workflowUpdate', handleWorkflowUpdate);
      };
    }
  }, [isConnected, socket, subscribe, workflowId]);

  return updates;
}

export function useExecutionUpdates(executionId?: string) {
  const { socket, isConnected, subscribe } = useSocket();
  const [execution, setExecution] = useState(null);

  useEffect(() => {
    if (isConnected && executionId) {
      subscribe([`execution:${executionId}`]);

      // Listen for execution updates
      const handleExecutionUpdate = (data: any) => {
        if (data.executionId === executionId) {
          setExecution(data);
        }
      };

      socket?.on('workflowUpdate', handleExecutionUpdate);

      return () => {
        socket?.off('workflowUpdate', handleExecutionUpdate);
      };
    }
  }, [isConnected, socket, subscribe, executionId]);

  return execution;
}
