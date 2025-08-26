import { Server as SocketIOServer, Socket } from 'socket.io';
import { ClusterStatus } from './ClusterManager';

export interface RealTimeMessage {
  type: string;
  data: any;
  timestamp: Date;
  source?: string;
}

export interface WorkflowUpdateMessage {
  type: 'execution_update' | 'workflow_created' | 'workflow_updated' | 'workflow_deleted';
  executionId?: string;
  workflowId?: string;
  status?: string;
  progress?: number;
  currentNode?: string;
  error?: string;
  data?: any;
}

export interface ClusterUpdateMessage {
  type: 'status_update' | 'node_update' | 'service_update' | 'alert';
  nodeId?: string;
  serviceId?: string;
  status?: string;
  data?: any;
}

export class RealTimeService {
  private io: SocketIOServer;
  private connectedClients: Map<string, Socket> = new Map();
  private messageHistory: RealTimeMessage[] = [];
  private readonly MAX_HISTORY_SIZE = 1000;

  constructor(io: SocketIOServer) {
    this.io = io;
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.io.on('connection', (socket: Socket) => {
      console.log(`🔗 Client connected: ${socket.id}`);
      this.connectedClients.set(socket.id, socket);

      // Send recent message history to new clients
      this.sendMessageHistory(socket);

      // Handle client-specific subscriptions
      socket.on('subscribe', (data) => {
        this.handleSubscription(socket, data);
      });

      socket.on('unsubscribe', (data) => {
        this.handleUnsubscription(socket, data);
      });

      // Handle client requests for current status
      socket.on('requestClusterStatus', () => {
        this.handleClusterStatusRequest(socket);
      });

      socket.on('requestWorkflowStatus', (data) => {
        this.handleWorkflowStatusRequest(socket, data);
      });

      // Handle disconnection
      socket.on('disconnect', () => {
        console.log(`🔌 Client disconnected: ${socket.id}`);
        this.connectedClients.delete(socket.id);
      });

      // Handle client errors
      socket.on('error', (error) => {
        console.error(`Socket error for client ${socket.id}:`, error);
      });
    });
  }

  private sendMessageHistory(socket: Socket): void {
    // Send last 50 messages to new clients
    const recentMessages = this.messageHistory.slice(-50);
    socket.emit('messageHistory', recentMessages);
  }

  private handleSubscription(socket: Socket, data: { topics: string[] }): void {
    if (data.topics) {
      data.topics.forEach(topic => {
        socket.join(topic);
        console.log(`📢 Client ${socket.id} subscribed to ${topic}`);
      });
      
      socket.emit('subscriptionConfirmed', { topics: data.topics });
    }
  }

  private handleUnsubscription(socket: Socket, data: { topics: string[] }): void {
    if (data.topics) {
      data.topics.forEach(topic => {
        socket.leave(topic);
        console.log(`📢 Client ${socket.id} unsubscribed from ${topic}`);
      });
      
      socket.emit('unsubscriptionConfirmed', { topics: data.topics });
    }
  }

  private handleClusterStatusRequest(socket: Socket): void {
    // This would typically fetch current cluster status
    // For now, emit a placeholder response
    socket.emit('clusterStatusResponse', {
      requested: true,
      timestamp: new Date().toISOString()
    });
  }

  private handleWorkflowStatusRequest(socket: Socket, data: { workflowId?: string, executionId?: string }): void {
    // This would typically fetch current workflow/execution status
    socket.emit('workflowStatusResponse', {
      requested: true,
      workflowId: data.workflowId,
      executionId: data.executionId,
      timestamp: new Date().toISOString()
    });
  }

  // Public methods for broadcasting updates
  
  broadcastClusterStatus(status: ClusterStatus): void {
    const message: RealTimeMessage = {
      type: 'cluster_status_update',
      data: status,
      timestamp: new Date(),
      source: 'cluster_manager'
    };

    this.addToHistory(message);
    this.io.to('cluster').emit('clusterStatus', message);
    this.io.emit('clusterStatus', message); // Also broadcast to all clients
    
    console.log(`📡 Broadcasted cluster status: ${status.overall}`);
  }

  broadcastWorkflowUpdate(update: WorkflowUpdateMessage): void {
    const message: RealTimeMessage = {
      type: 'workflow_update',
      data: update,
      timestamp: new Date(),
      source: 'workflow_engine'
    };

    this.addToHistory(message);

    // Broadcast to workflow subscribers
    this.io.to('workflows').emit('workflowUpdate', message);
    
    // Broadcast to specific workflow subscribers
    if (update.workflowId) {
      this.io.to(`workflow:${update.workflowId}`).emit('workflowUpdate', message);
    }

    // Broadcast to specific execution subscribers
    if (update.executionId) {
      this.io.to(`execution:${update.executionId}`).emit('workflowUpdate', message);
    }

    console.log(`📡 Broadcasted workflow update: ${update.type}`);
  }

  broadcastNodeUpdate(nodeId: string, status: string, data?: any): void {
    const message: RealTimeMessage = {
      type: 'node_update',
      data: {
        nodeId,
        status,
        data,
        timestamp: new Date()
      },
      timestamp: new Date(),
      source: 'cluster_manager'
    };

    this.addToHistory(message);
    this.io.to('cluster').emit('nodeUpdate', message);
    this.io.to(`node:${nodeId}`).emit('nodeUpdate', message);
    
    console.log(`📡 Broadcasted node update: ${nodeId} - ${status}`);
  }

  broadcastServiceUpdate(serviceId: string, status: string, data?: any): void {
    const message: RealTimeMessage = {
      type: 'service_update',
      data: {
        serviceId,
        status,
        data,
        timestamp: new Date()
      },
      timestamp: new Date(),
      source: 'cluster_manager'
    };

    this.addToHistory(message);
    this.io.to('cluster').emit('serviceUpdate', message);
    this.io.to(`service:${serviceId}`).emit('serviceUpdate', message);
    
    console.log(`📡 Broadcasted service update: ${serviceId} - ${status}`);
  }

  broadcastAlert(alert: { type: 'error' | 'warning' | 'info', message: string, source?: string, data?: any }): void {
    const message: RealTimeMessage = {
      type: 'alert',
      data: {
        ...alert,
        id: `alert-${Date.now()}`,
        timestamp: new Date()
      },
      timestamp: new Date(),
      source: alert.source || 'system'
    };

    this.addToHistory(message);
    this.io.emit('alert', message);
    
    console.log(`🚨 Broadcasted alert: ${alert.type} - ${alert.message}`);
  }

  broadcastSystemLog(level: 'info' | 'warn' | 'error' | 'debug', message: string, context?: any): void {
    const logMessage: RealTimeMessage = {
      type: 'system_log',
      data: {
        level,
        message,
        context,
        timestamp: new Date()
      },
      timestamp: new Date(),
      source: 'system'
    };

    this.addToHistory(logMessage);
    this.io.to('logs').emit('systemLog', logMessage);
    
    // Also emit errors to all clients for immediate attention
    if (level === 'error') {
      this.io.emit('systemError', logMessage);
    }
  }

  // Send a message to specific client
  sendToClient(socketId: string, event: string, data: any): void {
    const socket = this.connectedClients.get(socketId);
    if (socket) {
      socket.emit(event, data);
      console.log(`📤 Sent ${event} to client ${socketId}`);
    } else {
      console.warn(`⚠️ Client ${socketId} not found`);
    }
  }

  // Send a message to all clients in a room
  sendToRoom(room: string, event: string, data: any): void {
    this.io.to(room).emit(event, data);
    console.log(`📤 Sent ${event} to room ${room}`);
  }

  // Get connected client statistics
  getClientStats(): { totalClients: number, rooms: Record<string, number> } {
    const rooms: Record<string, number> = {};
    
    // Get room information (this is a simplified version)
    this.io.sockets.adapter.rooms.forEach((sockets, roomName) => {
      if (!roomName.startsWith('/')) { // Filter out default rooms
        rooms[roomName] = sockets.size;
      }
    });

    return {
      totalClients: this.connectedClients.size,
      rooms
    };
  }

  // Utility methods

  private addToHistory(message: RealTimeMessage): void {
    this.messageHistory.push(message);
    
    // Keep history size manageable
    if (this.messageHistory.length > this.MAX_HISTORY_SIZE) {
      this.messageHistory = this.messageHistory.slice(-this.MAX_HISTORY_SIZE + 100);
    }
  }

  getMessageHistory(limit: number = 100): RealTimeMessage[] {
    return this.messageHistory.slice(-limit);
  }

  clearMessageHistory(): void {
    this.messageHistory = [];
    console.log('🧹 Cleared message history');
  }

  // Analytics and monitoring

  getRealtimeAnalytics(): any {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    
    const recentMessages = this.messageHistory.filter(msg => msg.timestamp > oneHourAgo);
    
    const messageTypes = recentMessages.reduce((acc, msg) => {
      acc[msg.type] = (acc[msg.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const messageSources = recentMessages.reduce((acc, msg) => {
      const source = msg.source || 'unknown';
      acc[source] = (acc[source] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      period: {
        start: oneHourAgo.toISOString(),
        end: now.toISOString()
      },
      totalMessages: recentMessages.length,
      messageTypes,
      messageSources,
      clientStats: this.getClientStats()
    };
  }

  // Health check for real-time service
  healthCheck(): { status: 'healthy' | 'degraded' | 'unhealthy', details: any } {
    const stats = this.getClientStats();
    const messageCount = this.messageHistory.length;
    
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    
    // Simple health criteria
    if (stats.totalClients === 0) {
      status = 'degraded'; // No clients connected
    }
    
    if (messageCount === 0) {
      status = 'degraded'; // No message activity
    }

    return {
      status,
      details: {
        connectedClients: stats.totalClients,
        activeRooms: Object.keys(stats.rooms).length,
        messageHistorySize: messageCount,
        lastMessageTime: this.messageHistory.length > 0 
          ? this.messageHistory[this.messageHistory.length - 1].timestamp 
          : null
      }
    };
  }

  // Shutdown gracefully
  shutdown(): void {
    console.log('🛑 Shutting down real-time service...');
    
    // Notify all clients about shutdown
    this.io.emit('serverShutdown', { 
      message: 'Server is shutting down',
      timestamp: new Date().toISOString()
    });

    // Close all connections
    this.io.close();
    this.connectedClients.clear();
    this.messageHistory = [];
    
    console.log('✅ Real-time service shut down');
  }
}
