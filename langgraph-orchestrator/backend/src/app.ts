import express from 'express';
import http from 'http';
import { Server as SocketIOServer } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';

import { ClusterManager } from './services/ClusterManager';
import { WorkflowEngine } from './services/WorkflowEngine';
import { DatabaseManager } from './services/DatabaseManager';
import { RealTimeService } from './services/RealTimeService';

import clusterRoutes from './routes/cluster';
import workflowRoutes from './routes/workflows';
import healthRoutes from './routes/health';
import analyticsRoutes from './routes/analytics';

class LangGraphOrchestratorApp {
  private app: express.Application;
  private server: http.Server;
  private io: SocketIOServer;
  private clusterManager!: ClusterManager;
  private workflowEngine!: WorkflowEngine;
  private databaseManager!: DatabaseManager;
  private realTimeService!: RealTimeService;

  constructor() {
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = new SocketIOServer(this.server, {
      cors: {
        origin: [
          "http://localhost:3000", 
          "http://localhost:5173",
          "http://192.168.1.81:3000",
          "http://192.168.1.177:3000",
          "http://192.168.1.178:3000", 
          "http://192.168.1.105:3000",
          "http://192.168.1.137:3000",
          // Allow any origin in the local network for development
          /^http:\/\/192\.168\.1\.\d+:3000$/
        ],
        methods: ["GET", "POST"],
        credentials: true
      }
    });

    this.initializeServices();
    this.initializeMiddleware();
    this.initializeRoutes();
    this.initializeWebSocket();
  }

  private async initializeServices(): Promise<void> {
    // Initialize database
    this.databaseManager = new DatabaseManager();
    await this.databaseManager.initialize();

    // Initialize services
    this.clusterManager = new ClusterManager();
    this.workflowEngine = new WorkflowEngine(this.databaseManager);
    this.realTimeService = new RealTimeService(this.io);

    // Start cluster monitoring
    await this.clusterManager.startMonitoring();
    
    // Connect services
    this.clusterManager.on('statusUpdate', (status) => {
      this.realTimeService.broadcastClusterStatus(status);
    });

    this.workflowEngine.on('workflowUpdate', (update) => {
      this.realTimeService.broadcastWorkflowUpdate(update);
    });
  }

  private initializeMiddleware(): void {
    // Security and performance middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'", "ws:", "wss:"]
        }
      }
    }));
    
    this.app.use(compression());
    this.app.use(cors({
      origin: [
        "http://localhost:3000", 
        "http://localhost:5173",
        "http://192.168.1.81:3000",
        "http://192.168.1.177:3000",
        "http://192.168.1.178:3000", 
        "http://192.168.1.105:3000",
        "http://192.168.1.137:3000",
        // Allow any origin in the local network for development
        /^http:\/\/192\.168\.1\.\d+:3000$/
      ],
      credentials: true
    }));
    
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));
    
    this.app.use(morgan('combined'));

    // Add service instances to request
    this.app.use((req, _res, next) => {
      (req as any).services = {
        clusterManager: this.clusterManager,
        workflowEngine: this.workflowEngine,
        databaseManager: this.databaseManager,
        realTimeService: this.realTimeService
      };
      next();
    });
  }

  private initializeRoutes(): void {
    // API routes
    this.app.use('/api/cluster', clusterRoutes);
    this.app.use('/api/workflows', workflowRoutes);
    this.app.use('/api/health', healthRoutes);
    this.app.use('/api/analytics', analyticsRoutes);

    // Health check endpoint
    this.app.get('/api/ping', (_req, res) => {
      res.json({ 
        status: 'ok', 
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });
    });

    // Serve static files in production
    if (process.env.NODE_ENV === 'production') {
      this.app.use(express.static('public'));
      this.app.get('*', (_req, res) => {
        res.sendFile('index.html', { root: 'public' });
      });
    }

    // Error handling middleware
    this.app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
      console.error('Error:', err);
      res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
      });
    });

    // 404 handler
    this.app.use((_req, res) => {
      res.status(404).json({ error: 'Not found' });
    });
  }

  private initializeWebSocket(): void {
    this.io.on('connection', (socket) => {
      console.log(`Client connected: ${socket.id}`);

      // Send current cluster status to new clients
      this.clusterManager.getCurrentStatus().then(status => {
        socket.emit('clusterStatus', status);
      });

      // Handle client disconnection
      socket.on('disconnect', () => {
        console.log(`Client disconnected: ${socket.id}`);
      });

      // Handle workflow execution requests
      socket.on('executeWorkflow', async (data) => {
        try {
          const result = await this.workflowEngine.executeWorkflow(data.workflowId, data.inputs);
          socket.emit('workflowExecutionResult', { success: true, result });
        } catch (error) {
          socket.emit('workflowExecutionResult', { 
            success: false, 
            error: error instanceof Error ? error.message : 'Unknown error' 
          });
        }
      });

      // Handle cluster management requests
      socket.on('clusterAction', async (data) => {
        try {
          const result = await this.clusterManager.executeAction(data.action, data.parameters);
          socket.emit('clusterActionResult', { success: true, result });
        } catch (error) {
          socket.emit('clusterActionResult', { 
            success: false, 
            error: error instanceof Error ? error.message : 'Unknown error' 
          });
        }
      });
    });
  }

  public start(port: number = 3001): void {
    const host = process.env.HOST || '0.0.0.0';
    this.server.listen(port, host, () => {
      console.log(`🚀 LangGraph Orchestrator API running on ${host}:${port}`);
      console.log(`📊 WebSocket server ready for real-time updates`);
      console.log(`🔗 Frontend accessible from any cluster node at http://<node-ip>:3000`);
      console.log(`🌐 API accessible at http://<node-ip>:${port}`);
    });
  }

  public async stop(): Promise<void> {
    console.log('🛑 Shutting down LangGraph Orchestrator...');
    
    // Stop monitoring
    await this.clusterManager.stopMonitoring();
    
    // Close database connections
    await this.databaseManager.close();
    
    // Close server
    this.server.close();
    this.io.close();
    
    console.log('✅ LangGraph Orchestrator stopped gracefully');
  }
}

// Initialize and start the application
const app = new LangGraphOrchestratorApp();

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  await app.stop();
  process.exit(0);
});

process.on('SIGINT', async () => {
  await app.stop();
  process.exit(0);
});

// Start the server
const PORT = parseInt(process.env.PORT || '3001', 10);
app.start(PORT);

export default app;
