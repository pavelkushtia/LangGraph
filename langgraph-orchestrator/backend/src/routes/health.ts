import express from 'express';
import { DatabaseManager } from '../services/DatabaseManager';
import { RealTimeService } from '../services/RealTimeService';
import { ClusterManager } from '../services/ClusterManager';

const router = express.Router();

// Overall system health check
router.get('/', async (req, res) => {
  try {
    const services = (req as any).services;
    const databaseManager = services.databaseManager as DatabaseManager;
    const realTimeService = services.realTimeService as RealTimeService;
    const clusterManager = services.clusterManager as ClusterManager;

    // Check database health
    const dbStats = await databaseManager.getStats();
    
    // Check real-time service health
    const realTimeHealth = realTimeService.healthCheck();
    
    // Check cluster health
    const clusterStatus = await clusterManager.getCurrentStatus();
    
    // Determine overall health
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    const issues: string[] = [];
    
    if (realTimeHealth.status !== 'healthy') {
      overallStatus = 'degraded';
      issues.push('Real-time service degraded');
    }
    
    if (clusterStatus.overall === 'unhealthy') {
      overallStatus = 'unhealthy';
      issues.push('Cluster is unhealthy');
    } else if (clusterStatus.overall === 'degraded' && overallStatus === 'healthy') {
      overallStatus = 'degraded';
      issues.push('Cluster is degraded');
    }
    
    if (dbStats.activeExecutions > 50) {
      overallStatus = overallStatus === 'healthy' ? 'degraded' : overallStatus;
      issues.push('High number of active executions');
    }

    const healthData = {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: '1.0.0',
      issues,
      components: {
        database: {
          status: 'healthy',
          stats: dbStats
        },
        realTimeService: realTimeHealth,
        cluster: {
          status: clusterStatus.overall,
          nodes: clusterStatus.nodes.length,
          services: clusterStatus.services.length,
          alerts: clusterStatus.alerts.length
        }
      },
      memory: {
        used: process.memoryUsage().heapUsed,
        total: process.memoryUsage().heapTotal,
        external: process.memoryUsage().external,
        rss: process.memoryUsage().rss
      },
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        pid: process.pid
      }
    };

    res.json({
      success: true,
      data: healthData
    });
  } catch (error) {
    console.error('Health check failed:', error);
    res.status(500).json({
      success: false,
      data: {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    });
  }
});

// Database health specifically
router.get('/database', async (req, res) => {
  try {
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    
    const stats = await databaseManager.getStats();
    
    res.json({
      success: true,
      data: {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        stats
      }
    });
  } catch (error) {
    console.error('Database health check failed:', error);
    res.status(500).json({
      success: false,
      data: {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    });
  }
});

// Real-time service health
router.get('/realtime', async (req, res) => {
  try {
    const realTimeService = (req as any).services.realTimeService as RealTimeService;
    
    const health = realTimeService.healthCheck();
    const analytics = realTimeService.getRealtimeAnalytics();
    
    res.json({
      success: true,
      data: {
        ...health,
        analytics,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Real-time service health check failed:', error);
    res.status(500).json({
      success: false,
      data: {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    });
  }
});

// Cluster health (detailed)
router.get('/cluster', async (req, res) => {
  try {
    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    
    const status = await clusterManager.getCurrentStatus();
    
    // Additional health metrics
    const healthMetrics = {
      overallHealth: status.overall,
      nodeHealth: {
        total: status.nodes.length,
        online: status.nodes.filter(n => n.status === 'online').length,
        degraded: status.nodes.filter(n => n.status === 'degraded').length,
        offline: status.nodes.filter(n => n.status === 'offline').length
      },
      serviceHealth: {
        total: status.services.length,
        running: status.services.filter(s => s.status === 'running').length,
        stopped: status.services.filter(s => s.status === 'stopped').length,
        error: status.services.filter(s => s.status === 'error').length,
        healthy: status.services.filter(s => s.health === 'healthy').length
      },
      alertSummary: {
        total: status.alerts.length,
        unresolved: status.alerts.filter(a => !a.resolved).length,
        errors: status.alerts.filter(a => a.type === 'error').length,
        warnings: status.alerts.filter(a => a.type === 'warning').length
      },
      lastUpdate: status.lastUpdate
    };
    
    res.json({
      success: true,
      data: healthMetrics
    });
  } catch (error) {
    console.error('Cluster health check failed:', error);
    res.status(500).json({
      success: false,
      data: {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    });
  }
});

// System logs
router.get('/logs', async (req, res) => {
  try {
    const { limit = 100, level, offset = 0 } = req.query;
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    
    const logs = await databaseManager.getLogs(
      level as string,
      parseInt(limit as string) || 100,
      parseInt(offset as string) || 0
    );
    
    res.json({
      success: true,
      data: {
        logs,
        count: logs.length,
        filters: {
          limit: parseInt(limit as string) || 100,
          level: level || 'all'
        }
      }
    });
  } catch (error) {
    console.error('Error getting system logs:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Log a message (for testing)
router.post('/logs', async (req, res) => {
  try {
    const { level = 'info', message, context, source = 'api' } = req.body;
    
    if (!message) {
      return res.status(400).json({
        success: false,
        error: 'Message is required'
      });
    }
    
    const validLevels = ['info', 'warn', 'error', 'debug'];
    if (!validLevels.includes(level)) {
      return res.status(400).json({
        success: false,
        error: `Invalid log level. Must be one of: ${validLevels.join(', ')}`
      });
    }
    
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    await databaseManager.logMessage(level, message, context, source);
    
    res.json({
      success: true,
      message: 'Log entry created'
    });
  } catch (error) {
    console.error('Error creating log entry:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Performance metrics
router.get('/performance', async (req, res) => {
  try {
    const startTime = Date.now();
    
    // Gather performance metrics
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    // Database performance
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    const dbQueryStart = Date.now();
    const dbStats = await databaseManager.getStats();
    const dbQueryTime = Date.now() - dbQueryStart;
    
    // Real-time service metrics
    const realTimeService = (req as any).services.realTimeService as RealTimeService;
    const realtimeMetrics = realTimeService.getRealtimeAnalytics();
    
    const performanceMetrics = {
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: {
        heapUsed: memoryUsage.heapUsed,
        heapTotal: memoryUsage.heapTotal,
        external: memoryUsage.external,
        rss: memoryUsage.rss,
        heapUsedMB: Math.round(memoryUsage.heapUsed / 1024 / 1024),
        heapTotalMB: Math.round(memoryUsage.heapTotal / 1024 / 1024)
      },
      cpu: {
        user: cpuUsage.user,
        system: cpuUsage.system
      },
      database: {
        queryTime: dbQueryTime,
        stats: dbStats
      },
      realtime: {
        connectedClients: realtimeMetrics.clientStats.totalClients,
        messageCount: realtimeMetrics.totalMessages
      },
      responseTime: Date.now() - startTime
    };
    
    res.json({
      success: true,
      data: performanceMetrics
    });
  } catch (error) {
    console.error('Error getting performance metrics:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Readiness probe (for load balancers)
router.get('/ready', async (req, res) => {
  try {
    // Quick checks for readiness
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    
    // Test database connection
    const dbStats = await databaseManager.getStats();
    
    // Simple readiness criteria
    const isReady = dbStats !== null;
    
    if (isReady) {
      res.json({
        success: true,
        ready: true,
        timestamp: new Date().toISOString()
      });
    } else {
      res.status(503).json({
        success: false,
        ready: false,
        timestamp: new Date().toISOString(),
        reason: 'Database not ready'
      });
    }
  } catch (error) {
    console.error('Readiness check failed:', error);
    res.status(503).json({
      success: false,
      ready: false,
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Liveness probe (for health monitoring)
router.get('/live', (req, res) => {
  // Simple liveness check - if we can respond, we're alive
  res.json({
    success: true,
    alive: true,
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Get system information
router.get('/system', (req, res) => {
  try {
    const systemInfo = {
      timestamp: new Date().toISOString(),
      node: {
        version: process.version,
        platform: process.platform,
        arch: process.arch,
        pid: process.pid,
        uptime: process.uptime()
      },
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      environment: {
        nodeEnv: process.env.NODE_ENV || 'development',
        port: process.env.PORT || '3001'
      }
    };
    
    res.json({
      success: true,
      data: systemInfo
    });
  } catch (error) {
    console.error('Error getting system info:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
