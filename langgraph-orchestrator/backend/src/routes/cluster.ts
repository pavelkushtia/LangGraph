import express from 'express';
import { ClusterManager } from '../services/ClusterManager';

const router = express.Router();

// Get current cluster status
router.get('/status', async (req, res) => {
  try {
    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    const status = await clusterManager.getCurrentStatus();
    res.json({ success: true, data: status });
  } catch (error) {
    console.error('Error getting cluster status:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Execute cluster actions (start, stop, restart, test)
router.post('/action', async (req, res) => {
  try {
    const { action, parameters } = req.body;
    
    if (!action) {
      return res.status(400).json({ 
        success: false, 
        error: 'Action is required' 
      });
    }

    const validActions = ['start', 'stop', 'restart', 'test', 'status'];
    if (!validActions.includes(action)) {
      return res.status(400).json({ 
        success: false, 
        error: `Invalid action. Must be one of: ${validActions.join(', ')}` 
      });
    }

    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    const result = await clusterManager.executeAction(action, parameters);
    
    res.json({ 
      success: true, 
      data: result,
      message: `Cluster action '${action}' executed successfully`
    });
  } catch (error) {
    console.error('Error executing cluster action:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get detailed node information
router.get('/nodes', async (req, res) => {
  try {
    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    const status = await clusterManager.getCurrentStatus();
    
    res.json({ 
      success: true, 
      data: {
        nodes: status.nodes,
        lastUpdate: status.lastUpdate
      }
    });
  } catch (error) {
    console.error('Error getting node information:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get detailed service information
router.get('/services', async (req, res) => {
  try {
    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    const status = await clusterManager.getCurrentStatus();
    
    res.json({ 
      success: true, 
      data: {
        services: status.services,
        lastUpdate: status.lastUpdate
      }
    });
  } catch (error) {
    console.error('Error getting service information:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get cluster alerts
router.get('/alerts', async (req, res) => {
  try {
    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    const status = await clusterManager.getCurrentStatus();
    
    res.json({ 
      success: true, 
      data: {
        alerts: status.alerts,
        alertCount: status.alerts.length,
        unresolvedAlerts: status.alerts.filter(a => !a.resolved).length
      }
    });
  } catch (error) {
    console.error('Error getting cluster alerts:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get cluster health summary
router.get('/health', async (req, res) => {
  try {
    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    const status = await clusterManager.getCurrentStatus();
    
    const healthSummary = {
      overall: status.overall,
      nodesSummary: {
        total: status.nodes.length,
        online: status.nodes.filter(n => n.status === 'online').length,
        degraded: status.nodes.filter(n => n.status === 'degraded').length,
        offline: status.nodes.filter(n => n.status === 'offline').length
      },
      servicesSummary: {
        total: status.services.length,
        running: status.services.filter(s => s.status === 'running').length,
        stopped: status.services.filter(s => s.status === 'stopped').length,
        error: status.services.filter(s => s.status === 'error').length
      },
      alertsSummary: {
        total: status.alerts.length,
        unresolved: status.alerts.filter(a => !a.resolved).length,
        errors: status.alerts.filter(a => a.type === 'error').length,
        warnings: status.alerts.filter(a => a.type === 'warning').length
      },
      lastUpdate: status.lastUpdate
    };
    
    res.json({ 
      success: true, 
      data: healthSummary
    });
  } catch (error) {
    console.error('Error getting cluster health:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Test specific service endpoint
router.post('/test-service', async (req, res) => {
  try {
    const { serviceId, endpoint } = req.body;
    
    if (!serviceId || !endpoint) {
      return res.status(400).json({ 
        success: false, 
        error: 'Service ID and endpoint are required' 
      });
    }

    // This would implement specific service testing
    // For now, return a placeholder response
    res.json({ 
      success: true, 
      data: {
        serviceId,
        endpoint,
        status: 'tested',
        timestamp: new Date().toISOString(),
        message: `Service ${serviceId} test completed`
      }
    });
  } catch (error) {
    console.error('Error testing service:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get cluster configuration
router.get('/config', async (req, res) => {
  try {
    const config = {
      nodes: [
        {
          id: 'jetson',
          name: 'Jetson Orin Nano',
          ip: '192.168.1.177',
          services: ['ollama'],
          role: 'LLM Processing'
        },
        {
          id: 'cpu_coordinator',
          name: 'CPU Coordinator',
          ip: '192.168.1.81',
          services: ['redis-server', 'llama-server', 'haproxy'],
          role: 'Coordination & Load Balancing'
        },
        {
          id: 'rp_embeddings',
          name: 'RPi Embeddings',
          ip: '192.168.1.178',
          services: ['embeddings-server'],
          role: 'Vector Embeddings'
        },
        {
          id: 'worker_tools',
          name: 'Worker Tools',
          ip: '192.168.1.105',
          services: ['tools-server'],
          role: 'Tool Execution'
        },
        {
          id: 'worker_monitor',
          name: 'Worker Monitor',
          ip: '192.168.1.137',
          services: ['monitoring-server'],
          role: 'Monitoring & Analytics'
        }
      ],
      endpoints: {
        llm_balancer: 'http://192.168.1.81:9000',
        tools_balancer: 'http://192.168.1.81:9001',
        embeddings_balancer: 'http://192.168.1.81:9002',
        cluster_health: 'http://192.168.1.137:8083/cluster_health',
        haproxy_stats: 'http://192.168.1.81:9000/haproxy_stats'
      },
      serviceTypes: {
        'ollama': { type: 'LLM', description: 'Local Language Model' },
        'llama-server': { type: 'LLM', description: 'Llama.cpp Server' },
        'embeddings-server': { type: 'Embeddings', description: 'Vector Embeddings Service' },
        'tools-server': { type: 'Tools', description: 'Tool Execution Service' },
        'monitoring-server': { type: 'Monitoring', description: 'Cluster Monitoring' },
        'redis-server': { type: 'Cache', description: 'Redis Cache' },
        'haproxy': { type: 'LoadBalancer', description: 'Load Balancer' }
      }
    };
    
    res.json({ 
      success: true, 
      data: config
    });
  } catch (error) {
    console.error('Error getting cluster config:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get cluster metrics and performance data
router.get('/metrics', async (req, res) => {
  try {
    const clusterManager = (req as any).services.clusterManager as ClusterManager;
    const status = await clusterManager.getCurrentStatus();
    
    // Calculate aggregate metrics
    const metrics = {
      resourceUtilization: {
        avgCpu: status.nodes
          .filter(n => n.resources)
          .reduce((sum, n) => sum + (n.resources?.cpu || 0), 0) / 
          status.nodes.filter(n => n.resources).length || 0,
        
        avgMemory: status.nodes
          .filter(n => n.resources)
          .reduce((sum, n) => sum + (n.resources?.memory || 0), 0) / 
          status.nodes.filter(n => n.resources).length || 0,
        
        avgDisk: status.nodes
          .filter(n => n.resources)
          .reduce((sum, n) => sum + (n.resources?.disk || 0), 0) / 
          status.nodes.filter(n => n.resources).length || 0
      },
      
      serviceMetrics: {
        avgResponseTime: status.services
          .filter(s => s.responseTime)
          .reduce((sum, s) => sum + (s.responseTime || 0), 0) / 
          status.services.filter(s => s.responseTime).length || 0,
        
        healthyServices: status.services.filter(s => s.health === 'healthy').length,
        totalServices: status.services.length
      },
      
      clusterHealth: {
        overallStatus: status.overall,
        uptime: Date.now(), // This would be calculated from cluster start time
        lastStatusCheck: status.lastUpdate
      }
    };
    
    res.json({ 
      success: true, 
      data: metrics
    });
  } catch (error) {
    console.error('Error getting cluster metrics:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

export default router;
