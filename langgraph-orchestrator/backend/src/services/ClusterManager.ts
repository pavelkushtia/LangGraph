import { EventEmitter } from 'events';
import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import axios from 'axios';
import * as path from 'path';

const execAsync = promisify(exec);

export interface ClusterNode {
  id: string;
  name: string;
  ip: string;
  services: string[];
  status: 'online' | 'offline' | 'degraded';
  lastSeen: Date;
  resources?: {
    cpu: number;
    memory: number;
    disk: number;
  };
  sshEnabled?: boolean;
}

export interface ClusterStatus {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  nodes: ClusterNode[];
  services: ServiceStatus[];
  lastUpdate: Date;
  alerts: ClusterAlert[];
}

export interface ServiceStatus {
  id: string;
  name: string;
  node: string;
  status: 'running' | 'stopped' | 'error';
  health: 'healthy' | 'unhealthy' | 'unknown';
  endpoint?: string;
  responseTime?: number;
  lastCheck: Date;
}

export interface ClusterAlert {
  id: string;
  type: 'error' | 'warning' | 'info';
  message: string;
  timestamp: Date;
  resolved: boolean;
}

export class ClusterManager extends EventEmitter {
  private monitoringInterval: NodeJS.Timeout | null = null;
  private orchestratorPath: string;
  
  // Cluster configuration based on the documentation
  private readonly CLUSTER_CONFIG = {
    nodes: [
      {
        id: 'jetson',
        name: 'Jetson Orin Nano',
        ip: '192.168.1.177',
        services: ['ollama']
      },
      {
        id: 'cpu_coordinator',
        name: 'CPU Coordinator',
        ip: '192.168.1.81',
        services: ['redis-server', 'llama-server', 'haproxy']
      },
      {
        id: 'rp_embeddings',
        name: 'RPi Embeddings',
        ip: '192.168.1.178',
        services: ['embeddings-server']
      },
      {
        id: 'worker_tools',
        name: 'Worker Tools',
        ip: '192.168.1.105',
        services: ['tools-server']
      },
      {
        id: 'worker_monitor',
        name: 'Worker Monitor',
        ip: '192.168.1.137',
        services: ['monitoring-server']
      }
    ],
    endpoints: {
      llm_balancer: 'http://192.168.1.81:9000',
      tools_balancer: 'http://192.168.1.81:9001',
      embeddings_balancer: 'http://192.168.1.81:9002',
      cluster_health: 'http://192.168.1.137:8083/cluster_health',
      haproxy_stats: 'http://192.168.1.81:9000/haproxy_stats'
    }
  };

  constructor(orchestratorPath?: string) {
    super();
    // Default to the expected location based on documentation
    this.orchestratorPath = orchestratorPath || '/home/sanzad/ai-infrastructure/langgraph-config';
  }

  async startMonitoring(intervalMs: number = 30000): Promise<void> {
    console.log('🔍 Starting cluster monitoring...');
    
    // Initial status check
    await this.checkClusterStatus();
    
    // Set up periodic monitoring
    this.monitoringInterval = setInterval(async () => {
      try {
        await this.checkClusterStatus();
      } catch (error) {
        console.error('Monitoring error:', error);
        this.emit('error', error);
      }
    }, intervalMs);

    console.log(`✅ Cluster monitoring started (interval: ${intervalMs}ms)`);
  }

  async stopMonitoring(): Promise<void> {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
      console.log('🛑 Cluster monitoring stopped');
    }
  }

  async getCurrentStatus(): Promise<ClusterStatus> {
    return await this.checkClusterStatus();
  }

  private async checkClusterStatus(): Promise<ClusterStatus> {
    const nodes: ClusterNode[] = [];
    const services: ServiceStatus[] = [];
    const alerts: ClusterAlert[] = [];

    try {
      // Check each node and its services
      for (const nodeConfig of this.CLUSTER_CONFIG.nodes) {
        const node = await this.checkNodeStatus(nodeConfig);
        nodes.push(node);

        // Check services on this node
        for (const serviceName of nodeConfig.services) {
          const service = await this.checkServiceStatus(serviceName, nodeConfig);
          services.push(service);

          // Generate alerts for unhealthy services
          if (service.health === 'unhealthy' || service.status === 'error') {
            alerts.push({
              id: `${service.id}-${Date.now()}`,
              type: 'error',
              message: `Service ${service.name} on ${nodeConfig.name} is ${service.status}`,
              timestamp: new Date(),
              resolved: false
            });
          }
        }
      }

      // Check load balancer endpoints
      const balancerServices = await this.checkLoadBalancers();
      services.push(...balancerServices);

      // Determine overall cluster health
      const overall = this.determineOverallHealth(nodes, services);

      const status: ClusterStatus = {
        overall,
        nodes,
        services,
        lastUpdate: new Date(),
        alerts
      };

      // Emit status update
      this.emit('statusUpdate', status);

      return status;

    } catch (error) {
      console.error('Error checking cluster status:', error);
      
      const errorStatus: ClusterStatus = {
        overall: 'unhealthy',
        nodes: [],
        services: [],
        lastUpdate: new Date(),
        alerts: [{
          id: `cluster-error-${Date.now()}`,
          type: 'error',
          message: `Failed to check cluster status: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date(),
          resolved: false
        }]
      };

      this.emit('statusUpdate', errorStatus);
      return errorStatus;
    }
  }

  private async checkNodeStatus(nodeConfig: any): Promise<ClusterNode> {
    try {
      const startTime = Date.now();
      
      if (nodeConfig.id === 'cpu_coordinator') {
        // Local node - check directly
        await this.getLocalUptime();
        const resources = await this.getLocalResources();
        
        return {
          id: nodeConfig.id,
          name: nodeConfig.name,
          ip: nodeConfig.ip,
          services: nodeConfig.services,
          status: 'online',
          lastSeen: new Date(),
          resources
        };
      } else {
        // Remote node - check via SSH ping
        const sshResult = await this.executeSSHCommand(nodeConfig.ip, 'echo "ping"', 5000);
        const responseTime = Date.now() - startTime;
        
        if (!sshResult.success) {
          console.log(`⚠️ SSH connection failed to ${nodeConfig.name} (${nodeConfig.ip}): ${sshResult.error}`);
          console.log(`💡 Run './setup-ssh-keys.sh' to set up passwordless SSH access`);
          return {
            id: nodeConfig.id,
            name: nodeConfig.name,
            ip: nodeConfig.ip,
            services: nodeConfig.services || [],
            status: 'offline',
            lastSeen: new Date(),
            resources: { cpu: 0, memory: 0, disk: 0 }
          };
        }
        
        if (sshResult.success) {
          const resources = await this.getRemoteResources(nodeConfig.ip);
          
          return {
            id: nodeConfig.id,
            name: nodeConfig.name,
            ip: nodeConfig.ip,
            services: nodeConfig.services,
            status: responseTime > 2000 ? 'degraded' : 'online',
            lastSeen: new Date(),
            resources
          };
        } else {
          return {
            id: nodeConfig.id,
            name: nodeConfig.name,
            ip: nodeConfig.ip,
            services: nodeConfig.services,
            status: 'offline',
            lastSeen: new Date()
          };
        }
      }
    } catch (error) {
      return {
        id: nodeConfig.id,
        name: nodeConfig.name,
        ip: nodeConfig.ip,
        services: nodeConfig.services,
        status: 'offline',
        lastSeen: new Date()
      };
    }
  }

  private async checkServiceStatus(serviceName: string, nodeConfig: any): Promise<ServiceStatus> {
    const serviceId = `${nodeConfig.id}-${serviceName}`;
    
    try {
      const startTime = Date.now();
      let isRunning = false;
      let healthStatus: 'healthy' | 'unhealthy' | 'unknown' = 'unknown';
      let endpoint: string | undefined;
      
      if (nodeConfig.id === 'cpu_coordinator') {
        // Check local service
        const { stdout } = await execAsync(`systemctl is-active ${serviceName}`);
        isRunning = stdout.trim() === 'active';
      } else {
        // Check remote service
        const result = await this.executeSSHCommand(
          nodeConfig.ip, 
          `systemctl is-active ${serviceName}`,
          5000
        );
        isRunning = result.success && result.output?.trim() === 'active';
      }

      // Check service health endpoint if available
      if (isRunning) {
        const healthEndpoint = this.getServiceHealthEndpoint(serviceName, nodeConfig.ip);
        if (healthEndpoint) {
          endpoint = healthEndpoint;
          try {
            const response = await axios.get(healthEndpoint, { timeout: 5000 });
            healthStatus = response.status === 200 ? 'healthy' : 'unhealthy';
          } catch {
            healthStatus = 'unhealthy';
          }
        } else {
          healthStatus = 'healthy'; // Assume healthy if no health endpoint
        }
      }

      const responseTime = Date.now() - startTime;

      return {
        id: serviceId,
        name: serviceName,
        node: nodeConfig.id,
        status: isRunning ? 'running' : 'stopped',
        health: healthStatus,
        endpoint: endpoint || undefined,
        responseTime,
        lastCheck: new Date()
      };

    } catch (error) {
      return {
        id: serviceId,
        name: serviceName,
        node: nodeConfig.id,
        status: 'error',
        health: 'unhealthy',
        lastCheck: new Date()
      };
    }
  }

  private async checkLoadBalancers(): Promise<ServiceStatus[]> {
    const balancers = [
      { name: 'LLM Load Balancer', endpoint: this.CLUSTER_CONFIG.endpoints.llm_balancer },
      { name: 'Tools Load Balancer', endpoint: this.CLUSTER_CONFIG.endpoints.tools_balancer },
      { name: 'Embeddings Load Balancer', endpoint: this.CLUSTER_CONFIG.endpoints.embeddings_balancer },
      { name: 'HAProxy Stats', endpoint: this.CLUSTER_CONFIG.endpoints.haproxy_stats }
    ];

    const results: ServiceStatus[] = [];

    for (const balancer of balancers) {
      try {
        const startTime = Date.now();
        const response = await axios.get(`${balancer.endpoint}/health`, { 
          timeout: 5000,
          validateStatus: (status) => status < 500 // Accept 4xx as healthy, but not 5xx
        });
        
        const responseTime = Date.now() - startTime;
        
        results.push({
          id: `balancer-${balancer.name.toLowerCase().replace(/\s+/g, '-')}`,
          name: balancer.name,
          node: 'cpu_coordinator',
          status: 'running',
          health: response.status < 400 ? 'healthy' : 'unhealthy',
          endpoint: balancer.endpoint,
          responseTime,
          lastCheck: new Date()
        });
      } catch (error) {
        results.push({
          id: `balancer-${balancer.name.toLowerCase().replace(/\s+/g, '-')}`,
          name: balancer.name,
          node: 'cpu_coordinator',
          status: 'error',
          health: 'unhealthy',
          endpoint: balancer.endpoint,
          lastCheck: new Date()
        });
      }
    }

    return results;
  }

  private getServiceHealthEndpoint(serviceName: string, nodeIp: string): string | undefined {
    const endpointMap: Record<string, string> = {
      'ollama': `http://${nodeIp}:11434/api/tags`,
      'embeddings-server': `http://${nodeIp}:8081/health`,
      'tools-server': `http://${nodeIp}:8082/health`,
      'monitoring-server': `http://${nodeIp}:8083/health`,
      'haproxy': `http://${nodeIp}:8888/health`
    };

    return endpointMap[serviceName];
  }

  private determineOverallHealth(nodes: ClusterNode[], services: ServiceStatus[]): 'healthy' | 'degraded' | 'unhealthy' {
    const offlineNodes = nodes.filter(n => n.status === 'offline').length;
    const degradedNodes = nodes.filter(n => n.status === 'degraded').length;
    const failedServices = services.filter(s => s.status === 'error' || s.health === 'unhealthy').length;

    if (offlineNodes > 0 || failedServices > 2) {
      return 'unhealthy';
    } else if (degradedNodes > 0 || failedServices > 0) {
      return 'degraded';
    } else {
      return 'healthy';
    }
  }

  async executeAction(action: string, parameters?: any): Promise<any> {
    console.log(`🎬 Executing cluster action: ${action}`, parameters);

    try {
      switch (action) {
        case 'start':
          return await this.startCluster();
        case 'stop':
          return await this.stopCluster();
        case 'restart':
          return await this.restartCluster();
        case 'status':
          return await this.getCurrentStatus();
        case 'test':
          return await this.testCluster();
        default:
          throw new Error(`Unknown action: ${action}`);
      }
    } catch (error) {
      console.error(`Action ${action} failed:`, error);
      throw error;
    }
  }

  private async startCluster(): Promise<any> {
    const result = await this.executeOrchestratorCommand('start');
    await this.checkClusterStatus(); // Update status after start
    return result;
  }

  private async stopCluster(): Promise<any> {
    const result = await this.executeOrchestratorCommand('stop');
    await this.checkClusterStatus(); // Update status after stop
    return result;
  }

  private async restartCluster(): Promise<any> {
    const result = await this.executeOrchestratorCommand('restart');
    await this.checkClusterStatus(); // Update status after restart
    return result;
  }

  private async testCluster(): Promise<any> {
    return await this.executeOrchestratorCommand('test');
  }

  private async executeOrchestratorCommand(command: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const pythonScript = path.join(this.orchestratorPath, 'cluster_orchestrator.py');
      const childProcess = spawn('python3', [pythonScript, command], {
        cwd: this.orchestratorPath,
        env: { ...process.env, PYTHONPATH: this.orchestratorPath }
      });

      let stdout = '';
      let stderr = '';

      childProcess.stdout!.on('data', (data) => {
        stdout += data.toString();
      });

      childProcess.stderr!.on('data', (data) => {
        stderr += data.toString();
      });

      childProcess.on('close', (code) => {
        if (code === 0) {
          resolve({
            success: true,
            command,
            output: stdout,
            exitCode: code
          });
        } else {
          reject(new Error(`Command failed with exit code ${code}: ${stderr}`));
        }
      });

      childProcess.on('error', (error) => {
        reject(error);
      });
    });
  }

  private async executeSSHCommand(ip: string, command: string, timeout: number = 10000): Promise<{success: boolean, output?: string, error?: string}> {
    return new Promise((resolve) => {
      const sshCommand = `ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no -o PreferredAuthentications=publickey sanzad@${ip} '${command}'`;
      
      exec(sshCommand, { timeout }, (error, stdout, _stderr) => {
        if (error) {
          resolve({ success: false, error: error.message });
        } else {
          resolve({ success: true, output: stdout.trim() });
        }
      });
    });
  }

  private async getLocalUptime(): Promise<number> {
    try {
      const { stdout } = await execAsync('uptime -s');
      const bootTime = new Date(stdout.trim());
      return Date.now() - bootTime.getTime();
    } catch {
      return 0;
    }
  }

  private async getLocalResources(): Promise<{ cpu: number; memory: number; disk: number }> {
    try {
      const [cpuResult, memResult, diskResult] = await Promise.all([
        execAsync("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1"),
        execAsync("free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}'"),
        execAsync("df / | tail -1 | awk '{print $5}' | sed 's/%//'")
      ]);

      return {
        cpu: parseFloat(cpuResult.stdout.trim()) || 0,
        memory: parseFloat(memResult.stdout.trim()) || 0,
        disk: parseFloat(diskResult.stdout.trim()) || 0
      };
    } catch {
      return { cpu: 0, memory: 0, disk: 0 };
    }
  }

  private async getRemoteResources(ip: string): Promise<{ cpu: number; memory: number; disk: number }> {
    try {
      const cpuResult = await this.executeSSHCommand(ip, "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1");
      const memResult = await this.executeSSHCommand(ip, "free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}'");
      const diskResult = await this.executeSSHCommand(ip, "df / | tail -1 | awk '{print $5}' | sed 's/%//'");

      return {
        cpu: cpuResult.success ? parseFloat(cpuResult.output || '0') : 0,
        memory: memResult.success ? parseFloat(memResult.output || '0') : 0,
        disk: diskResult.success ? parseFloat(diskResult.output || '0') : 0
      };
    } catch {
      return { cpu: 0, memory: 0, disk: 0 };
    }
  }
}
