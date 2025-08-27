import { PrismaClient } from '@prisma/client';

export interface DatabaseConfig {
  dbPath?: string;
}

export class DatabaseManager {
  private prisma: PrismaClient;
  private config: DatabaseConfig;

  constructor(config: DatabaseConfig = {}) {
    this.config = config;
    this.prisma = new PrismaClient();
  }

  async connect(): Promise<void> {
    try {
      await this.prisma.$connect();
      console.log('✅ Database connected successfully');
    } catch (error) {
      console.error('❌ Database connection failed:', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    try {
      await this.prisma.$disconnect();
      console.log('✅ Database disconnected successfully');
    } catch (error) {
      console.error('❌ Database disconnection failed:', error);
      throw error;
    }
  }

  async initializeTables(): Promise<void> {
    try {
      // Tables are automatically created by Prisma migrations
      console.log('✅ Database tables initialized');
    } catch (error) {
      console.error('❌ Database table initialization failed:', error);
      throw error;
    }
  }

  // Workflow operations
  async createWorkflow(workflow: {
    name: string;
    description?: string;
    definition: string; // JSON string
    category?: string;
    isTemplate?: boolean;
  }) {
    return await this.prisma.workflow.create({
      data: workflow
    });
  }

  async getWorkflows() {
    return await this.prisma.workflow.findMany({
      include: {
        executions: {
          take: 5,
          orderBy: { createdAt: 'desc' }
        }
      }
    });
  }

  async getWorkflowById(id: string) {
    return await this.prisma.workflow.findUnique({
      where: { id },
      include: {
        executions: {
          include: {
            steps: true
          }
        }
      }
    });
  }

  async updateWorkflow(id: string, data: {
    name?: string;
    description?: string;
    definition?: string;
    category?: string;
  }) {
    return await this.prisma.workflow.update({
      where: { id },
      data
    });
  }

  async deleteWorkflow(id: string) {
    return await this.prisma.workflow.delete({
      where: { id }
    });
  }

  // Workflow execution operations
  async createWorkflowExecution(execution: {
    workflowId: string;
    input?: string; // JSON string
    status?: string;
  }) {
    return await this.prisma.workflowExecution.create({
      data: execution
    });
  }

  async updateWorkflowExecution(id: string, data: {
    status?: string;
    output?: string;
    error?: string;
    progress?: number;
    startedAt?: Date;
    completedAt?: Date;
  }) {
    return await this.prisma.workflowExecution.update({
      where: { id },
      data
    });
  }

  async getWorkflowExecutions(workflowId?: string) {
    return await this.prisma.workflowExecution.findMany({
      where: workflowId ? { workflowId } : undefined,
      include: {
        workflow: true,
        steps: true
      },
      orderBy: { createdAt: 'desc' }
    });
  }

  // Cluster node operations
  async upsertClusterNode(node: {
    name: string;
    ip: string;
    services: string; // JSON string
    status: string;
    resources?: string; // JSON string
  }) {
    return await this.prisma.clusterNode.upsert({
      where: { name: node.name },
      update: {
        ip: node.ip,
        services: node.services,
        status: node.status,
        resources: node.resources,
        lastSeen: new Date()
      },
      create: {
        ...node,
        lastSeen: new Date()
      }
    });
  }

  async getClusterNodes() {
    return await this.prisma.clusterNode.findMany({
      orderBy: { name: 'asc' }
    });
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      await this.prisma.$queryRaw`SELECT 1`;
      return true;
    } catch (error) {
      console.error('Database health check failed:', error);
      return false;
    }
  }

  // Get Prisma client for advanced operations
  getPrismaClient(): PrismaClient {
    return this.prisma;
  }

  // === Compatibility methods for legacy code ===
  
  async initialize(): Promise<void> {
    return this.initializeTables();
  }

  async close(): Promise<void> {
    return this.disconnect();
  }

  async saveWorkflow(workflow: any): Promise<any> {
    if (workflow.id) {
      return this.updateWorkflow(workflow.id, workflow);
    } else {
      return this.createWorkflow(workflow);
    }
  }

  async getWorkflow(id: string): Promise<any> {
    return this.getWorkflowById(id);
  }

  async listWorkflows(tags?: string[]): Promise<any> {
    // TODO: Add tags filtering
    return this.getWorkflows();
  }

  async saveExecution(execution: any): Promise<any> {
    if (execution.id) {
      return this.updateWorkflowExecution(execution.id, execution);
    } else {
      return this.createWorkflowExecution(execution);
    }
  }

  async getExecution(id: string): Promise<any> {
    return await this.prisma.workflowExecution.findUnique({
      where: { id },
      include: { workflow: true, steps: true }
    });
  }

  async listExecutions(workflowId?: string): Promise<any> {
    return this.getWorkflowExecutions(workflowId);
  }

  async getStats(): Promise<any> {
    // Basic stats for compatibility
    const [workflowCount, executionCount, nodeCount] = await Promise.all([
      this.prisma.workflow.count(),
      this.prisma.workflowExecution.count(),
      this.prisma.clusterNode.count()
    ]);
    
    return {
      workflows: workflowCount,
      executions: executionCount,
      nodes: nodeCount,
      dbSize: 0 // TODO: Calculate actual DB size
    };
  }

  async getAnalytics(startDate: Date, endDate: Date): Promise<any> {
    // Basic analytics for compatibility
    const executions = await this.prisma.workflowExecution.findMany({
      where: {
        createdAt: {
          gte: startDate,
          lte: endDate
        }
      }
    });

    return {
      totalExecutions: executions.length,
      successfulExecutions: executions.filter(e => e.status === 'COMPLETED').length,
      failedExecutions: executions.filter(e => e.status === 'FAILED').length,
      averageExecutionTime: 0 // TODO: Calculate actual metrics
    };
  }

  async getLogs(level?: string, limit?: number, offset?: number): Promise<any> {
    // TODO: Implement logging table and retrieval
    return [];
  }

  async logMessage(level: string, message: string, context?: any, source?: string): Promise<void> {
    // TODO: Implement logging to database
    console.log(`[${level}] ${source || 'system'}: ${message}`, context);
  }
}

export default DatabaseManager;