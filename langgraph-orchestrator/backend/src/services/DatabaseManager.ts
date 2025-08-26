import sqlite3 from 'sqlite3';
import * as path from 'path';
import { WorkflowDefinition, WorkflowExecution } from './WorkflowEngine';

export interface DatabaseConfig {
  dbPath?: string;
  enableWAL?: boolean;
  busyTimeout?: number;
}

export class DatabaseManager {
  private db: sqlite3.Database | null = null;
  private config: DatabaseConfig;

  private run(sql: string, params: any[] = []): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      this.db.run(sql, params, function(err) {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  private get(sql: string, params: any[] = []): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      this.db.get(sql, params, (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  private all(sql: string, params: any[] = []): Promise<any[]> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }
      this.db.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows || []);
      });
    });
  }

  constructor(config: DatabaseConfig = {}) {
    this.config = {
      dbPath: config.dbPath || path.join(process.cwd(), 'data', 'langgraph-orchestrator.db'),
      enableWAL: config.enableWAL ?? true,
      busyTimeout: config.busyTimeout ?? 5000
    };
  }

  async initialize(): Promise<void> {
    return new Promise((resolve, reject) => {
      // Ensure data directory exists
      const dataDir = path.dirname(this.config.dbPath!);
      const fs = require('fs');
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }

      this.db = new sqlite3.Database(this.config.dbPath!, (err) => {
        if (err) {
          reject(err);
          return;
        }

        console.log(`📁 Connected to SQLite database: ${this.config.dbPath}`);
        this.initializeTables().then(resolve).catch(reject);
      });

      // Configure database settings
      if (this.db) {
        this.db.configure('busyTimeout', this.config.busyTimeout!);
        
        if (this.config.enableWAL) {
          this.db.run('PRAGMA journal_mode=WAL;');
          this.db.run('PRAGMA synchronous=NORMAL;');
        }
      }
    });
  }

  private async initializeTables(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    // Workflows table
    await this.run(`
      CREATE TABLE IF NOT EXISTS workflows (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        version TEXT DEFAULT '1.0.0',
        definition TEXT NOT NULL,
        created DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_by TEXT DEFAULT 'system',
        tags TEXT DEFAULT '[]',
        UNIQUE(id)
      )
    `);

    // Workflow executions table
    await this.run(`
      CREATE TABLE IF NOT EXISTS executions (
        id TEXT PRIMARY KEY,
        workflow_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        progress INTEGER DEFAULT 0,
        current_node TEXT,
        inputs TEXT DEFAULT '{}',
        outputs TEXT DEFAULT '{}',
        node_results TEXT DEFAULT '{}',
        errors TEXT DEFAULT '[]',
        started DATETIME DEFAULT CURRENT_TIMESTAMP,
        completed DATETIME,
        duration INTEGER,
        executed_by TEXT DEFAULT 'system',
        FOREIGN KEY (workflow_id) REFERENCES workflows (id),
        UNIQUE(id)
      )
    `);

    // Analytics table for tracking metrics
    await this.run(`
      CREATE TABLE IF NOT EXISTS analytics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_data TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        workflow_id TEXT,
        execution_id TEXT,
        user_id TEXT
      )
    `);

    // System logs table
    await this.run(`
      CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        context TEXT,
        source TEXT
      )
    `);

    // Create indexes for better performance
    await this.run('CREATE INDEX IF NOT EXISTS idx_workflows_created ON workflows(created)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_workflows_tags ON workflows(tags)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_executions_workflow_id ON executions(workflow_id)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_executions_started ON executions(started)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_analytics_workflow_id ON analytics(workflow_id)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)');
    await this.run('CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)');

    console.log('✅ Database tables initialized');
  }

  async saveWorkflow(workflow: WorkflowDefinition): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    await this.run(`
      INSERT OR REPLACE INTO workflows 
      (id, name, description, version, definition, created, updated, created_by, tags)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      workflow.id,
      workflow.name,
      workflow.description,
      workflow.version,
      JSON.stringify({
        nodes: workflow.nodes,
        edges: workflow.edges,
        variables: workflow.variables,
        settings: workflow.settings
      }),
      workflow.created.toISOString(),
      workflow.updated.toISOString(),
      workflow.createdBy,
      JSON.stringify(workflow.tags)
    ]);

    await this.logEvent('workflow_saved', { workflowId: workflow.id, name: workflow.name });
  }

  async getWorkflow(id: string): Promise<WorkflowDefinition | null> {
    if (!this.db) throw new Error('Database not initialized');
    
    const row = await this.get(`
      SELECT * FROM workflows WHERE id = ?
    `, [id]) as any;

    if (!row) return null;

    const definition = JSON.parse(row.definition);
    
    return {
      id: row.id,
      name: row.name,
      description: row.description,
      version: row.version,
      nodes: definition.nodes || [],
      edges: definition.edges || [],
      variables: definition.variables || {},
      settings: definition.settings || { timeout: 300000, retries: 3, parallel: false },
      created: new Date(row.created),
      updated: new Date(row.updated),
      createdBy: row.created_by,
      tags: JSON.parse(row.tags || '[]')
    };
  }

  async listWorkflows(tags?: string[]): Promise<WorkflowDefinition[]> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    let query = 'SELECT * FROM workflows ORDER BY updated DESC';
    let params: any[] = [];

    if (tags && tags.length > 0) {
      const tagConditions = tags.map(() => 'tags LIKE ?').join(' OR ');
      query = `SELECT * FROM workflows WHERE (${tagConditions}) ORDER BY updated DESC`;
      params = tags.map(tag => `%"${tag}"%`);
    }

    const rows = await this.all(query, params) as any[];

    return rows.map(row => {
      const definition = JSON.parse(row.definition);
      
      return {
        id: row.id,
        name: row.name,
        description: row.description,
        version: row.version,
        nodes: definition.nodes || [],
        edges: definition.edges || [],
        variables: definition.variables || {},
        settings: definition.settings || { timeout: 300000, retries: 3, parallel: false },
        created: new Date(row.created),
        updated: new Date(row.updated),
        createdBy: row.created_by,
        tags: JSON.parse(row.tags || '[]')
      };
    });
  }

  async deleteWorkflow(id: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    // Delete associated executions first
    await this.run('DELETE FROM executions WHERE workflow_id = ?', [id]);
    
    // Delete the workflow
    await this.run('DELETE FROM workflows WHERE id = ?', [id]);

    await this.logEvent('workflow_deleted', { workflowId: id });
  }

  async saveExecution(execution: WorkflowExecution): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    await this.run(`
      INSERT OR REPLACE INTO executions 
      (id, workflow_id, status, progress, current_node, inputs, outputs, node_results, errors, 
       started, completed, duration, executed_by)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, [
      execution.id,
      execution.workflowId,
      execution.status,
      execution.progress,
      execution.currentNode || null,
      JSON.stringify(execution.inputs),
      JSON.stringify(execution.outputs),
      JSON.stringify(execution.nodeResults),
      JSON.stringify(execution.errors),
      execution.started.toISOString(),
      execution.completed?.toISOString() || null,
      execution.duration || null,
      execution.executedBy
    ]);

    await this.logEvent('execution_updated', { 
      executionId: execution.id, 
      workflowId: execution.workflowId,
      status: execution.status,
      progress: execution.progress
    });
  }

  async getExecution(id: string): Promise<WorkflowExecution | null> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    const row = await this.get(`
      SELECT * FROM executions WHERE id = ?
    `, [id]) as any;

    if (!row) return null;

    return {
      id: row.id,
      workflowId: row.workflow_id,
      status: row.status,
      progress: row.progress,
      currentNode: row.current_node,
      inputs: JSON.parse(row.inputs),
      outputs: JSON.parse(row.outputs),
      nodeResults: JSON.parse(row.node_results),
      errors: JSON.parse(row.errors),
      started: new Date(row.started),
      completed: row.completed ? new Date(row.completed) : undefined,
      duration: row.duration,
      executedBy: row.executed_by
    };
  }

  async listExecutions(workflowId?: string): Promise<WorkflowExecution[]> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    let query = 'SELECT * FROM executions ORDER BY started DESC LIMIT 100';
    let params: any[] = [];

    if (workflowId) {
      query = 'SELECT * FROM executions WHERE workflow_id = ? ORDER BY started DESC LIMIT 100';
      params = [workflowId];
    }

    const rows = await this.all(query, params) as any[];

    return rows.map(row => ({
      id: row.id,
      workflowId: row.workflow_id,
      status: row.status,
      progress: row.progress,
      currentNode: row.current_node,
      inputs: JSON.parse(row.inputs),
      outputs: JSON.parse(row.outputs),
      nodeResults: JSON.parse(row.node_results),
      errors: JSON.parse(row.errors),
      started: new Date(row.started),
      completed: row.completed ? new Date(row.completed) : undefined,
      duration: row.duration,
      executedBy: row.executed_by
    }));
  }

  async getAnalytics(startDate?: Date, endDate?: Date): Promise<any> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    let dateFilter = '';
    let params: any[] = [];

    if (startDate && endDate) {
      dateFilter = 'WHERE timestamp BETWEEN ? AND ?';
      params = [startDate.toISOString(), endDate.toISOString()];
    } else if (startDate) {
      dateFilter = 'WHERE timestamp >= ?';
      params = [startDate.toISOString()];
    }

    // Workflow execution statistics
    const executionStats = await this.all(`
      SELECT 
        status,
        COUNT(*) as count,
        AVG(duration) as avg_duration,
        MIN(duration) as min_duration,
        MAX(duration) as max_duration
      FROM executions 
      ${dateFilter}
      GROUP BY status
    `, params) as any[];

    // Daily execution counts
    const dailyStats = await this.all(`
      SELECT 
        DATE(started) as date,
        COUNT(*) as executions,
        AVG(duration) as avg_duration
      FROM executions 
      ${dateFilter}
      GROUP BY DATE(started)
      ORDER BY date DESC
      LIMIT 30
    `, params) as any[];

    // Popular workflows
    const popularWorkflows = await this.all(`
      SELECT 
        w.name,
        w.id,
        COUNT(e.id) as execution_count,
        AVG(e.duration) as avg_duration
      FROM workflows w
      LEFT JOIN executions e ON w.id = e.workflow_id
      ${dateFilter ? `WHERE e.started ${dateFilter.substring(5)}` : ''}
      GROUP BY w.id, w.name
      ORDER BY execution_count DESC
      LIMIT 10
    `, params) as any[];

    // Error analysis
    const errorStats = await this.all(`
      SELECT 
        workflow_id,
        COUNT(*) as error_count
      FROM executions 
      WHERE status = 'failed' ${dateFilter ? `AND started ${dateFilter.substring(5)}` : ''}
      GROUP BY workflow_id
      ORDER BY error_count DESC
      LIMIT 10
    `, params) as any[];

    return {
      executionStats,
      dailyStats,
      popularWorkflows,
      errorStats,
      period: {
        start: startDate?.toISOString(),
        end: endDate?.toISOString()
      }
    };
  }

  async logEvent(eventType: string, eventData: any, workflowId?: string, executionId?: string, userId?: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    await this.run(`
      INSERT INTO analytics (event_type, event_data, workflow_id, execution_id, user_id)
      VALUES (?, ?, ?, ?, ?)
    `, [
      eventType,
      JSON.stringify(eventData),
      workflowId || null,
      executionId || null,
      userId || null
    ]);
  }

  async logMessage(level: 'info' | 'warn' | 'error' | 'debug', message: string, context?: any, source?: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    await this.run(`
      INSERT INTO logs (level, message, context, source)
      VALUES (?, ?, ?, ?)
    `, [
      level,
      message,
      context ? JSON.stringify(context) : null,
      source || null
    ]);
  }

  async getLogs(limit: number = 100, level?: string): Promise<any[]> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    let query = 'SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?';
    let params: any[] = [limit];

    if (level) {
      query = 'SELECT * FROM logs WHERE level = ? ORDER BY timestamp DESC LIMIT ?';
      params = [level, limit];
    }

    const rows = await this.all(query, params) as any[];

    return rows.map(row => ({
      id: row.id,
      level: row.level,
      message: row.message,
      timestamp: row.timestamp,
      context: row.context ? JSON.parse(row.context) : null,
      source: row.source
    }));
  }

  async cleanup(olderThanDays: number = 30): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - olderThanDays);

    // Clean up old executions
    await this.run(`
      DELETE FROM executions 
      WHERE started < ? AND status IN ('completed', 'failed', 'cancelled')
    `, [cutoffDate.toISOString()]);

    // Clean up old analytics
    await this.run(`
      DELETE FROM analytics 
      WHERE timestamp < ?
    `, [cutoffDate.toISOString()]);

    // Clean up old logs
    await this.run(`
      DELETE FROM logs 
      WHERE timestamp < ?
    `, [cutoffDate.toISOString()]);

    console.log(`🧹 Cleaned up data older than ${olderThanDays} days`);
  }

  async getStats(): Promise<any> {
    if (!this.db) throw new Error('Database not initialized');

    
    
    const workflowCount = await this.get('SELECT COUNT(*) as count FROM workflows') as any;
    const executionCount = await this.get('SELECT COUNT(*) as count FROM executions') as any;
    const activeExecutions = await this.get('SELECT COUNT(*) as count FROM executions WHERE status IN ("pending", "running")') as any;
    const recentExecutions = await this.get('SELECT COUNT(*) as count FROM executions WHERE started > datetime("now", "-24 hours")') as any;

    return {
      workflows: workflowCount.count,
      totalExecutions: executionCount.count,
      activeExecutions: activeExecutions.count,
      recentExecutions: recentExecutions.count
    };
  }

  async close(): Promise<void> {
    if (this.db) {
      return new Promise((resolve) => {
        this.db!.close((err) => {
          if (err) {
            console.error('Error closing database:', err);
          } else {
            console.log('📁 Database connection closed');
          }
          this.db = null;
          resolve();
        });
      });
    }
  }
}
