import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import { DatabaseManager } from './DatabaseManager';

export interface WorkflowNode {
  id: string;
  type: 'llm' | 'embeddings' | 'tools' | 'decision' | 'data' | 'output';
  label: string;
  config: {
    service?: 'jetson' | 'cpu' | 'embeddings' | 'tools' | 'monitoring';
    endpoint?: string;
    model?: string;
    prompt?: string;
    parameters?: Record<string, any>;
    condition?: string;
    outputFormat?: string;
  };
  position: { x: number; y: number };
  inputs: string[];
  outputs: string[];
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
  condition?: string;
}

export interface WorkflowDefinition {
  id: string;
  name: string;
  description: string;
  version: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  variables: Record<string, any>;
  settings: {
    timeout: number;
    retries: number;
    parallel: boolean;
  };
  created: Date;
  updated: Date;
  createdBy: string;
  tags: string[];
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentNode?: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  nodeResults: Record<string, any>;
  errors: WorkflowError[];
  started: Date;
  completed?: Date;
  duration?: number;
  executedBy: string;
}

export interface WorkflowError {
  nodeId: string;
  message: string;
  timestamp: Date;
  retryCount: number;
}

export interface NodeExecutionResult {
  success: boolean;
  output: any;
  error?: string;
  duration: number;
  metadata?: Record<string, any>;
}

export class WorkflowEngine extends EventEmitter {
  private databaseManager: DatabaseManager;
  private activeExecutions: Map<string, WorkflowExecution> = new Map();
  
  // Service endpoints based on cluster configuration
  private readonly SERVICE_ENDPOINTS = {
    llm_jetson: 'http://192.168.1.177:11434/api/generate',
    llm_cpu: 'http://192.168.1.81:8080/completion',
    llm_balancer: 'http://192.168.1.81:9000/api/generate',
    embeddings: 'http://192.168.1.81:9002/embeddings',
    tools: 'http://192.168.1.81:9001',
    monitoring: 'http://192.168.1.137:8083'
  };

  constructor(databaseManager: DatabaseManager) {
    super();
    this.databaseManager = databaseManager;
  }

  async createWorkflow(definition: Partial<WorkflowDefinition>, createdBy: string = 'system'): Promise<WorkflowDefinition> {
    const workflow: WorkflowDefinition = {
      id: definition.id || uuidv4(),
      name: definition.name || 'Untitled Workflow',
      description: definition.description || '',
      version: definition.version || '1.0.0',
      nodes: definition.nodes || [],
      edges: definition.edges || [],
      variables: definition.variables || {},
      settings: {
        timeout: 300000, // 5 minutes default
        retries: 3,
        parallel: false,
        ...definition.settings
      },
      created: new Date(),
      updated: new Date(),
      createdBy,
      tags: definition.tags || []
    };

    // Validate workflow
    this.validateWorkflow(workflow);

    // Save to database
    await this.databaseManager.saveWorkflow(workflow);

    console.log(`📝 Created workflow: ${workflow.name} (${workflow.id})`);
    return workflow;
  }

  async updateWorkflow(id: string, updates: Partial<WorkflowDefinition>): Promise<WorkflowDefinition> {
    const existing = await this.databaseManager.getWorkflow(id);
    if (!existing) {
      throw new Error(`Workflow ${id} not found`);
    }

    const updated: WorkflowDefinition = {
      ...existing,
      ...updates,
      id: existing.id, // Prevent ID changes
      updated: new Date()
    };

    // Validate updated workflow
    this.validateWorkflow(updated);

    // Save updates
    await this.databaseManager.saveWorkflow(updated);

    console.log(`📝 Updated workflow: ${updated.name} (${updated.id})`);
    return updated;
  }

  async deleteWorkflow(id: string): Promise<void> {
    await this.databaseManager.deleteWorkflow(id);
    console.log(`🗑️ Deleted workflow: ${id}`);
  }

  async getWorkflow(id: string): Promise<WorkflowDefinition | null> {
    return await this.databaseManager.getWorkflow(id);
  }

  async listWorkflows(tags?: string[]): Promise<WorkflowDefinition[]> {
    return await this.databaseManager.listWorkflows(tags);
  }

  async executeWorkflow(workflowId: string, inputs: Record<string, any>, executedBy: string = 'system'): Promise<WorkflowExecution> {
    const workflow = await this.getWorkflow(workflowId);
    if (!workflow) {
      throw new Error(`Workflow ${workflowId} not found`);
    }

    const execution: WorkflowExecution = {
      id: uuidv4(),
      workflowId,
      status: 'pending',
      progress: 0,
      inputs,
      outputs: {},
      nodeResults: {},
      errors: [],
      started: new Date(),
      executedBy
    };

    // Store execution
    this.activeExecutions.set(execution.id, execution);
    await this.databaseManager.saveExecution(execution);

    console.log(`🚀 Starting workflow execution: ${execution.id}`);

    // Start execution asynchronously
    this.runWorkflowExecution(workflow, execution).catch(error => {
      console.error(`Workflow execution ${execution.id} failed:`, error);
      execution.status = 'failed';
      execution.errors.push({
        nodeId: 'system',
        message: error.message,
        timestamp: new Date(),
        retryCount: 0
      });
      this.updateExecutionStatus(execution);
    });

    return execution;
  }

  async getExecution(executionId: string): Promise<WorkflowExecution | null> {
    return this.activeExecutions.get(executionId) || 
           await this.databaseManager.getExecution(executionId);
  }

  async listExecutions(workflowId?: string): Promise<WorkflowExecution[]> {
    return await this.databaseManager.listExecutions(workflowId);
  }

  async cancelExecution(executionId: string): Promise<void> {
    const execution = this.activeExecutions.get(executionId);
    if (execution && ['pending', 'running'].includes(execution.status)) {
      execution.status = 'cancelled';
      execution.completed = new Date();
      execution.duration = execution.completed.getTime() - execution.started.getTime();
      
      await this.updateExecutionStatus(execution);
      this.activeExecutions.delete(executionId);
      
      console.log(`❌ Cancelled workflow execution: ${executionId}`);
    }
  }

  private async runWorkflowExecution(workflow: WorkflowDefinition, execution: WorkflowExecution): Promise<void> {
    try {
      execution.status = 'running';
      await this.updateExecutionStatus(execution);

      // Build execution graph
      const executionGraph = this.buildExecutionGraph(workflow);
      
      // Execute nodes in topological order
      const totalNodes = workflow.nodes.length;
      let completedNodes = 0;

      for (const nodeId of executionGraph.executionOrder) {
        const node = workflow.nodes.find(n => n.id === nodeId);
        if (!node) continue;

        execution.currentNode = nodeId;
        execution.progress = Math.round((completedNodes / totalNodes) * 100);
        await this.updateExecutionStatus(execution);

        // Check if execution was cancelled
        if (execution.status === 'cancelled' as any) {
          return;
        }

        // Prepare node inputs
        const nodeInputs = this.prepareNodeInputs(node, execution);

        // Execute node with retries
        const result = await this.executeNodeWithRetries(node, nodeInputs, workflow.settings.retries);
        
        execution.nodeResults[nodeId] = result;

        if (!result.success) {
          execution.errors.push({
            nodeId,
            message: result.error || 'Node execution failed',
            timestamp: new Date(),
            retryCount: workflow.settings.retries
          });
          
          // For now, fail fast - could implement partial failure handling
          throw new Error(`Node ${nodeId} failed: ${result.error}`);
        }

        completedNodes++;
      }

      // Extract final outputs
      execution.outputs = this.extractWorkflowOutputs(workflow, execution);
      execution.status = 'completed';
      execution.progress = 100;

    } catch (error) {
      execution.status = 'failed';
      if (!execution.errors.some(e => e.nodeId === 'system')) {
        execution.errors.push({
          nodeId: 'system',
          message: error instanceof Error ? error.message : 'Unknown error',
          timestamp: new Date(),
          retryCount: 0
        });
      }
    } finally {
      execution.completed = new Date();
      execution.duration = execution.completed.getTime() - execution.started.getTime();
      execution.currentNode = undefined as any;
      
      await this.updateExecutionStatus(execution);
      this.activeExecutions.delete(execution.id);
      
      console.log(`✅ Completed workflow execution: ${execution.id} (${execution.status})`);
    }
  }

  private async executeNodeWithRetries(node: WorkflowNode, inputs: Record<string, any>, maxRetries: number): Promise<NodeExecutionResult> {
    let lastError: string = '';
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const result = await this.executeNode(node, inputs);
        if (result.success || attempt === maxRetries) {
          return result;
        }
        lastError = result.error || 'Unknown error';
      } catch (error) {
        lastError = error instanceof Error ? error.message : 'Unknown error';
        if (attempt === maxRetries) {
          return {
            success: false,
            output: null,
            error: lastError,
            duration: 0
          };
        }
      }
      
      // Wait before retry (exponential backoff)
      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }

    return {
      success: false,
      output: null,
      error: lastError,
      duration: 0
    };
  }

  private async executeNode(node: WorkflowNode, inputs: Record<string, any>): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    try {
      switch (node.type) {
        case 'llm':
          return await this.executeLLMNode(node, inputs);
        case 'embeddings':
          return await this.executeEmbeddingsNode(node, inputs);
        case 'tools':
          return await this.executeToolsNode(node, inputs);
        case 'decision':
          return await this.executeDecisionNode(node, inputs);
        case 'data':
          return await this.executeDataNode(node, inputs);
        case 'output':
          return await this.executeOutputNode(node, inputs);
        default:
          throw new Error(`Unknown node type: ${node.type}`);
      }
    } catch (error) {
      return {
        success: false,
        output: null,
        error: error instanceof Error ? error.message : 'Unknown error',
        duration: Date.now() - startTime
      };
    }
  }

  private async executeLLMNode(node: WorkflowNode, inputs: Record<string, any>): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    const prompt = this.interpolateString(node.config.prompt || '', inputs);
    const model = node.config.model || 'llama3.2:3b';
    
    // Choose endpoint based on service preference
    let endpoint = this.SERVICE_ENDPOINTS.llm_balancer; // Default to load balancer
    if (node.config.service === 'jetson') {
      endpoint = this.SERVICE_ENDPOINTS.llm_jetson;
    } else if (node.config.service === 'cpu') {
      endpoint = this.SERVICE_ENDPOINTS.llm_cpu;
    }

    const response = await axios.post(endpoint, {
      model,
      prompt,
      stream: false,
      ...node.config.parameters
    }, { timeout: 30000 });

    const output = response.data.response || response.data.content || response.data;

    return {
      success: true,
      output,
      duration: Date.now() - startTime,
      metadata: {
        model,
        endpoint,
        tokens: response.data.eval_count || 0
      }
    };
  }

  private async executeEmbeddingsNode(node: WorkflowNode, inputs: Record<string, any>): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    const text = this.interpolateString(node.config.prompt || '', inputs);
    
    const response = await axios.post(this.SERVICE_ENDPOINTS.embeddings, {
      texts: [text],
      model: node.config.model || 'default'
    }, { timeout: 10000 });

    return {
      success: true,
      output: response.data.embeddings[0],
      duration: Date.now() - startTime,
      metadata: {
        model: node.config.model,
        dimensions: response.data.embeddings[0]?.length || 0
      }
    };
  }

  private async executeToolsNode(node: WorkflowNode, inputs: Record<string, any>): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    const endpoint = `${this.SERVICE_ENDPOINTS.tools}${node.config.endpoint || '/web_search'}`;
    const parameters = {
      ...node.config.parameters,
      ...inputs
    };

    const response = await axios.post(endpoint, parameters, { timeout: 30000 });

    return {
      success: true,
      output: response.data,
      duration: Date.now() - startTime,
      metadata: {
        endpoint: node.config.endpoint,
        statusCode: response.status
      }
    };
  }

  private async executeDecisionNode(node: WorkflowNode, inputs: Record<string, any>): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    const condition = this.interpolateString(node.config.condition || 'true', inputs);
    
    // Simple condition evaluation (could be enhanced with a proper expression parser)
    const result = this.evaluateCondition(condition, inputs);

    return {
      success: true,
      output: result,
      duration: Date.now() - startTime,
      metadata: {
        condition,
        evaluatedCondition: String(result)
      }
    };
  }

  private async executeDataNode(node: WorkflowNode, inputs: Record<string, any>): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    // Data transformation node
    const output = { ...inputs, ...node.config.parameters };

    return {
      success: true,
      output,
      duration: Date.now() - startTime,
      metadata: {
        transformation: 'data_merge'
      }
    };
  }

  private async executeOutputNode(node: WorkflowNode, inputs: Record<string, any>): Promise<NodeExecutionResult> {
    const startTime = Date.now();
    
    const format = node.config.outputFormat || 'json';
    let output = inputs;
    
    if (format === 'text' && typeof inputs === 'object') {
      output = JSON.stringify(inputs, null, 2) as any;
    }

    return {
      success: true,
      output,
      duration: Date.now() - startTime,
      metadata: {
        format
      }
    };
  }

  private interpolateString(template: string, variables: Record<string, any>): string {
    return template.replace(/\{\{([^}]+)\}\}/g, (match, key) => {
      const value = this.getNestedValue(variables, key.trim());
      return value !== undefined ? String(value) : match;
    });
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  private evaluateCondition(condition: string, inputs: Record<string, any>): boolean {
    // Simple condition evaluation - could be enhanced
    try {
      // Replace variable references with actual values
      const interpolated = this.interpolateString(condition, inputs);
      // For safety, only allow simple comparisons
      if (/^[0-9\s+\-*/<>=!()."'true false]+$/.test(interpolated)) {
        return Boolean(eval(interpolated));
      }
      return Boolean(interpolated);
    } catch {
      return false;
    }
  }

  private buildExecutionGraph(workflow: WorkflowDefinition): { executionOrder: string[] } {
    // Simple topological sort - assumes no cycles
    const visited = new Set<string>();
    const executionOrder: string[] = [];
    
    const visit = (nodeId: string) => {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      
      // Visit dependencies first
      const dependencies = workflow.edges
        .filter(edge => edge.target === nodeId)
        .map(edge => edge.source);
      
      dependencies.forEach(visit);
      executionOrder.push(nodeId);
    };

    workflow.nodes.forEach(node => visit(node.id));
    
    return { executionOrder };
  }

  private prepareNodeInputs(node: WorkflowNode, execution: WorkflowExecution): Record<string, any> {
    const inputs = { ...execution.inputs };
    
    // Add outputs from dependent nodes
    const dependencies = node.inputs || [];
    dependencies.forEach(depId => {
      if (execution.nodeResults[depId]?.success) {
        inputs[depId] = execution.nodeResults[depId].output;
      }
    });

    return inputs;
  }

  private extractWorkflowOutputs(workflow: WorkflowDefinition, execution: WorkflowExecution): Record<string, any> {
    const outputs: Record<string, any> = {};
    
    // Find output nodes
    const outputNodes = workflow.nodes.filter(node => node.type === 'output');
    
    if (outputNodes.length > 0) {
      outputNodes.forEach(node => {
        if (execution.nodeResults[node.id]?.success) {
          outputs[node.id] = execution.nodeResults[node.id].output;
        }
      });
    } else {
      // If no explicit output nodes, return all node results
      Object.keys(execution.nodeResults).forEach(nodeId => {
        if (execution.nodeResults[nodeId].success) {
          outputs[nodeId] = execution.nodeResults[nodeId].output;
        }
      });
    }

    return outputs;
  }

  private validateWorkflow(workflow: WorkflowDefinition): void {
    if (!workflow.name) {
      throw new Error('Workflow name is required');
    }

    if (workflow.nodes.length === 0) {
      throw new Error('Workflow must have at least one node');
    }

    // Validate node IDs are unique
    const nodeIds = workflow.nodes.map(n => n.id);
    if (new Set(nodeIds).size !== nodeIds.length) {
      throw new Error('Node IDs must be unique');
    }

    // Validate edges reference valid nodes
    workflow.edges.forEach(edge => {
      if (!nodeIds.includes(edge.source)) {
        throw new Error(`Edge references unknown source node: ${edge.source}`);
      }
      if (!nodeIds.includes(edge.target)) {
        throw new Error(`Edge references unknown target node: ${edge.target}`);
      }
    });
  }

  private async updateExecutionStatus(execution: WorkflowExecution): Promise<void> {
    await this.databaseManager.saveExecution(execution);
    this.emit('workflowUpdate', {
      type: 'execution_update',
      executionId: execution.id,
      status: execution.status,
      progress: execution.progress,
      currentNode: execution.currentNode
    });
  }

  // Workflow Templates
  async getWorkflowTemplates(): Promise<WorkflowDefinition[]> {
    return [
      await this.createSimpleResearchTemplate(),
      await this.createDocumentQATemplate(),
      await this.createCompetitiveAnalysisTemplate(),
      await this.createContentCreationTemplate()
    ];
  }

  private async createSimpleResearchTemplate(): Promise<WorkflowDefinition> {
    return {
      id: 'template-simple-research',
      name: 'Intelligent Research Assistant',
      description: 'Quick research on emerging topics with source verification',
      version: '1.0.0',
      nodes: [
        {
          id: 'query-analysis',
          type: 'llm',
          label: 'Analyze Query',
          config: {
            service: 'jetson',
            prompt: 'Analyze this research query and generate optimal search terms: {{query}}',
            model: 'llama3.2:3b'
          },
          position: { x: 100, y: 100 },
          inputs: [],
          outputs: ['search-execution']
        },
        {
          id: 'search-execution',
          type: 'tools',
          label: 'Web Search',
          config: {
            endpoint: '/web_search',
            parameters: {
              max_results: 5
            }
          },
          position: { x: 300, y: 100 },
          inputs: ['query-analysis'],
          outputs: ['content-analysis']
        },
        {
          id: 'content-analysis',
          type: 'llm',
          label: 'Analyze Content',
          config: {
            service: 'cpu',
            prompt: 'Analyze and summarize these research results: {{search-execution}}',
            model: 'llama3.2:3b'
          },
          position: { x: 500, y: 100 },
          inputs: ['search-execution'],
          outputs: ['final-output']
        },
        {
          id: 'final-output',
          type: 'output',
          label: 'Research Summary',
          config: {
            outputFormat: 'json'
          },
          position: { x: 700, y: 100 },
          inputs: ['content-analysis'],
          outputs: []
        }
      ],
      edges: [
        { id: 'e1', source: 'query-analysis', target: 'search-execution' },
        { id: 'e2', source: 'search-execution', target: 'content-analysis' },
        { id: 'e3', source: 'content-analysis', target: 'final-output' }
      ],
      variables: {},
      settings: {
        timeout: 300000,
        retries: 2,
        parallel: false
      },
      created: new Date(),
      updated: new Date(),
      createdBy: 'system',
      tags: ['research', 'simple', 'template']
    };
  }

  private async createDocumentQATemplate(): Promise<WorkflowDefinition> {
    return {
      id: 'template-document-qa',
      name: 'Smart Document Q&A System',
      description: 'Instant analysis and Q&A for large technical documents',
      version: '1.0.0',
      nodes: [
        {
          id: 'document-embedding',
          type: 'embeddings',
          label: 'Process Document',
          config: {
            model: 'default'
          },
          position: { x: 100, y: 100 },
          inputs: [],
          outputs: ['query-embedding']
        },
        {
          id: 'query-embedding',
          type: 'embeddings',
          label: 'Process Question',
          config: {
            prompt: '{{question}}',
            model: 'default'
          },
          position: { x: 300, y: 100 },
          inputs: ['document-embedding'],
          outputs: ['similarity-search']
        },
        {
          id: 'similarity-search',
          type: 'decision',
          label: 'Find Relevant Sections',
          config: {
            condition: 'true'
          },
          position: { x: 500, y: 100 },
          inputs: ['query-embedding'],
          outputs: ['answer-generation']
        },
        {
          id: 'answer-generation',
          type: 'llm',
          label: 'Generate Answer',
          config: {
            service: 'cpu',
            prompt: 'Based on this document content, answer the question: {{question}}\n\nContent: {{similarity-search}}',
            model: 'llama3.2:3b'
          },
          position: { x: 700, y: 100 },
          inputs: ['similarity-search'],
          outputs: ['final-answer']
        },
        {
          id: 'final-answer',
          type: 'output',
          label: 'Answer with Citations',
          config: {
            outputFormat: 'json'
          },
          position: { x: 900, y: 100 },
          inputs: ['answer-generation'],
          outputs: []
        }
      ],
      edges: [
        { id: 'e1', source: 'document-embedding', target: 'query-embedding' },
        { id: 'e2', source: 'query-embedding', target: 'similarity-search' },
        { id: 'e3', source: 'similarity-search', target: 'answer-generation' },
        { id: 'e4', source: 'answer-generation', target: 'final-answer' }
      ],
      variables: {},
      settings: {
        timeout: 180000,
        retries: 2,
        parallel: false
      },
      created: new Date(),
      updated: new Date(),
      createdBy: 'system',
      tags: ['document', 'qa', 'template']
    };
  }

  private async createCompetitiveAnalysisTemplate(): Promise<WorkflowDefinition> {
    return {
      id: 'template-competitive-analysis',
      name: 'Automated Competitive Analysis',
      description: 'Complete competitor intelligence gathering and analysis',
      version: '1.0.0',
      nodes: [
        {
          id: 'competitor-discovery',
          type: 'llm',
          label: 'Discover Competitors',
          config: {
            service: 'jetson',
            prompt: 'Identify direct and indirect competitors for this business: {{business_description}}',
            model: 'llama3.2:3b'
          },
          position: { x: 100, y: 100 },
          inputs: [],
          outputs: ['competitor-research']
        },
        {
          id: 'competitor-research',
          type: 'tools',
          label: 'Research Competitors',
          config: {
            endpoint: '/web_search',
            parameters: {
              max_results: 10
            }
          },
          position: { x: 300, y: 100 },
          inputs: ['competitor-discovery'],
          outputs: ['data-analysis']
        },
        {
          id: 'data-analysis',
          type: 'llm',
          label: 'Analyze Data',
          config: {
            service: 'cpu',
            prompt: 'Analyze this competitor data and create insights: {{competitor-research}}',
            model: 'llama3.2:3b'
          },
          position: { x: 500, y: 100 },
          inputs: ['competitor-research'],
          outputs: ['strategic-insights']
        },
        {
          id: 'strategic-insights',
          type: 'llm',
          label: 'Generate Insights',
          config: {
            service: 'cpu',
            prompt: 'Generate strategic recommendations based on this analysis: {{data-analysis}}',
            model: 'llama3.2:3b'
          },
          position: { x: 700, y: 100 },
          inputs: ['data-analysis'],
          outputs: ['final-report']
        },
        {
          id: 'final-report',
          type: 'output',
          label: 'Analysis Report',
          config: {
            outputFormat: 'json'
          },
          position: { x: 900, y: 100 },
          inputs: ['strategic-insights'],
          outputs: []
        }
      ],
      edges: [
        { id: 'e1', source: 'competitor-discovery', target: 'competitor-research' },
        { id: 'e2', source: 'competitor-research', target: 'data-analysis' },
        { id: 'e3', source: 'data-analysis', target: 'strategic-insights' },
        { id: 'e4', source: 'strategic-insights', target: 'final-report' }
      ],
      variables: {},
      settings: {
        timeout: 600000, // 10 minutes
        retries: 2,
        parallel: false
      },
      created: new Date(),
      updated: new Date(),
      createdBy: 'system',
      tags: ['competitive', 'analysis', 'business', 'template']
    };
  }

  private async createContentCreationTemplate(): Promise<WorkflowDefinition> {
    return {
      id: 'template-content-creation',
      name: 'Content Creation Pipeline',
      description: 'Research-backed technical content with verified examples',
      version: '1.0.0',
      nodes: [
        {
          id: 'research-phase',
          type: 'tools',
          label: 'Research Topic',
          config: {
            endpoint: '/web_search',
            parameters: {
              max_results: 8
            }
          },
          position: { x: 100, y: 100 },
          inputs: [],
          outputs: ['content-outline']
        },
        {
          id: 'content-outline',
          type: 'llm',
          label: 'Create Outline',
          config: {
            service: 'jetson',
            prompt: 'Create a detailed content outline for: {{topic}}\n\nBased on research: {{research-phase}}',
            model: 'llama3.2:3b'
          },
          position: { x: 300, y: 100 },
          inputs: ['research-phase'],
          outputs: ['content-writing']
        },
        {
          id: 'content-writing',
          type: 'llm',
          label: 'Write Content',
          config: {
            service: 'cpu',
            prompt: 'Write detailed technical content based on this outline: {{content-outline}}',
            model: 'llama3.2:3b'
          },
          position: { x: 500, y: 100 },
          inputs: ['content-outline'],
          outputs: ['content-optimization']
        },
        {
          id: 'content-optimization',
          type: 'llm',
          label: 'Optimize for SEO',
          config: {
            service: 'cpu',
            prompt: 'Optimize this content for SEO and readability: {{content-writing}}',
            model: 'llama3.2:3b'
          },
          position: { x: 700, y: 100 },
          inputs: ['content-writing'],
          outputs: ['final-content']
        },
        {
          id: 'final-content',
          type: 'output',
          label: 'Published Content',
          config: {
            outputFormat: 'text'
          },
          position: { x: 900, y: 100 },
          inputs: ['content-optimization'],
          outputs: []
        }
      ],
      edges: [
        { id: 'e1', source: 'research-phase', target: 'content-outline' },
        { id: 'e2', source: 'content-outline', target: 'content-writing' },
        { id: 'e3', source: 'content-writing', target: 'content-optimization' },
        { id: 'e4', source: 'content-optimization', target: 'final-content' }
      ],
      variables: {},
      settings: {
        timeout: 480000, // 8 minutes
        retries: 2,
        parallel: false
      },
      created: new Date(),
      updated: new Date(),
      createdBy: 'system',
      tags: ['content', 'creation', 'seo', 'template']
    };
  }
}
