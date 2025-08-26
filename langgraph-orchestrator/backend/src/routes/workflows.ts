import express from 'express';
import Joi from 'joi';
import { WorkflowEngine, WorkflowDefinition, WorkflowNode, WorkflowEdge } from '../services/WorkflowEngine';

const router = express.Router();

// Validation schemas
const nodeSchema = Joi.object({
  id: Joi.string().required(),
  type: Joi.string().valid('llm', 'embeddings', 'tools', 'decision', 'data', 'output').required(),
  label: Joi.string().required(),
  config: Joi.object().default({}),
  position: Joi.object({
    x: Joi.number().required(),
    y: Joi.number().required()
  }).required(),
  inputs: Joi.array().items(Joi.string()).default([]),
  outputs: Joi.array().items(Joi.string()).default([])
});

const edgeSchema = Joi.object({
  id: Joi.string().required(),
  source: Joi.string().required(),
  target: Joi.string().required(),
  sourceHandle: Joi.string().optional(),
  targetHandle: Joi.string().optional(),
  condition: Joi.string().optional()
});

const workflowSchema = Joi.object({
  id: Joi.string().optional(),
  name: Joi.string().required(),
  description: Joi.string().default(''),
  version: Joi.string().default('1.0.0'),
  nodes: Joi.array().items(nodeSchema).min(1).required(),
  edges: Joi.array().items(edgeSchema).default([]),
  variables: Joi.object().default({}),
  settings: Joi.object({
    timeout: Joi.number().min(1000).max(3600000).default(300000),
    retries: Joi.number().min(0).max(10).default(3),
    parallel: Joi.boolean().default(false)
  }).default({}),
  tags: Joi.array().items(Joi.string()).default([])
});

const executionInputSchema = Joi.object({
  workflowId: Joi.string().required(),
  inputs: Joi.object().default({}),
  executedBy: Joi.string().default('system')
});

// Get all workflows
router.get('/', async (req, res) => {
  try {
    const { tags } = req.query;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const tagFilter = tags ? (Array.isArray(tags) ? tags as string[] : [tags as string]) : undefined;
    const workflows = await workflowEngine.listWorkflows(tagFilter);
    
    res.json({ 
      success: true, 
      data: workflows,
      count: workflows.length
    });
  } catch (error) {
    console.error('Error listing workflows:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get workflow by ID
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const workflow = await workflowEngine.getWorkflow(id);
    
    if (!workflow) {
      return res.status(404).json({ 
        success: false, 
        error: 'Workflow not found' 
      });
    }
    
    res.json({ 
      success: true, 
      data: workflow
    });
  } catch (error) {
    console.error('Error getting workflow:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Create new workflow
router.post('/', async (req, res) => {
  try {
    const { error, value } = workflowSchema.validate(req.body);
    
    if (error) {
      return res.status(400).json({ 
        success: false, 
        error: error.details[0].message 
      });
    }
    
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    const createdBy = req.body.createdBy || 'system';
    
    const workflow = await workflowEngine.createWorkflow(value, createdBy);
    
    res.status(201).json({ 
      success: true, 
      data: workflow,
      message: 'Workflow created successfully'
    });
  } catch (error) {
    console.error('Error creating workflow:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Update workflow
router.put('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { error, value } = workflowSchema.validate(req.body);
    
    if (error) {
      return res.status(400).json({ 
        success: false, 
        error: error.details[0].message 
      });
    }
    
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const workflow = await workflowEngine.updateWorkflow(id, value);
    
    res.json({ 
      success: true, 
      data: workflow,
      message: 'Workflow updated successfully'
    });
  } catch (error) {
    console.error('Error updating workflow:', error);
    
    if (error instanceof Error && error.message.includes('not found')) {
      return res.status(404).json({ 
        success: false, 
        error: error.message 
      });
    }
    
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Delete workflow
router.delete('/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    await workflowEngine.deleteWorkflow(id);
    
    res.json({ 
      success: true, 
      message: 'Workflow deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting workflow:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Execute workflow
router.post('/:id/execute', async (req, res) => {
  try {
    const { id } = req.params;
    const { inputs = {}, executedBy = 'system' } = req.body;
    
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const execution = await workflowEngine.executeWorkflow(id, inputs, executedBy);
    
    res.status(202).json({ 
      success: true, 
      data: execution,
      message: 'Workflow execution started'
    });
  } catch (error) {
    console.error('Error executing workflow:', error);
    
    if (error instanceof Error && error.message.includes('not found')) {
      return res.status(404).json({ 
        success: false, 
        error: error.message 
      });
    }
    
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get workflow executions
router.get('/:id/executions', async (req, res) => {
  try {
    const { id } = req.params;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const executions = await workflowEngine.listExecutions(id);
    
    res.json({ 
      success: true, 
      data: executions,
      count: executions.length
    });
  } catch (error) {
    console.error('Error getting workflow executions:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get all executions (across all workflows)
router.get('/executions/all', async (req, res) => {
  try {
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const executions = await workflowEngine.listExecutions();
    
    res.json({ 
      success: true, 
      data: executions,
      count: executions.length
    });
  } catch (error) {
    console.error('Error getting all executions:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get specific execution
router.get('/executions/:executionId', async (req, res) => {
  try {
    const { executionId } = req.params;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const execution = await workflowEngine.getExecution(executionId);
    
    if (!execution) {
      return res.status(404).json({ 
        success: false, 
        error: 'Execution not found' 
      });
    }
    
    res.json({ 
      success: true, 
      data: execution
    });
  } catch (error) {
    console.error('Error getting execution:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Cancel execution
router.post('/executions/:executionId/cancel', async (req, res) => {
  try {
    const { executionId } = req.params;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    await workflowEngine.cancelExecution(executionId);
    
    res.json({ 
      success: true, 
      message: 'Execution cancelled successfully'
    });
  } catch (error) {
    console.error('Error cancelling execution:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get workflow templates
router.get('/templates/list', async (req, res) => {
  try {
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const templates = await workflowEngine.getWorkflowTemplates();
    
    res.json({ 
      success: true, 
      data: templates,
      count: templates.length
    });
  } catch (error) {
    console.error('Error getting workflow templates:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Create workflow from template
router.post('/templates/:templateId/create', async (req, res) => {
  try {
    const { templateId } = req.params;
    const { name, description, variables = {}, createdBy = 'system' } = req.body;
    
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    // Get templates and find the requested one
    const templates = await workflowEngine.getWorkflowTemplates();
    const template = templates.find(t => t.id === templateId);
    
    if (!template) {
      return res.status(404).json({ 
        success: false, 
        error: 'Template not found' 
      });
    }
    
    // Create workflow from template
    const workflowData = {
      ...template,
      id: undefined, // Generate new ID
      name: name || `${template.name} - Copy`,
      description: description || template.description,
      variables: { ...template.variables, ...variables },
      created: new Date(),
      updated: new Date()
    };
    
    const workflow = await workflowEngine.createWorkflow(workflowData, createdBy);
    
    res.status(201).json({ 
      success: true, 
      data: workflow,
      message: 'Workflow created from template successfully'
    });
  } catch (error) {
    console.error('Error creating workflow from template:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Validate workflow definition
router.post('/validate', async (req, res) => {
  try {
    const { error, value } = workflowSchema.validate(req.body);
    
    if (error) {
      return res.status(400).json({ 
        success: false, 
        error: error.details[0].message,
        valid: false
      });
    }
    
    // Additional workflow-specific validation
    const validationResults = {
      valid: true,
      errors: [] as string[],
      warnings: [] as string[]
    };
    
    // Check for disconnected nodes
    const nodeIds = new Set(value.nodes.map((n: WorkflowNode) => n.id));
    const connectedNodes = new Set();
    
    value.edges.forEach((edge: WorkflowEdge) => {
      connectedNodes.add(edge.source);
      connectedNodes.add(edge.target);
    });
    
    const disconnectedNodes = [...nodeIds].filter(id => !connectedNodes.has(id));
    if (disconnectedNodes.length > 0 && value.nodes.length > 1) {
      validationResults.warnings.push(`Disconnected nodes found: ${disconnectedNodes.join(', ')}`);
    }
    
    // Check for cycles (simple check)
    const hasOutputNodes = value.nodes.some((n: WorkflowNode) => n.type === 'output');
    if (!hasOutputNodes) {
      validationResults.warnings.push('No output nodes defined');
    }
    
    // Check for required configurations
    value.nodes.forEach((node: WorkflowNode) => {
      if (node.type === 'llm' && !node.config.prompt) {
        validationResults.warnings.push(`LLM node '${node.label}' missing prompt configuration`);
      }
      
      if (node.type === 'tools' && !node.config.endpoint) {
        validationResults.warnings.push(`Tools node '${node.label}' missing endpoint configuration`);
      }
      
      if (node.type === 'decision' && !node.config.condition) {
        validationResults.warnings.push(`Decision node '${node.label}' missing condition configuration`);
      }
    });
    
    res.json({ 
      success: true, 
      data: validationResults
    });
  } catch (error) {
    console.error('Error validating workflow:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

// Get workflow statistics
router.get('/:id/stats', async (req, res) => {
  try {
    const { id } = req.params;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const executions = await workflowEngine.listExecutions(id);
    
    const stats = {
      totalExecutions: executions.length,
      successfulExecutions: executions.filter(e => e.status === 'completed').length,
      failedExecutions: executions.filter(e => e.status === 'failed').length,
      runningExecutions: executions.filter(e => e.status === 'running').length,
      averageDuration: executions
        .filter(e => e.duration)
        .reduce((sum, e) => sum + (e.duration || 0), 0) / 
        executions.filter(e => e.duration).length || 0,
      lastExecution: executions.length > 0 ? executions[0] : null,
      successRate: executions.length > 0 ? 
        (executions.filter(e => e.status === 'completed').length / executions.length) * 100 : 0
    };
    
    res.json({ 
      success: true, 
      data: stats
    });
  } catch (error) {
    console.error('Error getting workflow stats:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
});

export default router;
