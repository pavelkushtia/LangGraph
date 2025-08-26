import express from 'express';
import { DatabaseManager } from '../services/DatabaseManager';
import { WorkflowEngine } from '../services/WorkflowEngine';
import { RealTimeService } from '../services/RealTimeService';

const router = express.Router();

// Get comprehensive analytics dashboard data
router.get('/dashboard', async (req, res) => {
  try {
    const { startDate, endDate } = req.query;
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    const realTimeService = (req as any).services.realTimeService as RealTimeService;

    // Parse date range
    const start = startDate ? new Date(startDate as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
    const end = endDate ? new Date(endDate as string) : new Date();

    // Get analytics data
    const analyticsData = await databaseManager.getAnalytics(start, end);
    
    // Get workflow statistics
    const workflows = await workflowEngine.listWorkflows();
    const allExecutions = await workflowEngine.listExecutions();
    
    // Get real-time metrics
    const realtimeAnalytics = realTimeService.getRealtimeAnalytics();
    
    // Calculate additional metrics
    const totalWorkflows = workflows.length;
    const totalExecutions = allExecutions.length;
    const activeExecutions = allExecutions.filter(e => ['pending', 'running'].includes(e.status)).length;
    const recentExecutions = allExecutions.filter(e => e.started >= start).length;
    
    // Success rate calculation
    const completedExecutions = allExecutions.filter(e => e.status === 'completed').length;
    const failedExecutions = allExecutions.filter(e => e.status === 'failed').length;
    const successRate = totalExecutions > 0 ? (completedExecutions / totalExecutions) * 100 : 0;
    
    // Average execution time
    const executionsWithDuration = allExecutions.filter(e => e.duration && e.duration > 0);
    const avgExecutionTime = executionsWithDuration.length > 0 
      ? executionsWithDuration.reduce((sum, e) => sum + (e.duration || 0), 0) / executionsWithDuration.length
      : 0;

    // Tag popularity
    const tagCounts = workflows.reduce((acc, workflow) => {
      workflow.tags.forEach(tag => {
        acc[tag] = (acc[tag] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);

    const dashboardData = {
      summary: {
        totalWorkflows,
        totalExecutions,
        activeExecutions,
        recentExecutions,
        successRate: Math.round(successRate * 100) / 100,
        avgExecutionTime: Math.round(avgExecutionTime),
        connectedClients: realtimeAnalytics.clientStats.totalClients
      },
      executionStats: analyticsData.executionStats,
      dailyStats: analyticsData.dailyStats,
      popularWorkflows: analyticsData.popularWorkflows,
      errorStats: analyticsData.errorStats,
      tagPopularity: Object.entries(tagCounts)
        .sort(([,a], [,b]) => (b as number) - (a as number))
        .slice(0, 10)
        .map(([tag, count]) => ({ tag, count })),
      realtimeMetrics: realtimeAnalytics,
      period: {
        start: start.toISOString(),
        end: end.toISOString()
      }
    };

    res.json({
      success: true,
      data: dashboardData
    });
  } catch (error) {
    console.error('Error getting dashboard analytics:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get workflow execution analytics
router.get('/executions', async (req, res) => {
  try {
    const { startDate, endDate, workflowId } = req.query;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;

    // Parse date range
    const start = startDate ? new Date(startDate as string) : new Date(Date.now() - 7 * 24 * 60 * 60 * 1000); // 7 days ago
    const end = endDate ? new Date(endDate as string) : new Date();

    // Get executions
    let executions = await workflowEngine.listExecutions(workflowId as string);
    
    // Filter by date range
    executions = executions.filter(e => e.started >= start && e.started <= end);

    // Calculate analytics
    const analytics = {
      period: {
        start: start.toISOString(),
        end: end.toISOString()
      },
      totalExecutions: executions.length,
      statusBreakdown: {
        completed: executions.filter(e => e.status === 'completed').length,
        failed: executions.filter(e => e.status === 'failed').length,
        running: executions.filter(e => e.status === 'running').length,
        pending: executions.filter(e => e.status === 'pending').length,
        cancelled: executions.filter(e => e.status === 'cancelled').length
      },
      performanceMetrics: {
        avgDuration: executions
          .filter(e => e.duration)
          .reduce((sum, e) => sum + (e.duration || 0), 0) / 
          executions.filter(e => e.duration).length || 0,
        
        minDuration: Math.min(...executions
          .filter(e => e.duration)
          .map(e => e.duration || 0)) || 0,
        
        maxDuration: Math.max(...executions
          .filter(e => e.duration)
          .map(e => e.duration || 0)) || 0,
        
        successRate: executions.length > 0 
          ? (executions.filter(e => e.status === 'completed').length / executions.length) * 100 
          : 0
      },
      timeSeriesData: (this as any).generateTimeSeriesData(executions, start, end),
      errorAnalysis: (this as any).analyzeErrors(executions),
      executionTrends: (this as any).generateExecutionTrends(executions)
    };

    res.json({
      success: true,
      data: analytics
    });
  } catch (error) {
    console.error('Error getting execution analytics:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get workflow popularity and usage analytics
router.get('/workflows', async (req, res) => {
  try {
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;
    
    const workflows = await workflowEngine.listWorkflows();
    const allExecutions = await workflowEngine.listExecutions();

    // Calculate workflow analytics
    const workflowAnalytics = workflows.map(workflow => {
      const executions = allExecutions.filter(e => e.workflowId === workflow.id);
      const successfulExecutions = executions.filter(e => e.status === 'completed');
      const failedExecutions = executions.filter(e => e.status === 'failed');
      
      const avgDuration = executions
        .filter(e => e.duration)
        .reduce((sum, e) => sum + (e.duration || 0), 0) / 
        executions.filter(e => e.duration).length || 0;

      return {
        workflowId: workflow.id,
        name: workflow.name,
        description: workflow.description,
        tags: workflow.tags,
        created: workflow.created,
        updated: workflow.updated,
        metrics: {
          totalExecutions: executions.length,
          successfulExecutions: successfulExecutions.length,
          failedExecutions: failedExecutions.length,
          successRate: executions.length > 0 ? (successfulExecutions.length / executions.length) * 100 : 0,
          avgDuration: Math.round(avgDuration),
          lastExecution: executions.length > 0 ? executions[0].started : null,
          nodeCount: workflow.nodes.length,
          edgeCount: workflow.edges.length
        }
      };
    });

    // Sort by popularity (total executions)
    workflowAnalytics.sort((a, b) => b.metrics.totalExecutions - a.metrics.totalExecutions);

    // Calculate additional insights
    const insights = {
      totalWorkflows: workflows.length,
      avgWorkflowComplexity: workflows.reduce((sum, w) => sum + w.nodes.length, 0) / workflows.length || 0,
      mostPopularTags: (this as any).calculateTagPopularity(workflows),
      workflowsByStatus: {
        neverExecuted: workflowAnalytics.filter(w => w.metrics.totalExecutions === 0).length,
        active: workflowAnalytics.filter(w => w.metrics.totalExecutions > 0 && w.metrics.lastExecution && 
          new Date(w.metrics.lastExecution) > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)).length,
        inactive: workflowAnalytics.filter(w => w.metrics.lastExecution && 
          new Date(w.metrics.lastExecution) <= new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)).length
      }
    };

    res.json({
      success: true,
      data: {
        workflows: workflowAnalytics,
        insights
      }
    });
  } catch (error) {
    console.error('Error getting workflow analytics:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get performance metrics over time
router.get('/performance', async (req, res) => {
  try {
    const { startDate, endDate, granularity = 'day' } = req.query;
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    
    // Parse date range
    const start = startDate ? new Date(startDate as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    const end = endDate ? new Date(endDate as string) : new Date();

    // Get analytics data
    const analyticsData = await databaseManager.getAnalytics(start, end);

    // Performance trends
    const performanceData = {
      period: {
        start: start.toISOString(),
        end: end.toISOString(),
        granularity
      },
      executionTrends: analyticsData.dailyStats,
      performanceMetrics: {
        avgDailyExecutions: analyticsData.dailyStats.length > 0 
          ? analyticsData.dailyStats.reduce((sum, day) => sum + day.executions, 0) / analyticsData.dailyStats.length
          : 0,
        
        avgDuration: analyticsData.dailyStats.length > 0
          ? analyticsData.dailyStats
              .filter(day => day.avg_duration)
              .reduce((sum, day) => sum + (day.avg_duration || 0), 0) / 
            analyticsData.dailyStats.filter(day => day.avg_duration).length
          : 0,
        
        peakDay: analyticsData.dailyStats.length > 0 
          ? analyticsData.dailyStats.reduce((max, day) => day.executions > max.executions ? day : max)
          : null
      },
      errorTrends: analyticsData.errorStats,
      resourceUtilization: {
        // This would include cluster resource metrics
        // For now, providing placeholder structure
        cpu: [],
        memory: [],
        disk: []
      }
    };

    res.json({
      success: true,
      data: performanceData
    });
  } catch (error) {
    console.error('Error getting performance analytics:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get real-time analytics
router.get('/realtime', async (req, res) => {
  try {
    const realTimeService = (req as any).services.realTimeService as RealTimeService;
    const analytics = realTimeService.getRealtimeAnalytics();

    res.json({
      success: true,
      data: analytics
    });
  } catch (error) {
    console.error('Error getting real-time analytics:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Export analytics data
router.get('/export', async (req, res) => {
  try {
    const { format = 'json', startDate, endDate } = req.query;
    const databaseManager = (req as any).services.databaseManager as DatabaseManager;
    const workflowEngine = (req as any).services.workflowEngine as WorkflowEngine;

    // Parse date range
    const start = startDate ? new Date(startDate as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    const end = endDate ? new Date(endDate as string) : new Date();

    // Get comprehensive data
    const analyticsData = await databaseManager.getAnalytics(start, end);
    const workflows = await workflowEngine.listWorkflows();
    const executions = await workflowEngine.listExecutions();

    const exportData = {
      exportTimestamp: new Date().toISOString(),
      period: {
        start: start.toISOString(),
        end: end.toISOString()
      },
      summary: {
        totalWorkflows: workflows.length,
        totalExecutions: executions.length
      },
      workflows: workflows,
      executions: executions.filter(e => e.started >= start && e.started <= end),
      analytics: analyticsData
    };

    if (format === 'csv') {
      // Convert to CSV format
      const csv = (this as any).convertToCSV(exportData);
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename=langgraph-analytics-${Date.now()}.csv`);
      res.send(csv);
    } else {
      // JSON format
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Disposition', `attachment; filename=langgraph-analytics-${Date.now()}.json`);
      res.json(exportData);
    }
  } catch (error) {
    console.error('Error exporting analytics:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Helper methods (these would typically be in a separate utility file)

function generateTimeSeriesData(executions: any[], start: Date, end: Date): any[] {
  const timeSlots = [];
  const slotDuration = 24 * 60 * 60 * 1000; // 1 day
  
  for (let time = start.getTime(); time <= end.getTime(); time += slotDuration) {
    const slotStart = new Date(time);
    const slotEnd = new Date(time + slotDuration);
    
    const slotExecutions = executions.filter(e => 
      e.started >= slotStart && e.started < slotEnd
    );
    
    timeSlots.push({
      timestamp: slotStart.toISOString(),
      totalExecutions: slotExecutions.length,
      successfulExecutions: slotExecutions.filter(e => e.status === 'completed').length,
      failedExecutions: slotExecutions.filter(e => e.status === 'failed').length,
      avgDuration: slotExecutions
        .filter(e => e.duration)
        .reduce((sum, e) => sum + (e.duration || 0), 0) / 
        slotExecutions.filter(e => e.duration).length || 0
    });
  }
  
  return timeSlots;
}

function analyzeErrors(executions: any[]): any {
  const failedExecutions = executions.filter(e => e.status === 'failed');
  
  const errorPatterns = failedExecutions.reduce((acc, execution) => {
    execution.errors.forEach((error: any) => {
      const key = `${error.nodeId}:${error.message}`;
      if (!acc[key]) {
        acc[key] = {
          nodeId: error.nodeId,
          message: error.message,
          count: 0,
          firstOccurrence: error.timestamp,
          lastOccurrence: error.timestamp
        };
      }
      acc[key].count++;
      if (new Date(error.timestamp) > new Date(acc[key].lastOccurrence)) {
        acc[key].lastOccurrence = error.timestamp;
      }
    });
    return acc;
  }, {} as Record<string, any>);

  return {
    totalErrors: failedExecutions.length,
    uniqueErrorPatterns: Object.keys(errorPatterns).length,
    mostCommonErrors: Object.values(errorPatterns)
      .sort((a: any, b: any) => b.count - a.count)
      .slice(0, 10)
  };
}

function generateExecutionTrends(executions: any[]): any {
  const now = new Date();
  const periods = [
    { name: 'Last 24 hours', start: new Date(now.getTime() - 24 * 60 * 60 * 1000) },
    { name: 'Last 7 days', start: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000) },
    { name: 'Last 30 days', start: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000) }
  ];

  return periods.map(period => {
    const periodExecutions = executions.filter(e => e.started >= period.start);
    return {
      period: period.name,
      totalExecutions: periodExecutions.length,
      successRate: periodExecutions.length > 0 
        ? (periodExecutions.filter(e => e.status === 'completed').length / periodExecutions.length) * 100 
        : 0,
      avgDuration: periodExecutions
        .filter(e => e.duration)
        .reduce((sum, e) => sum + (e.duration || 0), 0) / 
        periodExecutions.filter(e => e.duration).length || 0
    };
  });
}

function calculateTagPopularity(workflows: any[]): any[] {
  const tagCounts = workflows.reduce((acc, workflow) => {
    workflow.tags.forEach((tag: string) => {
      acc[tag] = (acc[tag] || 0) + 1;
    });
    return acc;
  }, {} as Record<string, number>);

  return Object.entries(tagCounts)
    .sort(([,a], [,b]) => (b as number) - (a as number))
    .slice(0, 10)
    .map(([tag, count]) => ({ tag, count }));
}

function convertToCSV(data: any): string {
  // Simple CSV conversion for executions data
  const executions = data.executions || [];
  
  const headers = ['Execution ID', 'Workflow ID', 'Status', 'Started', 'Duration', 'Executed By'];
  const rows = executions.map((e: any) => [
    e.id,
    e.workflowId,
    e.status,
    e.started,
    e.duration || '',
    e.executedBy
  ]);

  const csvContent = [headers, ...rows]
    .map(row => row.map(field => `"${field}"`).join(','))
    .join('\n');

  return csvContent;
}

// Attach helper functions to router for use in route handlers
(router as any).generateTimeSeriesData = generateTimeSeriesData;
(router as any).analyzeErrors = analyzeErrors;
(router as any).generateExecutionTrends = generateExecutionTrends;
(router as any).calculateTagPopularity = calculateTagPopularity;
(router as any).convertToCSV = convertToCSV;

export default router;
