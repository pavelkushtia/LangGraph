import React, { useState, useEffect } from 'react';

interface ClusterNode {
  name: string;
  ip: string;
  status: 'online' | 'offline' | 'degraded';
  services: string[];
}

const Dashboard: React.FC = () => {
  const [clusterStatus, setClusterStatus] = useState<string>('checking...');
  const [nodes, setNodes] = useState<ClusterNode[]>([]);

  useEffect(() => {
    // Fetch cluster status from backend
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/cluster/status');
        if (response.ok) {
          const data = await response.json();
          setClusterStatus(data.overall || 'unknown');
          setNodes(data.nodes || []);
        }
      } catch (error) {
        console.error('Failed to fetch cluster status:', error);
        setClusterStatus('error');
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Cluster Dashboard</h1>
        <p className="text-gray-600">Real-time overview of your LangGraph cluster</p>
      </div>

      {/* Cluster Status Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-2">Cluster Status</h3>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              clusterStatus === 'healthy' ? 'bg-green-500' : 
              clusterStatus === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
            }`}></div>
            <span className="text-xl font-bold capitalize">{clusterStatus}</span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-2">Active Nodes</h3>
          <span className="text-2xl font-bold">{nodes.filter(n => n.status === 'online').length}</span>
          <span className="text-gray-500">/{nodes.length}</span>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-2">Total Services</h3>
          <span className="text-2xl font-bold">
            {nodes.reduce((total, node) => total + node.services.length, 0)}
          </span>
        </div>
      </div>

      {/* Nodes Status */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold">Cluster Nodes</h3>
        </div>
        <div className="p-6">
          {nodes.length === 0 ? (
            <p className="text-gray-500">Loading cluster information...</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {nodes.map((node) => (
                <div key={node.ip} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold">{node.name}</h4>
                    <div className={`w-2 h-2 rounded-full ${
                      node.status === 'online' ? 'bg-green-500' : 
                      node.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}></div>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{node.ip}</p>
                  <div className="text-sm">
                    <span className="font-medium">Services:</span>
                    <div className="mt-1">
                      {node.services.map((service) => (
                        <span key={service} className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded mr-1 mb-1">
                          {service}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
