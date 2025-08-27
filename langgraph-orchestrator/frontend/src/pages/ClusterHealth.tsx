import React from 'react';

const ClusterHealth: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Cluster Health</h1>
        <p className="text-gray-600">Monitor the health of all cluster nodes and services</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Nodes Health */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold">Node Health</h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {[
                { name: 'Jetson Orin Nano', ip: '192.168.1.177', status: 'healthy', services: ['ollama'] },
                { name: 'CPU Coordinator', ip: '192.168.1.81', status: 'healthy', services: ['ollama', 'haproxy', 'redis'] },
                { name: 'RPi Embeddings', ip: '192.168.1.178', status: 'healthy', services: ['embeddings-server'] },
                { name: 'Worker Tools', ip: '192.168.1.190', status: 'healthy', services: ['tools-server'] },
                { name: 'Worker Monitor', ip: '192.168.1.191', status: 'healthy', services: ['monitoring-server'] }
              ].map((node) => (
                <div key={node.ip} className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-semibold">{node.name}</h4>
                    <p className="text-sm text-gray-600">{node.ip}</p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-sm font-medium text-green-600">Healthy</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Services Health */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold">Service Health</h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {[
                { name: 'Jetson Ollama', endpoint: '192.168.1.177:11434', status: 'running' },
                { name: 'CPU Ollama', endpoint: '192.168.1.81:11435', status: 'running' },
                { name: 'HAProxy Load Balancer', endpoint: '192.168.1.81:9000', status: 'running' },
                { name: 'Redis Cache', endpoint: '192.168.1.81:6379', status: 'running' },
                { name: 'Embeddings Server', endpoint: '192.168.1.178:8081', status: 'running' },
                { name: 'Tools Server', endpoint: '192.168.1.190:8082', status: 'running' },
                { name: 'Monitoring Server', endpoint: '192.168.1.191:8083', status: 'running' }
              ].map((service) => (
                <div key={service.endpoint} className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-semibold">{service.name}</h4>
                    <p className="text-sm text-gray-600">{service.endpoint}</p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span className="text-sm font-medium text-green-600">Running</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClusterHealth;
