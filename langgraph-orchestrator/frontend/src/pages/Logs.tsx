import React from 'react';

const Logs: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">System Logs</h1>
        <p className="text-gray-600">View cluster and application logs</p>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold">Recent Logs</h3>
        </div>
        <div className="p-6">
          <div className="bg-black text-green-400 font-mono text-sm p-4 rounded-lg h-96 overflow-y-auto">
            <div>2025-08-26 20:30:15 [INFO] Cluster monitoring started</div>
            <div>2025-08-26 20:30:16 [INFO] All services healthy</div>
            <div>2025-08-26 20:30:17 [INFO] WebSocket connections active</div>
            <div>2025-08-26 20:30:18 [INFO] Database connection established</div>
            <div className="text-yellow-400">2025-08-26 20:30:19 [WARN] High CPU usage on worker-node3</div>
            <div>2025-08-26 20:30:20 [INFO] Backup completed successfully</div>
            <div className="cursor">█</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Logs;
