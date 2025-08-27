import React from 'react';

const Analytics: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600">Performance metrics and usage analytics</p>
      </div>

      <div className="bg-white rounded-lg shadow p-8 text-center">
        <div className="mb-4">
          <div className="w-16 h-16 mx-auto bg-green-100 rounded-full flex items-center justify-center">
            <span className="text-2xl">📈</span>
          </div>
        </div>
        <h3 className="text-lg font-semibold mb-2">Analytics Dashboard</h3>
        <p className="text-gray-600">
          Detailed analytics and charts coming soon!
        </p>
      </div>
    </div>
  );
};

export default Analytics;
