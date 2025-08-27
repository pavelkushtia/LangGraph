import React from 'react';

const WorkflowDesigner: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Workflow Designer</h1>
        <p className="text-gray-600">Visual workflow creation and editing</p>
      </div>

      <div className="bg-white rounded-lg shadow p-8 text-center">
        <div className="mb-4">
          <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
            <span className="text-2xl">🎨</span>
          </div>
        </div>
        <h3 className="text-lg font-semibold mb-2">Workflow Designer</h3>
        <p className="text-gray-600 mb-4">
          Visual workflow designer with drag-and-drop nodes coming soon!
        </p>
        <p className="text-sm text-gray-500">
          This will include React Flow integration for creating complex AI workflows
        </p>
      </div>
    </div>
  );
};

export default WorkflowDesigner;
