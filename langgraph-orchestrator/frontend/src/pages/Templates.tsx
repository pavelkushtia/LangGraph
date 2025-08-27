import React from 'react';

const Templates: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Workflow Templates</h1>
        <p className="text-gray-600">Pre-built workflow templates for common use cases</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          { name: 'Research Assistant', icon: '🔍', description: 'Query analysis → Web search → Content analysis' },
          { name: 'Document Q&A', icon: '📄', description: 'Document processing → Question answering' },
          { name: 'Content Creation', icon: '✍️', description: 'Research → Outline → Writing → Optimization' }
        ].map((template) => (
          <div key={template.name} className="bg-white rounded-lg shadow p-6">
            <div className="text-3xl mb-3">{template.icon}</div>
            <h3 className="text-lg font-semibold mb-2">{template.name}</h3>
            <p className="text-gray-600 text-sm mb-4">{template.description}</p>
            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors">
              Use Template
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Templates;
