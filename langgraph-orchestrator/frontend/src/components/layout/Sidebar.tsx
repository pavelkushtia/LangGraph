import React from 'react';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onToggle }) => {
  const menuItems = [
    { name: 'Dashboard', path: '/dashboard', icon: '📊' },
    { name: 'Workflow Designer', path: '/workflow-designer', icon: '🎨' },
    { name: 'Operations', path: '/operations', icon: '⚡' },
    { name: 'Cluster Health', path: '/cluster-health', icon: '🏥' },
    { name: 'Templates', path: '/templates', icon: '📋' },
    { name: 'Analytics', path: '/analytics', icon: '📈' },
    { name: 'Settings', path: '/settings', icon: '⚙️' },
    { name: 'Logs', path: '/logs', icon: '📝' },
  ];

  return (
    <div className={`fixed left-0 top-0 h-full bg-gray-800 text-white transition-transform duration-300 z-50 ${
      isOpen ? 'translate-x-0' : '-translate-x-full'
    } w-64 md:translate-x-0 md:static md:z-auto`}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-xl font-bold">LangGraph Orchestrator</h1>
          <button 
            onClick={onToggle}
            className="md:hidden text-gray-400 hover:text-white"
          >
            ✕
          </button>
        </div>
        
        <nav>
          <ul className="space-y-2">
            {menuItems.map((item) => (
              <li key={item.path}>
                <a
                  href={`#${item.path}`}
                  className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <span>{item.icon}</span>
                  <span>{item.name}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </div>
  );
};

export default Sidebar;
