import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useSocket } from './hooks/useSocket';
import { useTheme } from './hooks/useTheme';

// Layout components
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import LoadingSpinner from './components/ui/LoadingSpinner';

// Page components
import Dashboard from './pages/Dashboard';
import WorkflowDesigner from './pages/WorkflowDesigner';
import Operations from './pages/Operations';
import ClusterHealth from './pages/ClusterHealth';
import Templates from './pages/Templates';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import Logs from './pages/Logs';

// Types
interface AppState {
  loading: boolean;
  error: string | null;
  sidebarOpen: boolean;
}

function App() {
  const { isDark } = useTheme();
  const { isConnected, connectionStatus } = useSocket();
  
  const [state, setState] = useState<AppState>({
    loading: true,
    error: null,
    sidebarOpen: true,
  });

  // Initialize app
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Simulate initialization delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setState(prev => ({ ...prev, loading: false }));
      } catch (error) {
        setState(prev => ({
          ...prev,
          loading: false,
          error: error instanceof Error ? error.message : 'Failed to initialize app'
        }));
      }
    };

    initializeApp();
  }, []);

  // Toggle sidebar
  const toggleSidebar = () => {
    setState(prev => ({ ...prev, sidebarOpen: !prev.sidebarOpen }));
  };

  // Loading state
  if (state.loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <h2 className="mt-4 text-lg font-medium text-gray-900">
            Loading LangGraph Orchestrator...
          </h2>
          <p className="mt-2 text-gray-600">
            Initializing workspace and connecting to cluster
          </p>
        </div>
      </div>
    );
  }

  // Error state
  if (state.error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-error-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-error-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h1 className="text-xl font-semibold text-gray-900 mb-2">
            Initialization Failed
          </h1>
          <p className="text-gray-600 mb-6">{state.error}</p>
          <button
            onClick={() => window.location.reload()}
            className="btn-primary w-full"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-gray-50 ${isDark ? 'dark' : ''}`}>
      {/* Connection status indicator */}
      {!isConnected && (
        <div className="bg-warning-100 border-b border-warning-200 px-4 py-2">
          <div className="flex items-center justify-center">
            <svg className="w-4 h-4 text-warning-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <span className="text-sm text-warning-800">
              Connection lost - Attempting to reconnect... ({connectionStatus})
            </span>
          </div>
        </div>
      )}

      {/* Main layout */}
      <div className="flex h-screen">
        {/* Sidebar */}
        <Sidebar isOpen={state.sidebarOpen} onToggle={toggleSidebar} />

        {/* Main content area */}
        <div className={`flex-1 flex flex-col ${state.sidebarOpen ? 'ml-64' : 'ml-16'} transition-all duration-300`}>
          {/* Header */}
          <Header onToggleSidebar={toggleSidebar} />

          {/* Page content */}
          <main className="flex-1 overflow-hidden">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/designer" element={<WorkflowDesigner />} />
              <Route path="/operations" element={<Operations />} />
              <Route path="/cluster" element={<ClusterHealth />} />
              <Route path="/templates" element={<Templates />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
