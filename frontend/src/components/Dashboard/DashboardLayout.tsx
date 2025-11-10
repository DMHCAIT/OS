import React, { useState } from 'react';
import PerformanceDashboard from './PerformanceDashboard';
import BusinessIntelligenceDashboard from './BusinessIntelligenceDashboard';

interface DashboardLayoutProps {}

const DashboardLayout: React.FC<DashboardLayoutProps> = () => {
  const [activeView, setActiveView] = useState<'performance' | 'business'>('performance');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">AI Sales Analytics Dashboard</h1>
              <p className="text-gray-600">Real-time performance monitoring and business intelligence</p>
            </div>
            
            {/* View Toggle */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setActiveView('performance')}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  activeView === 'performance'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                System Performance
              </button>
              <button
                onClick={() => setActiveView('business')}
                className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  activeView === 'business'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Business Intelligence
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeView === 'performance' && <PerformanceDashboard />}
        {activeView === 'business' && <BusinessIntelligenceDashboard />}
      </div>
    </div>
  );
};

export default DashboardLayout;