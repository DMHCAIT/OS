import React from 'react';

interface VoiceAIStatusProps {
  status?: 'online' | 'offline' | 'processing';
  lastActivity?: string;
  callsToday?: number;
  successRate?: number;
  activeCalls?: number;
  avgLeadScore?: number;
}

const VoiceAIStatus: React.FC<VoiceAIStatusProps> = ({
  status = 'online',
  lastActivity = '2 minutes ago',
  callsToday = 24,
  successRate = 78,
  activeCalls = 0,
  avgLeadScore = 0
}) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'bg-green-100 text-green-800';
      case 'offline':
        return 'bg-red-100 text-red-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return '🟢';
      case 'offline':
        return '🔴';
      case 'processing':
        return '🟡';
      default:
        return '⚪';
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Voice AI Status</h3>
        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(status)}`}>
          <span className="mr-2">{getStatusIcon(status)}</span>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>
      
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Last Activity</span>
          <span className="text-sm font-medium text-gray-900">{lastActivity}</span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Calls Today</span>
          <span className="text-sm font-medium text-gray-900">{callsToday}</span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Success Rate</span>
          <span className="text-sm font-medium text-gray-900">{successRate}%</span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Active Calls</span>
          <span className="text-sm font-medium text-gray-900">{activeCalls}</span>
        </div>
        
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Avg Lead Score</span>
          <span className="text-sm font-medium text-gray-900">{avgLeadScore}</span>
        </div>
        
        <div className="mt-4">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">Performance</span>
            <span className="text-gray-900">{successRate}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full" 
              style={{ width: `${successRate}%` }}
            ></div>
          </div>
        </div>
        
        {status === 'online' && (
          <div className="mt-4 p-3 bg-green-50 rounded-lg">
            <div className="flex items-center">
              <span className="text-green-600 mr-2">🤖</span>
              <p className="text-sm text-green-800">AI is actively processing leads and making calls</p>
            </div>
          </div>
        )}
        
        {status === 'offline' && (
          <div className="mt-4 p-3 bg-red-50 rounded-lg">
            <div className="flex items-center">
              <span className="text-red-600 mr-2">⚠️</span>
              <p className="text-sm text-red-800">AI system is currently offline</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceAIStatus;