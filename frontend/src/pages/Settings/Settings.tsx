import React from 'react';

const Settings: React.FC = () => {
  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600 mt-1">Configure your system preferences</p>
      </div>
      
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="text-center py-12">
          <div className="text-4xl mb-4">⚙️</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">System Settings</h3>
          <p className="text-gray-500">Settings configuration coming soon...</p>
        </div>
      </div>
    </div>
  );
};

export default Settings;