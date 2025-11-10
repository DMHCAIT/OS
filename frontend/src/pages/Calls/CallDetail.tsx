import React from 'react';
import { useParams } from 'react-router-dom';

const CallDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Call Detail</h1>
        <p className="text-gray-600 mt-1">Call ID: {id}</p>
      </div>
      
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="text-center py-12">
          <div className="text-4xl mb-4">📊</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Call Detail View</h3>
          <p className="text-gray-500">Call detail view coming soon...</p>
        </div>
      </div>
    </div>
  );
};

export default CallDetail;