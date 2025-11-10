import React from 'react';
import { useParams } from 'react-router-dom';

const LeadDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Lead Detail</h1>
        <p className="text-gray-600 mt-1">Lead ID: {id}</p>
      </div>
      
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="text-center py-12">
          <div className="text-4xl mb-4">📋</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Lead Detail View</h3>
          <p className="text-gray-500">Lead detail view coming soon...</p>
        </div>
      </div>
    </div>
  );
};

export default LeadDetail;