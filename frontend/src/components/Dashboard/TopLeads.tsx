import React from 'react';

interface Lead {
  id: string;
  name: string;
  company: string;
  score: number;
  status: string;
  lastContact?: string;
}

interface TopLeadsProps {
  leads?: Lead[];
}

const TopLeads: React.FC<TopLeadsProps> = ({ leads = [] }) => {
  const defaultLeads: Lead[] = [
    {
      id: '1',
      name: 'John Doe',
      company: 'TechCorp Inc',
      score: 95,
      status: 'hot',
      lastContact: '2 hours ago',
    },
    {
      id: '2',
      name: 'Jane Smith',
      company: 'Digital Solutions',
      score: 88,
      status: 'warm',
      lastContact: '1 day ago',
    },
    {
      id: '3',
      name: 'Bob Wilson',
      company: 'StartupXYZ',
      score: 82,
      status: 'warm',
      lastContact: '3 days ago',
    },
    {
      id: '4',
      name: 'Alice Johnson',
      company: 'Enterprise Co',
      score: 76,
      status: 'cold',
      lastContact: '1 week ago',
    },
  ];

  const displayLeads = leads.length > 0 ? leads : defaultLeads;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'hot':
        return 'bg-red-100 text-red-800';
      case 'warm':
        return 'bg-yellow-100 text-yellow-800';
      case 'cold':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Leads</h3>
      <div className="space-y-4">
        {displayLeads.map((lead) => (
          <div key={lead.id} className="flex items-center justify-between p-3 rounded-lg border border-gray-100">
            <div className="flex-1">
              <div className="flex items-center space-x-3">
                <div>
                  <h4 className="font-medium text-gray-900">{lead.name}</h4>
                  <p className="text-sm text-gray-500">{lead.company}</p>
                </div>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(lead.status)}`}>
                  {lead.status}
                </span>
              </div>
              {lead.lastContact && (
                <p className="text-xs text-gray-400 mt-1">Last contact: {lead.lastContact}</p>
              )}
            </div>
            <div className="text-right">
              <div className={`text-lg font-bold ${getScoreColor(lead.score)}`}>
                {lead.score}
              </div>
              <div className="text-xs text-gray-500">Score</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TopLeads;