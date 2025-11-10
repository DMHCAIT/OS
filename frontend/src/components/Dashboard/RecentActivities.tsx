import React from 'react';

interface Activity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  user?: string;
}

interface RecentActivitiesProps {
  activities?: Activity[];
}

const RecentActivities: React.FC<RecentActivitiesProps> = ({ activities = [] }) => {
  const defaultActivities: Activity[] = [
    {
      id: '1',
      type: 'call',
      description: 'Called John Doe about product demo',
      timestamp: '2 minutes ago',
      user: 'Sarah Johnson',
    },
    {
      id: '2',
      type: 'lead',
      description: 'New lead added: Jane Smith',
      timestamp: '5 minutes ago',
      user: 'AI System',
    },
    {
      id: '3',
      type: 'analysis',
      description: 'Lead score updated for ABC Corp',
      timestamp: '10 minutes ago',
      user: 'AI System',
    },
    {
      id: '4',
      type: 'call',
      description: 'Successful call with TechStart Inc',
      timestamp: '15 minutes ago',
      user: 'Mike Wilson',
    },
  ];

  const displayActivities = activities.length > 0 ? activities : defaultActivities;

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'call':
        return '📞';
      case 'lead':
        return '👤';
      case 'analysis':
        return '🔍';
      default:
        return '📝';
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activities</h3>
      <div className="space-y-4">
        {displayActivities.map((activity) => (
          <div key={activity.id} className="flex items-start space-x-3">
            <div className="text-2xl">{getActivityIcon(activity.type)}</div>
            <div className="flex-1">
              <p className="text-sm text-gray-900">{activity.description}</p>
              <div className="flex items-center space-x-2 mt-1">
                <p className="text-xs text-gray-500">{activity.timestamp}</p>
                {activity.user && (
                  <>
                    <span className="text-xs text-gray-300">•</span>
                    <p className="text-xs text-gray-500">{activity.user}</p>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RecentActivities;