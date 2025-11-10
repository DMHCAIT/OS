// Dashboard service functions

export interface DashboardMetrics {
  totalLeads: number;
  newLeadsToday: number;
  totalCalls: number;
  callsToday: number;
  successfulCalls: number;
  conversionRate: number;
  avgLeadScore: number;
  revenue: number;
  voiceAICallsActive: number;
  aiProcessingTime: number;
}

export interface Activity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  user?: string;
}

export const getDashboardMetrics = async (): Promise<DashboardMetrics> => {
  // Mock data - replace with actual API call
  return {
    totalLeads: 1247,
    newLeadsToday: 23,
    totalCalls: 89,
    callsToday: 15,
    successfulCalls: 67,
    conversionRate: 24.5,
    avgLeadScore: 78,
    revenue: 45780,
    voiceAICallsActive: 3,
    aiProcessingTime: 1.2,
  };
};

export const getRecentActivities = async (): Promise<Activity[]> => {
  // Mock data - replace with actual API call
  return [
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
};