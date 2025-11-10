import React, { useState, useEffect } from 'react';
import { 
  ChartBarIcon,
  PhoneIcon,
  UserGroupIcon,
  CurrencyDollarIcon,
  ArrowTrendingUpIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

// Components
import MetricCard from '../../components/Dashboard/MetricCard';
import LeadScoreChart from '../../components/Dashboard/LeadScoreChart';
import CallActivityChart from '../../components/Dashboard/CallActivityChart';
import RecentActivities from '../../components/Dashboard/RecentActivities';
import TopLeads from '../../components/Dashboard/TopLeads';
import VoiceAIStatus from '../../components/Dashboard/VoiceAIStatus';

// Services
import { getDashboardMetrics, getRecentActivities } from '../../services/dashboard';

// Types
interface DashboardMetrics {
  totalLeads: number;
  newLeadsToday: number;
  totalCalls: number;
  callsToday: number;
  conversionRate: number;
  avgLeadScore: number;
  revenue: number;
  voiceAICallsActive: number;
}

interface Activity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  leadName?: string;
  leadId?: string;
}

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [metricsData, activitiesData] = await Promise.all([
        getDashboardMetrics(),
        getRecentActivities()
      ]);
      
      setMetrics(metricsData);
      setActivities(activitiesData);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-red-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">{error}</h3>
          <button
            onClick={fetchDashboardData}
            className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Page header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900">
              AI Lead Management Dashboard
            </h1>
            <p className="mt-2 text-gray-600">
              Real-time insights into your lead management and voice AI performance
            </p>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4 mb-8">
            <MetricCard
              title="Total Leads"
              value={metrics?.totalLeads || 0}
              change={`+${metrics?.newLeadsToday || 0} today`}
              icon={<UserGroupIcon className="h-6 w-6" />}
              color="blue"
            />
            <MetricCard
              title="Voice AI Calls"
              value={metrics?.totalCalls || 0}
              change={`${metrics?.callsToday || 0} today`}
              icon={<PhoneIcon className="h-6 w-6" />}
              color="green"
            />
            <MetricCard
              title="Conversion Rate"
              value={`${metrics?.conversionRate || 0}%`}
              change="+2.1% from last month"
              icon={<ArrowTrendingUpIcon className="h-6 w-6" />}
              color="purple"
            />
            <MetricCard
              title="Revenue Generated"
              value={`$${(metrics?.revenue || 0).toLocaleString()}`}
              change="+12.5% from last month"
              icon={<CurrencyDollarIcon className="h-6 w-6" />}
              color="indigo"
            />
          </div>

          {/* Voice AI Status */}
          <div className="mb-8">
            <VoiceAIStatus 
              activeCalls={metrics?.voiceAICallsActive || 0}
              avgLeadScore={metrics?.avgLeadScore || 0}
            />
          </div>

          {/* Charts and Analytics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Lead Score Distribution
              </h2>
              <LeadScoreChart />
            </div>
            
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Call Activity (Last 7 Days)
              </h2>
              <CallActivityChart />
            </div>
          </div>

          {/* Recent Activities and Top Leads */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">
                  Recent Activities
                </h2>
              </div>
              <RecentActivities activities={activities} />
            </div>
            
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">
                  Top Scoring Leads
                </h2>
              </div>
              <TopLeads />
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Quick Actions
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <button
                onClick={() => window.location.href = '/leads?status=new'}
                className="flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 transition-colors"
              >
                <UserGroupIcon className="h-5 w-5 mr-2" />
                Review New Leads
              </button>
              
              <button
                onClick={() => window.location.href = '/voice-ai'}
                className="flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-green-600 hover:bg-green-700 transition-colors"
              >
                <PhoneIcon className="h-5 w-5 mr-2" />
                Start Voice Campaign
              </button>
              
              <button
                onClick={() => window.location.href = '/analytics'}
                className="flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 transition-colors"
              >
                <ChartBarIcon className="h-5 w-5 mr-2" />
                View Analytics
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;