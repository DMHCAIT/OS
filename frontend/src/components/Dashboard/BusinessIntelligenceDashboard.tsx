import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  FunnelChart,
  Funnel,
  LabelList
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target, 
  Users, 
  Phone,
  Award,
  AlertCircle,
  CheckCircle,
  Clock,
  Lightbulb
} from 'lucide-react';

interface ConversionFunnel {
  stage_counts: Record<string, number>;
  stage_values: Record<string, number>;
  conversion_rates: Record<string, number>;
  total_leads: number;
  won_deals: number;
  overall_conversion_rate: number;
}

interface ROIAnalysis {
  overall_roi_percentage: number;
  total_revenue: number;
  total_cost: number;
  total_conversions: number;
  channel_performance: Record<string, {
    roi_percentage: number;
    conversion_rate: number;
    cost_per_conversion: number;
    total_revenue: number;
    total_cost: number;
  }>;
  best_performing_channel: string;
  most_cost_effective: string;
}

interface PipelineAnalysis {
  average_stage_durations_days: Record<string, number>;
  bottleneck_stage: string;
  bottleneck_duration_days: number;
  recommendations: string[];
  total_pipeline_length_days: number;
  vs_industry_benchmark: {
    current_cycle: number;
    industry_average: number;
    performance_vs_benchmark: string;
  };
}

interface SalesRepRankings {
  rep_rankings: Array<{
    rep_id: string;
    rep_name: string;
    calls_made: number;
    leads_generated: number;
    deals_closed: number;
    total_revenue: number;
    conversion_rate: number;
    avg_deal_size: number;
    performance_score: number;
    rank: number;
  }>;
  team_averages: {
    avg_conversion_rate: number;
    avg_deal_size: number;
    total_team_revenue: number;
    total_team_deals: number;
  };
}

interface PredictiveInsight {
  insight_type: string;
  prediction: string;
  confidence: number;
  impact_score: number;
  recommended_actions: string[];
  expected_roi: number;
  timeframe: string;
}

interface BusinessDashboardData {
  conversion_funnel: ConversionFunnel;
  roi_analysis: ROIAnalysis;
  pipeline_analysis: PipelineAnalysis;
  sales_rep_rankings: SalesRepRankings;
  predictive_insights: PredictiveInsight[];
  business_health_score: number;
  key_metrics: {
    total_revenue_30d: number;
    deals_closed_30d: number;
    conversion_rate_30d: number;
    avg_deal_size_30d: number;
    total_leads_30d: number;
    pipeline_value: number;
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const BusinessIntelligenceDashboard: React.FC = () => {
  const [data, setData] = useState<BusinessDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchDashboardData();
    
    // Set up periodic refresh
    const interval = setInterval(fetchDashboardData, 60000); // Refresh every minute
    
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/dashboard/business/comprehensive');
      const result = await response.json();
      
      if (result.success) {
        setData(result.data);
        setError(null);
      } else {
        setError('Failed to fetch business data');
      }
    } catch (err) {
      setError('Error connecting to business intelligence service');
      console.error('Error fetching business data:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number) => 
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

  const getHealthScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getHealthScoreLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Needs Attention';
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center p-8">
        <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Unable to Load Business Data</h3>
        <p className="text-gray-600 mb-4">{error || 'No data available'}</p>
        <Button onClick={fetchDashboardData}>Retry</Button>
      </div>
    );
  }

  // Prepare funnel data for chart
  const funnelData = [
    { name: 'Leads', value: data.conversion_funnel.stage_counts.lead || 0, fill: COLORS[0] },
    { name: 'Contacted', value: data.conversion_funnel.stage_counts.contacted || 0, fill: COLORS[1] },
    { name: 'Qualified', value: data.conversion_funnel.stage_counts.qualified || 0, fill: COLORS[2] },
    { name: 'Proposal', value: data.conversion_funnel.stage_counts.proposal || 0, fill: COLORS[3] },
    { name: 'Won', value: data.conversion_funnel.stage_counts.closed_won || 0, fill: COLORS[4] }
  ];

  // Prepare channel ROI data
  const channelData = Object.entries(data.roi_analysis.channel_performance).map(([channel, perf]) => ({
    channel,
    roi: perf.roi_percentage,
    revenue: perf.total_revenue,
    cost: perf.total_cost,
    conversion_rate: perf.conversion_rate
  }));

  // Prepare pipeline stage data
  const pipelineStageData = Object.entries(data.pipeline_analysis.average_stage_durations_days).map(([stage, days]) => ({
    stage: stage.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
    days: days,
    isBottleneck: stage === data.pipeline_analysis.bottleneck_stage
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Business Intelligence Dashboard</h1>
          <p className="text-gray-600 mt-1">Sales performance analytics and insights</p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-500">Business Health Score</div>
          <div className={`text-2xl font-bold ${getHealthScoreColor(data.business_health_score)}`}>
            {data.business_health_score.toFixed(0)}/100
          </div>
          <div className={`text-sm ${getHealthScoreColor(data.business_health_score)}`}>
            {getHealthScoreLabel(data.business_health_score)}
          </div>
        </div>
      </div>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Revenue (30d)</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(data.key_metrics.total_revenue_30d)}
            </div>
            <p className="text-xs text-muted-foreground">
              {data.key_metrics.deals_closed_30d} deals closed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Conversion Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatPercentage(data.key_metrics.conversion_rate_30d)}
            </div>
            <p className="text-xs text-muted-foreground">
              {data.key_metrics.total_leads_30d} leads processed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Deal Size</CardTitle>
            <Award className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(data.key_metrics.avg_deal_size_30d)}
            </div>
            <p className="text-xs text-muted-foreground">
              Per closed deal
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pipeline Value</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(data.key_metrics.pipeline_value)}
            </div>
            <p className="text-xs text-muted-foreground">
              Active opportunities
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Predictive Insights */}
      {data.predictive_insights.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="h-5 w-5" />
              AI Predictive Insights
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {data.predictive_insights.map((insight, index) => (
              <div key={index} className="border-l-4 border-blue-500 bg-blue-50 p-4 rounded">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold text-blue-900">
                    {insight.prediction}
                  </h4>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {(insight.confidence * 100).toFixed(0)}% confidence
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {insight.timeframe}
                    </Badge>
                  </div>
                </div>
                <p className="text-sm text-blue-800 mb-2">
                  Expected ROI: {formatCurrency(insight.expected_roi)}
                </p>
                <div className="text-sm">
                  <strong>Recommended Actions:</strong>
                  <ul className="list-disc list-inside mt-1 text-blue-800">
                    {insight.recommended_actions.map((action, actionIndex) => (
                      <li key={actionIndex}>{action}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Main Dashboard Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Sales Funnel</TabsTrigger>
          <TabsTrigger value="roi">ROI Analysis</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="performance">Team Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Conversion Funnel */}
            <Card>
              <CardHeader>
                <CardTitle>Sales Conversion Funnel</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={funnelData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Conversion Rates */}
            <Card>
              <CardHeader>
                <CardTitle>Stage-to-Stage Conversion Rates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(data.conversion_funnel.conversion_rates).map(([transition, rate]) => {
                    const [from, to] = transition.split('_to_');
                    return (
                      <div key={transition} className="flex justify-between items-center">
                        <span className="text-sm">
                          {from.replace('_', ' ')} → {to.replace('_', ' ')}
                        </span>
                        <div className="flex items-center gap-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${rate * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium w-12">
                            {formatPercentage(rate)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="roi" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Channel Performance */}
            <Card>
              <CardHeader>
                <CardTitle>Channel ROI Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={channelData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="channel" />
                    <YAxis />
                    <Tooltip formatter={(value, name) => [
                      name === 'roi' ? `${value.toFixed(1)}%` : formatCurrency(value),
                      name === 'roi' ? 'ROI' : name === 'revenue' ? 'Revenue' : 'Cost'
                    ]} />
                    <Bar dataKey="roi" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* ROI Summary */}
            <Card>
              <CardHeader>
                <CardTitle>ROI Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-green-50 rounded">
                  <span>Overall ROI</span>
                  <span className="text-2xl font-bold text-green-600">
                    {data.roi_analysis.overall_roi_percentage.toFixed(1)}%
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Total Revenue:</span>
                    <span className="font-semibold">{formatCurrency(data.roi_analysis.total_revenue)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Cost:</span>
                    <span className="font-semibold">{formatCurrency(data.roi_analysis.total_cost)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Conversions:</span>
                    <span className="font-semibold">{data.roi_analysis.total_conversions}</span>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-600">Best Performing Channel:</span>
                    <Badge variant="outline">{data.roi_analysis.best_performing_channel}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Most Cost Effective:</span>
                    <Badge variant="outline">{data.roi_analysis.most_cost_effective}</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="pipeline" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Pipeline Stage Duration */}
            <Card>
              <CardHeader>
                <CardTitle>Average Time in Each Stage</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={pipelineStageData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="stage" />
                    <YAxis />
                    <Tooltip formatter={(value) => [`${value.toFixed(1)} days`, 'Duration']} />
                    <Bar 
                      dataKey="days" 
                      fill={(entry: any) => entry.isBottleneck ? "#ff8042" : "#8884d8"}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Pipeline Analysis */}
            <Card>
              <CardHeader>
                <CardTitle>Pipeline Analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-orange-50 border border-orange-200 rounded">
                  <div className="flex items-center gap-2 mb-2">
                    <Clock className="h-4 w-4 text-orange-600" />
                    <span className="font-semibold text-orange-800">Bottleneck Detected</span>
                  </div>
                  <p className="text-sm text-orange-700">
                    <strong>{data.pipeline_analysis.bottleneck_stage.replace('_', ' ')}</strong> stage 
                    taking {data.pipeline_analysis.bottleneck_duration_days.toFixed(1)} days on average
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Total Cycle Length:</span>
                    <span className="font-semibold">
                      {data.pipeline_analysis.total_pipeline_length_days.toFixed(1)} days
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Industry Benchmark:</span>
                    <span className="font-semibold">
                      {data.pipeline_analysis.vs_industry_benchmark.industry_average} days
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Performance:</span>
                    <Badge 
                      variant={data.pipeline_analysis.vs_industry_benchmark.performance_vs_benchmark === 'above' ? 'default' : 'destructive'}
                    >
                      {data.pipeline_analysis.vs_industry_benchmark.performance_vs_benchmark === 'above' 
                        ? 'Above Benchmark' 
                        : 'Below Benchmark'}
                    </Badge>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h4 className="font-semibold mb-2">Recommendations:</h4>
                  <ul className="list-disc list-inside text-sm space-y-1">
                    {data.pipeline_analysis.recommendations.map((rec, index) => (
                      <li key={index} className="text-gray-700">{rec}</li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          {/* Sales Rep Rankings */}
          <Card>
            <CardHeader>
              <CardTitle>Sales Team Performance Rankings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Rank</th>
                      <th className="text-left p-2">Rep Name</th>
                      <th className="text-right p-2">Calls</th>
                      <th className="text-right p-2">Leads</th>
                      <th className="text-right p-2">Deals</th>
                      <th className="text-right p-2">Revenue</th>
                      <th className="text-right p-2">Conv. Rate</th>
                      <th className="text-right p-2">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.sales_rep_rankings.rep_rankings.slice(0, 10).map((rep) => (
                      <tr key={rep.rep_id} className="border-b hover:bg-gray-50">
                        <td className="p-2">
                          <Badge variant={rep.rank <= 3 ? 'default' : 'outline'}>
                            #{rep.rank}
                          </Badge>
                        </td>
                        <td className="p-2 font-medium">{rep.rep_name}</td>
                        <td className="p-2 text-right">{rep.calls_made}</td>
                        <td className="p-2 text-right">{rep.leads_generated}</td>
                        <td className="p-2 text-right">{rep.deals_closed}</td>
                        <td className="p-2 text-right">{formatCurrency(rep.total_revenue)}</td>
                        <td className="p-2 text-right">{formatPercentage(rep.conversion_rate)}</td>
                        <td className="p-2 text-right">
                          <span className={`font-semibold ${
                            rep.performance_score >= 80 ? 'text-green-600' :
                            rep.performance_score >= 60 ? 'text-yellow-600' :
                            'text-red-600'
                          }`}>
                            {rep.performance_score.toFixed(0)}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Team Averages */}
              <div className="mt-6 pt-4 border-t">
                <h4 className="font-semibold mb-4">Team Averages</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {formatPercentage(data.sales_rep_rankings.team_averages.avg_conversion_rate)}
                    </div>
                    <div className="text-sm text-gray-600">Avg Conversion Rate</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {formatCurrency(data.sales_rep_rankings.team_averages.avg_deal_size)}
                    </div>
                    <div className="text-sm text-gray-600">Avg Deal Size</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {formatCurrency(data.sales_rep_rankings.team_averages.total_team_revenue)}
                    </div>
                    <div className="text-sm text-gray-600">Total Revenue</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {data.sales_rep_rankings.team_averages.total_team_deals}
                    </div>
                    <div className="text-sm text-gray-600">Total Deals</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BusinessIntelligenceDashboard;