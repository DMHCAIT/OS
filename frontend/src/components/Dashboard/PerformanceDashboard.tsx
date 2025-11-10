import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
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
  Bar
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  Activity, 
  Cpu, 
  Database,
  Users,
  Phone,
  Brain,
  Zap,
  DollarSign
} from 'lucide-react';

interface PerformanceMetrics {
  timestamp: string;
  system_health: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    active_connections: number;
    response_time_avg: number;
    error_rate: number;
    ml_accuracy: number;
    call_success_rate: number;
  };
  api_metrics: {
    avg_response_time: number;
    error_rate: number;
    active_users: number;
  };
  ml_metrics: {
    avg_accuracy: number;
    models_performance: Record<string, number>;
  };
  call_metrics: {
    success_rate: number;
    concurrent_calls: number;
  };
}

interface AlertData {
  id: string;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  category: string;
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  resolved: boolean;
}

const PerformanceDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Set up real-time data streaming
    const eventSource = new EventSource('http://localhost:8000/api/dashboard/stream/performance');
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'performance') {
          setMetrics(data.data);
          setLoading(false);
        }
      } catch (err) {
        console.error('Error parsing performance data:', err);
      }
    };

    eventSource.onerror = (error) => {
      console.error('Performance stream error:', error);
      setError('Failed to connect to real-time performance stream');
    };

    // Set up alert streaming
    const alertEventSource = new EventSource('http://localhost:8000/api/dashboard/stream/alerts');
    
    alertEventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'new_alerts') {
          setAlerts(prev => [...data.data, ...prev].slice(0, 10)); // Keep last 10 alerts
        }
      } catch (err) {
        console.error('Error parsing alert data:', err);
      }
    };

    // Fetch initial data
    fetchInitialData();

    return () => {
      eventSource.close();
      alertEventSource.close();
    };
  }, []);

  const fetchInitialData = async () => {
    try {
      // Fetch current metrics
      const metricsResponse = await fetch('http://localhost:8000/api/dashboard/performance/current');
      const metricsData = await metricsResponse.json();
      if (metricsData.success) {
        setMetrics(metricsData.data);
      }

      // Fetch alerts
      const alertsResponse = await fetch('http://localhost:8000/api/dashboard/alerts/active');
      const alertsData = await alertsResponse.json();
      if (alertsData.success) {
        setAlerts(alertsData.data);
      }

      // Fetch historical data for charts
      const historyResponse = await fetch('http://localhost:8000/api/dashboard/performance/history?hours=6');
      const historyData = await historyResponse.json();
      if (historyData.success) {
        setHistoricalData(historyData.data.system_health_history || []);
      }

      setLoading(false);
    } catch (err) {
      setError('Failed to fetch initial dashboard data');
      setLoading(false);
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await fetch('http://localhost:8000/api/dashboard/alerts/acknowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_id: alertId, user_id: 'current_user' })
      });
      
      setAlerts(prev => 
        prev.map(alert => 
          alert.id === alertId ? { ...alert, acknowledged: true } : alert
        )
      );
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  const resolveAlert = async (alertId: string) => {
    try {
      await fetch('http://localhost:8000/api/dashboard/alerts/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          alert_id: alertId, 
          user_id: 'current_user',
          resolution_note: 'Resolved from dashboard'
        })
      });
      
      setAlerts(prev => prev.filter(alert => alert.id !== alertId));
    } catch (err) {
      console.error('Failed to resolve alert:', err);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'emergency':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'info':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatDuration = (ms: number) => `${ms.toFixed(0)}ms`;

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="border-red-200 bg-red-50">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!metrics) {
    return (
      <Alert className="border-yellow-200 bg-yellow-50">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>No Data</AlertTitle>
        <AlertDescription>No performance data available</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Performance Dashboard</h1>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Activity className="h-4 w-4" />
          Live Updates
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
        </div>
      </div>

      {/* Alert Panel */}
      {alerts.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Active Alerts</h2>
          {alerts.map((alert) => (
            <Alert 
              key={alert.id} 
              className={`${getSeverityColor(alert.severity)} ${alert.acknowledged ? 'opacity-75' : ''}`}
            >
              <AlertTriangle className="h-4 w-4" />
              <div className="flex justify-between items-start w-full">
                <div className="flex-1">
                  <AlertTitle className="flex items-center gap-2">
                    {alert.title}
                    <Badge variant="outline" className="text-xs">
                      {alert.severity.toUpperCase()}
                    </Badge>
                  </AlertTitle>
                  <AlertDescription className="mt-1">
                    {alert.message}
                  </AlertDescription>
                  <div className="text-xs text-gray-500 mt-2">
                    {new Date(alert.timestamp).toLocaleString()}
                  </div>
                </div>
                <div className="flex gap-2">
                  {!alert.acknowledged && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => acknowledgeAlert(alert.id)}
                    >
                      Acknowledge
                    </Button>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => resolveAlert(alert.id)}
                  >
                    Resolve
                  </Button>
                </div>
              </div>
            </Alert>
          ))}
        </div>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* System Health */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Health</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.system_health ? formatPercentage(1 - Math.max(
                metrics.system_health.cpu_usage,
                metrics.system_health.memory_usage
              )) : 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">
              CPU: {metrics.system_health ? formatPercentage(metrics.system_health.cpu_usage) : 'N/A'} | 
              Memory: {metrics.system_health ? formatPercentage(metrics.system_health.memory_usage) : 'N/A'}
            </p>
          </CardContent>
        </Card>

        {/* API Performance */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Response Time</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatDuration(metrics.api_metrics.avg_response_time)}
            </div>
            <p className="text-xs text-muted-foreground flex items-center">
              Error Rate: {formatPercentage(metrics.api_metrics.error_rate)}
              {metrics.api_metrics.error_rate < 0.05 ? (
                <TrendingUp className="h-3 w-3 ml-1 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 ml-1 text-red-500" />
              )}
            </p>
          </CardContent>
        </Card>

        {/* ML Performance */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ML Accuracy</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatPercentage(metrics.ml_metrics.avg_accuracy)}
            </div>
            <p className="text-xs text-muted-foreground">
              Models: {Object.keys(metrics.ml_metrics.models_performance).length}
            </p>
          </CardContent>
        </Card>

        {/* Call Success Rate */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Call Success Rate</CardTitle>
            <Phone className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatPercentage(metrics.call_metrics.success_rate)}
            </div>
            <p className="text-xs text-muted-foreground">
              Active Calls: {metrics.call_metrics.concurrent_calls}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Health Over Time */}
        <Card>
          <CardHeader>
            <CardTitle>System Health Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={historicalData.slice(-20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis domain={[0, 1]} tickFormatter={formatPercentage} />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                  formatter={(value: number, name: string) => [formatPercentage(value), name]}
                />
                <Line 
                  type="monotone" 
                  dataKey="cpu_usage" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  name="CPU Usage"
                />
                <Line 
                  type="monotone" 
                  dataKey="memory_usage" 
                  stroke="#82ca9d" 
                  strokeWidth={2}
                  name="Memory Usage"
                />
                <Line 
                  type="monotone" 
                  dataKey="ml_accuracy" 
                  stroke="#ffc658" 
                  strokeWidth={2}
                  name="ML Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* ML Model Performance */}
        <Card>
          <CardHeader>
            <CardTitle>ML Model Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={Object.entries(metrics.ml_metrics.models_performance).map(([name, accuracy]) => ({
                name: name.replace('_', ' '),
                accuracy
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} tickFormatter={formatPercentage} />
                <Tooltip formatter={(value: number) => [formatPercentage(value), 'Accuracy']} />
                <Bar dataKey="accuracy" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Response Time Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Response Time & Error Rate Trend</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={historicalData.slice(-20)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis yAxisId="left" orientation="left" />
              <YAxis yAxisId="right" orientation="right" tickFormatter={formatPercentage} />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleString()}
                formatter={(value: number, name: string) => {
                  if (name === 'Response Time') {
                    return [formatDuration(value), name];
                  }
                  return [formatPercentage(value), name];
                }}
              />
              <Area 
                yAxisId="left"
                type="monotone" 
                dataKey="response_time_avg" 
                stroke="#8884d8" 
                fill="#8884d8" 
                fillOpacity={0.3}
                name="Response Time"
              />
              <Area 
                yAxisId="right"
                type="monotone" 
                dataKey="error_rate" 
                stroke="#ff7c7c" 
                fill="#ff7c7c" 
                fillOpacity={0.3}
                name="Error Rate"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Active Users */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Active Connections</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-2">
              {metrics.system_health?.active_connections || 0}
            </div>
            <p className="text-sm text-muted-foreground">
              Current active API connections
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Database Performance</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-2">
              {metrics.system_health ? formatPercentage(1 - metrics.system_health.disk_usage) : 'N/A'}
            </div>
            <p className="text-sm text-muted-foreground">
              Available disk space
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PerformanceDashboard;