import React, { useState, useEffect, useCallback } from 'react';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Brain,
  MessageCircle,
  Globe,
  Heart,
  TrendingUp,
  AlertTriangle,
  Users,
  Zap,
  Target,
  Shield,
  Mic,
  Languages,
  Activity
} from 'lucide-react';

interface ConversationAnalysis {
  conversation_id: string;
  participant_id: string;
  message: string;
  analysis_timestamp: string;
  language_analysis: any;
  emotion_analysis: any;
  behavioral_insights: any;
  script_adaptation: any;
  competitive_analysis: any;
  priority_actions: string[];
  conversation_health_score: number;
  engagement_forecast: string;
  adaptive_strategy: string;
}

interface ConversationInsights {
  current_state: {
    health_score: number;
    engagement_forecast: string;
    adaptive_strategy: string;
    priority_actions: string[];
  };
  participant_profile: {
    language: string;
    cultural_background: any;
    emotional_state: string;
    engagement_trend: number;
  };
  competitive_context: any;
  conversation_flow: {
    current_stage: string;
    message_count: number;
    timeline_length: number;
  };
}

interface ActiveConversation {
  conversation_id: string;
  participant_count: number;
  message_count: number;
  current_stage: string;
  engagement_trend: number;
  health_score: number;
  last_activity: string;
  competitive_active: boolean;
  language: string;
}

const ConversationIntelligenceDashboard: React.FC = () => {
  const [activeConversations, setActiveConversations] = useState<ActiveConversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [conversationInsights, setConversationInsights] = useState<ConversationInsights | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<any>(null);
  const [realTimeAnalysis, setRealTimeAnalysis] = useState<ConversationAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch active conversations
  const fetchActiveConversations = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/conversation-intelligence/active-conversations');
      const data = await response.json();
      if (data.success) {
        setActiveConversations(data.conversations);
      }
    } catch (error) {
      console.error('Error fetching active conversations:', error);
    }
  }, []);

  // Fetch conversation insights
  const fetchConversationInsights = useCallback(async (conversationId: string) => {
    try {
      setIsLoading(true);
      const response = await fetch(`/api/v1/conversation-intelligence/insights/${conversationId}`);
      const data = await response.json();
      if (data.success) {
        setConversationInsights(data.insights);
      }
    } catch (error) {
      console.error('Error fetching conversation insights:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch system performance metrics
  const fetchSystemMetrics = useCallback(async () => {
    try {
      const response = await fetch('/api/v1/conversation-intelligence/performance-metrics');
      const data = await response.json();
      if (data.success) {
        setSystemMetrics(data.analytics);
      }
    } catch (error) {
      console.error('Error fetching system metrics:', error);
    }
  }, []);

  // Initialize data
  useEffect(() => {
    fetchActiveConversations();
    fetchSystemMetrics();
  }, [fetchActiveConversations, fetchSystemMetrics]);

  // Auto-refresh active conversations
  useEffect(() => {
    const interval = setInterval(fetchActiveConversations, 5000);
    return () => clearInterval(interval);
  }, [fetchActiveConversations]);

  // Fetch insights when conversation is selected
  useEffect(() => {
    if (selectedConversation) {
      fetchConversationInsights(selectedConversation);
    }
  }, [selectedConversation, fetchConversationInsights]);

  const getHealthScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getEngagementIcon = (trend: number) => {
    if (trend > 0.7) return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (trend > 0.4) return <Activity className="h-4 w-4 text-yellow-600" />;
    return <AlertTriangle className="h-4 w-4 text-red-600" />;
  };

  const getEmotionalStateColor = (state: string) => {
    const colors = {
      happy: 'bg-green-100 text-green-800',
      excited: 'bg-blue-100 text-blue-800',
      neutral: 'bg-gray-100 text-gray-800',
      concerned: 'bg-yellow-100 text-yellow-800',
      frustrated: 'bg-red-100 text-red-800',
      confused: 'bg-purple-100 text-purple-800'
    };
    return colors[state as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="h-8 w-8 text-blue-600" />
            Conversation Intelligence
          </h1>
          <p className="text-gray-600">Advanced AI-powered conversation analysis and insights</p>
        </div>
        
        <div className="flex gap-2">
          <Button onClick={fetchActiveConversations} variant="outline">
            Refresh
          </Button>
          <Button onClick={fetchSystemMetrics} variant="outline">
            <Activity className="h-4 w-4 mr-2" />
            System Health
          </Button>
        </div>
      </div>

      {/* System Overview Cards */}
      {systemMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemMetrics.usage_statistics.active_sessions}</div>
              <p className="text-xs text-muted-foreground">
                Avg {Math.round(systemMetrics.usage_statistics.average_session_length)} msgs/session
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Multilingual</CardTitle>
              <Languages className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemMetrics.feature_utilization.multilingual_conversations}</div>
              <p className="text-xs text-muted-foreground">Non-English conversations</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Competitive Alerts</CardTitle>
              <Shield className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemMetrics.feature_utilization.competitive_situations_active}</div>
              <p className="text-xs text-muted-foreground">Active competitive situations</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">High Adaptation</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemMetrics.feature_utilization.high_adaptation_conversations}</div>
              <p className="text-xs text-muted-foreground">Dynamic script adaptations</p>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Conversations List */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageCircle className="h-5 w-5" />
              Active Conversations
            </CardTitle>
            <CardDescription>
              {activeConversations.length} live conversations
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {activeConversations.map((conversation) => (
              <div
                key={conversation.conversation_id}
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedConversation === conversation.conversation_id
                    ? 'border-blue-500 bg-blue-50'
                    : 'hover:bg-gray-50'
                }`}
                onClick={() => setSelectedConversation(conversation.conversation_id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-sm">
                    {conversation.conversation_id.substring(0, 8)}...
                  </span>
                  <div className="flex gap-1">
                    {conversation.competitive_active && (
                      <Badge variant="destructive" className="text-xs">
                        Competitive
                      </Badge>
                    )}
                    {conversation.language !== 'en' && (
                      <Badge variant="outline" className="text-xs">
                        {conversation.language.toUpperCase()}
                      </Badge>
                    )}
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-gray-600">
                  <span>{conversation.message_count} messages</span>
                  {getEngagementIcon(conversation.engagement_trend)}
                </div>

                <div className="mt-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span>Health Score</span>
                    <span className={getHealthScoreColor(conversation.health_score)}>
                      {Math.round(conversation.health_score * 100)}%
                    </span>
                  </div>
                  <Progress 
                    value={conversation.health_score * 100} 
                    className="h-1"
                  />
                </div>
              </div>
            ))}

            {activeConversations.length === 0 && (
              <div className="text-center py-6 text-gray-500">
                <MessageCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No active conversations</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Detailed Insights */}
        <div className="lg:col-span-2">
          {selectedConversation && conversationInsights ? (
            <Tabs defaultValue="overview" className="space-y-4">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="participant">Participant</TabsTrigger>
                <TabsTrigger value="competitive">Competitive</TabsTrigger>
                <TabsTrigger value="actions">Actions</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Conversation Health
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Health Score</span>
                          <span className={`px-2 py-1 rounded text-sm ${getHealthScoreColor(conversationInsights.current_state.health_score)}`}>
                            {Math.round(conversationInsights.current_state.health_score * 100)}%
                          </span>
                        </div>
                        <Progress 
                          value={conversationInsights.current_state.health_score * 100} 
                          className="h-2"
                        />
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <span className="text-sm font-medium text-gray-600">Engagement Forecast</span>
                          <p className="text-sm mt-1">{conversationInsights.current_state.engagement_forecast}</p>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-gray-600">Adaptive Strategy</span>
                          <p className="text-sm mt-1">{conversationInsights.current_state.adaptive_strategy}</p>
                        </div>
                      </div>

                      <div>
                        <span className="text-sm font-medium text-gray-600 mb-2 block">Conversation Flow</span>
                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div className="font-medium">{conversationInsights.conversation_flow.message_count}</div>
                            <div className="text-xs text-gray-600">Messages</div>
                          </div>
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div className="font-medium">{conversationInsights.conversation_flow.current_stage}</div>
                            <div className="text-xs text-gray-600">Stage</div>
                          </div>
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div className="font-medium">{conversationInsights.conversation_flow.timeline_length}</div>
                            <div className="text-xs text-gray-600">Analyses</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="participant" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Users className="h-5 w-5" />
                      Participant Profile
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <span className="text-sm font-medium text-gray-600">Language</span>
                          <div className="flex items-center gap-2 mt-1">
                            <Globe className="h-4 w-4" />
                            <span className="text-sm">{conversationInsights.participant_profile.language.toUpperCase()}</span>
                          </div>
                        </div>
                        <div>
                          <span className="text-sm font-medium text-gray-600">Emotional State</span>
                          <div className="flex items-center gap-2 mt-1">
                            <Heart className="h-4 w-4" />
                            <Badge className={getEmotionalStateColor(conversationInsights.participant_profile.emotional_state)}>
                              {conversationInsights.participant_profile.emotional_state}
                            </Badge>
                          </div>
                        </div>
                      </div>

                      <div>
                        <span className="text-sm font-medium text-gray-600 mb-2 block">Engagement Trend</span>
                        <div className="flex items-center gap-2">
                          {getEngagementIcon(conversationInsights.participant_profile.engagement_trend)}
                          <Progress 
                            value={conversationInsights.participant_profile.engagement_trend * 100} 
                            className="flex-1 h-2"
                          />
                          <span className="text-sm">
                            {Math.round(conversationInsights.participant_profile.engagement_trend * 100)}%
                          </span>
                        </div>
                      </div>

                      {conversationInsights.participant_profile.cultural_background && (
                        <div>
                          <span className="text-sm font-medium text-gray-600">Cultural Context</span>
                          <p className="text-sm mt-1 text-gray-700">
                            {JSON.stringify(conversationInsights.participant_profile.cultural_background, null, 2)}
                          </p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="competitive" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Competitive Context
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {conversationInsights.competitive_context && Object.keys(conversationInsights.competitive_context).length > 0 ? (
                      <div className="space-y-3">
                        {conversationInsights.competitive_context.active_competitors && (
                          <div>
                            <span className="text-sm font-medium text-gray-600">Active Competitors</span>
                            <div className="flex flex-wrap gap-2 mt-1">
                              {conversationInsights.competitive_context.active_competitors.map((competitor: string, index: number) => (
                                <Badge key={index} variant="destructive">
                                  {competitor}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto">
                          {JSON.stringify(conversationInsights.competitive_context, null, 2)}
                        </pre>
                      </div>
                    ) : (
                      <div className="text-center py-6 text-gray-500">
                        <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p>No competitive mentions detected</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="actions" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="h-5 w-5" />
                      Recommended Actions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {conversationInsights.current_state.priority_actions.map((action, index) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-start gap-2">
                            <div className={`w-2 h-2 rounded-full mt-2 ${
                              action.includes('URGENT') ? 'bg-red-500' : 
                              action.includes('Implement') ? 'bg-yellow-500' : 'bg-blue-500'
                            }`} />
                            <span className="text-sm">{action}</span>
                          </div>
                        </div>
                      ))}

                      {conversationInsights.current_state.priority_actions.length === 0 && (
                        <div className="text-center py-6 text-gray-500">
                          <Zap className="h-8 w-8 mx-auto mb-2 opacity-50" />
                          <p>No specific actions recommended</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center h-96">
                {isLoading ? (
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading conversation insights...</p>
                  </div>
                ) : (
                  <div className="text-center text-gray-500">
                    <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Select a conversation to view detailed insights</p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConversationIntelligenceDashboard;