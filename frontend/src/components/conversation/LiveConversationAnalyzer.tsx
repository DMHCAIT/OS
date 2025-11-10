import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import {
  Mic,
  MicOff,
  Send,
  Brain,
  Heart,
  Globe,
  Shield,
  Zap,
  MessageCircle,
  Users,
  TrendingUp,
  AlertTriangle,
  Volume2
} from 'lucide-react';

interface RealTimeAnalysisResult {
  success: boolean;
  conversation_id: string;
  analysis: {
    priority_actions: string[];
    health_score: number;
    engagement_forecast: string;
    adaptive_strategy: string;
    emotion_state: string;
    competitive_alerts: boolean;
    localization_needed: boolean;
  };
  detailed_analysis: {
    language: any;
    emotion: any;
    adaptation: any;
    competitive: any;
  };
  recommendations: {
    script_adaptation: string;
    empathy_response: any;
    competitive_responses: any;
    localized_response: any;
  };
  processing_time: number;
}

interface LiveConversationAnalyzer extends React.FC {
  conversationId?: string;
  onAnalysisUpdate?: (analysis: RealTimeAnalysisResult) => void;
}

const LiveConversationAnalyzer: LiveConversationAnalyzer = ({ 
  conversationId: propConversationId, 
  onAnalysisUpdate 
}) => {
  const [conversationId, setConversationId] = useState(propConversationId || '');
  const [participantId, setParticipantId] = useState('');
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [autoAnalysis, setAutoAnalysis] = useState(true);
  const [currentAnalysis, setCurrentAnalysis] = useState<RealTimeAnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<RealTimeAnalysisResult[]>([]);
  const [streamingInsights, setStreamingInsights] = useState<any>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Initialize conversation ID if not provided
  useEffect(() => {
    if (!conversationId) {
      setConversationId(`conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    }
    if (!participantId) {
      setParticipantId(`user_${Math.random().toString(36).substr(2, 9)}`);
    }
  }, [conversationId, participantId]);

  // Setup real-time streaming insights
  useEffect(() => {
    if (conversationId && autoAnalysis) {
      const eventSource = new EventSource(
        `/api/v1/conversation-intelligence/stream-insights/${conversationId}`
      );

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setStreamingInsights(data);
        } catch (error) {
          console.error('Error parsing streaming data:', error);
        }
      };

      eventSource.onerror = (error) => {
        console.error('EventSource failed:', error);
        eventSource.close();
      };

      eventSourceRef.current = eventSource;

      return () => {
        eventSource.close();
      };
    }
  }, [conversationId, autoAnalysis]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await analyzeWithAudio(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const analyzeWithAudio = async (audioBlob?: Blob) => {
    if (!message.trim() || !conversationId || !participantId) return;

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('conversation_id', conversationId);
      formData.append('participant_id', participantId);
      formData.append('message', message);
      
      if (audioBlob) {
        formData.append('audio_file', audioBlob, 'audio.wav');
      }

      const response = await fetch('/api/v1/conversation-intelligence/analyze-with-audio', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      
      if (result.success) {
        setCurrentAnalysis(result);
        setAnalysisHistory(prev => [...prev, result].slice(-10)); // Keep last 10 analyses
        onAnalysisUpdate?.(result);
      }
    } catch (error) {
      console.error('Error analyzing conversation:', error);
    } finally {
      setIsAnalyzing(false);
      setMessage('');
    }
  };

  const quickAnalyze = async () => {
    if (!message.trim() || !conversationId || !participantId) return;

    setIsAnalyzing(true);

    try {
      const response = await fetch('/api/v1/conversation-intelligence/quick-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          participant_id: participantId,
          message: message,
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        // Update with quick insights format
        const quickAnalysis = {
          ...result,
          analysis: result.quick_insights,
          detailed_analysis: {},
          recommendations: {},
          processing_time: 0
        };
        setCurrentAnalysis(quickAnalysis);
        onAnalysisUpdate?.(quickAnalysis);
      }
    } catch (error) {
      console.error('Error in quick analysis:', error);
    } finally {
      setIsAnalyzing(false);
      setMessage('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (autoAnalysis) {
      await analyzeWithAudio();
    } else {
      await quickAnalyze();
    }
  };

  const getHealthScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getEmotionColor = (emotion: string) => {
    const colors = {
      happy: 'bg-green-100 text-green-800',
      excited: 'bg-blue-100 text-blue-800',
      neutral: 'bg-gray-100 text-gray-800',
      concerned: 'bg-yellow-100 text-yellow-800',
      frustrated: 'bg-red-100 text-red-800',
      confused: 'bg-purple-100 text-purple-800'
    };
    return colors[emotion as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Live Conversation Analyzer
          </CardTitle>
          <CardDescription>
            Real-time AI analysis of conversation messages with advanced intelligence
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="text-sm font-medium">Conversation ID</label>
              <Input 
                value={conversationId} 
                onChange={(e) => setConversationId(e.target.value)}
                placeholder="auto-generated"
                className="mt-1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Participant ID</label>
              <Input 
                value={participantId} 
                onChange={(e) => setParticipantId(e.target.value)}
                placeholder="participant identifier"
                className="mt-1"
              />
            </div>
            <div className="flex items-end">
              <div className="flex items-center space-x-2">
                <Switch 
                  checked={autoAnalysis} 
                  onCheckedChange={setAutoAnalysis}
                  id="auto-analysis"
                />
                <label htmlFor="auto-analysis" className="text-sm font-medium">
                  Full Analysis
                </label>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Message Input */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageCircle className="h-5 w-5" />
              Message Input
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Textarea
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Enter conversation message..."
                  rows={4}
                  className="resize-none"
                />
              </div>

              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant={isRecording ? "destructive" : "outline"}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={isAnalyzing}
                >
                  {isRecording ? (
                    <>
                      <MicOff className="h-4 w-4 mr-2" />
                      Stop Recording
                    </>
                  ) : (
                    <>
                      <Mic className="h-4 w-4 mr-2" />
                      Record Audio
                    </>
                  )}
                </Button>

                <Button
                  type="submit"
                  disabled={!message.trim() || isAnalyzing || isRecording}
                  className="flex-1"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      {autoAnalysis ? 'Full Analysis' : 'Quick Analysis'}
                    </>
                  )}
                </Button>
              </div>

              {isRecording && (
                <div className="flex items-center gap-2 text-red-600">
                  <div className="animate-pulse">
                    <Volume2 className="h-4 w-4" />
                  </div>
                  <span className="text-sm">Recording in progress...</span>
                </div>
              )}
            </form>
          </CardContent>
        </Card>

        {/* Real-time Insights */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Real-time Insights
            </CardTitle>
          </CardHeader>
          <CardContent>
            {streamingInsights ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Health Score</span>
                  <span className={`font-bold ${getHealthScoreColor(streamingInsights.health_score)}`}>
                    {Math.round(streamingInsights.health_score * 100)}%
                  </span>
                </div>
                <Progress value={streamingInsights.health_score * 100} className="h-2" />

                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-600">Engagement:</span>
                    <p className="font-medium">{streamingInsights.engagement}</p>
                  </div>
                  <div>
                    <span className="text-gray-600">Strategy:</span>
                    <p className="font-medium">{streamingInsights.adaptive_strategy}</p>
                  </div>
                </div>

                {streamingInsights.priority_action && (
                  <div className="p-2 bg-blue-50 rounded">
                    <span className="text-xs font-medium text-blue-800">Next Action:</span>
                    <p className="text-sm text-blue-700">{streamingInsights.priority_action}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Start conversation to see real-time insights</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Analysis Results */}
      {currentAnalysis && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Analysis Results
              </span>
              <Badge variant="outline">
                {currentAnalysis.processing_time?.toFixed(2)}s
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Quick Insights */}
              <div className="space-y-4">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  Quick Insights
                </h3>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Health Score</span>
                    <span className={`font-bold ${getHealthScoreColor(currentAnalysis.analysis.health_score)}`}>
                      {Math.round(currentAnalysis.analysis.health_score * 100)}%
                    </span>
                  </div>

                  {currentAnalysis.analysis.emotion_state && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Emotion</span>
                      <Badge className={getEmotionColor(currentAnalysis.analysis.emotion_state)}>
                        {currentAnalysis.analysis.emotion_state}
                      </Badge>
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <span className="text-sm">Engagement</span>
                    <span className="text-sm font-medium">{currentAnalysis.analysis.engagement_forecast}</span>
                  </div>

                  {currentAnalysis.analysis.competitive_alerts && (
                    <div className="flex items-center gap-2 p-2 bg-red-50 rounded">
                      <Shield className="h-4 w-4 text-red-600" />
                      <span className="text-sm text-red-700">Competitive mention detected</span>
                    </div>
                  )}

                  {currentAnalysis.analysis.localization_needed && (
                    <div className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                      <Globe className="h-4 w-4 text-blue-600" />
                      <span className="text-sm text-blue-700">Localization available</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Priority Actions */}
              <div className="space-y-4">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Priority Actions
                </h3>

                <div className="space-y-2">
                  {currentAnalysis.analysis.priority_actions?.map((action, index) => (
                    <div key={index} className="p-2 bg-gray-50 rounded text-sm">
                      <div className="flex items-start gap-2">
                        <div className={`w-2 h-2 rounded-full mt-2 ${
                          action.includes('URGENT') ? 'bg-red-500' : 
                          action.includes('Implement') ? 'bg-yellow-500' : 'bg-blue-500'
                        }`} />
                        <span>{action}</span>
                      </div>
                    </div>
                  )) || (
                    <p className="text-sm text-gray-500">No specific actions recommended</p>
                  )}
                </div>
              </div>

              {/* Recommendations */}
              <div className="space-y-4">
                <h3 className="font-semibold text-sm flex items-center gap-2">
                  <Heart className="h-4 w-4" />
                  Recommendations
                </h3>

                <div className="space-y-3">
                  {currentAnalysis.recommendations?.script_adaptation && (
                    <div>
                      <span className="text-xs font-medium text-gray-600">Script Adaptation:</span>
                      <p className="text-sm mt-1">{currentAnalysis.recommendations.script_adaptation}</p>
                    </div>
                  )}

                  {currentAnalysis.recommendations?.empathy_response && (
                    <div>
                      <span className="text-xs font-medium text-gray-600">Empathy Response:</span>
                      <div className="text-sm mt-1 space-y-1">
                        {typeof currentAnalysis.recommendations.empathy_response === 'object' ? 
                          Object.entries(currentAnalysis.recommendations.empathy_response).map(([key, value]) => (
                            <div key={key}>
                              <span className="font-medium">{key}:</span> {String(value)}
                            </div>
                          )) :
                          <p>{String(currentAnalysis.recommendations.empathy_response)}</p>
                        }
                      </div>
                    </div>
                  )}

                  {currentAnalysis.recommendations?.localized_response && (
                    <div>
                      <span className="text-xs font-medium text-gray-600">Localized Response:</span>
                      <p className="text-sm mt-1">{String(currentAnalysis.recommendations.localized_response)}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Analysis History */}
      {analysisHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Analysis History
            </CardTitle>
            <CardDescription>
              Recent conversation analyses
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {analysisHistory.slice(-5).reverse().map((analysis, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${getHealthScoreColor(analysis.analysis.health_score).replace('text-', 'bg-')}`} />
                    <span className="text-sm">
                      Score: {Math.round(analysis.analysis.health_score * 100)}%
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {analysis.processing_time?.toFixed(2)}s
                    </Badge>
                  </div>
                  <span className="text-xs text-gray-500">
                    {analysis.analysis.engagement_forecast}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default LiveConversationAnalyzer;