import React, { useState, useEffect, useRef } from 'react';
import { CheckCircle, Clock, AlertCircle, Activity, Database, BarChart3, FileText } from 'lucide-react';
import { apiService, RunStatus } from '../services/api';

interface ProgressTrackerProps {
  onComplete: () => void;
  runId?: string;
}

const ProgressTracker: React.FC<ProgressTrackerProps> = ({ onComplete, runId }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [currentAgent, setCurrentAgent] = useState('Data Ingestion Agent');
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const statusCheckInterval = useRef<NodeJS.Timeout | null>(null);

  const analysisSteps = [
    { name: 'Data Ingestion', progress: 15, agent: 'Data Ingestion Agent' },
    { name: 'Schema Detection', progress: 30, agent: 'Schema Analysis Agent' },
    { name: 'Data Cleaning', progress: 50, agent: 'Data Cleaning Agent' },
    { name: 'Transformation', progress: 70, agent: 'Feature Engineering Agent' },
    { name: 'Analysis', progress: 85, agent: 'Statistical Analysis Agent' },
    { name: 'Report Generation', progress: 100, agent: 'Report Generation Agent' },
  ];

  const sampleLogs = [
    '✓ Loaded 45,230 rows from vehicle_registrations.csv',
    '📊 Detected schema: 12 columns (8 categorical, 4 numerical)',
    '⚠ Found 8% missing values in owner_gender column',
    '🔧 Applied median imputation to 3 numeric columns',
    '✓ Standardized date formats across temporal columns',
    '🎯 Generated 15 derived features from base dataset',
    '📈 Computing correlation matrix for 23 variables',
    '🗺️ Geocoding district names for spatial analysis',
    '📊 Running statistical hypothesis tests',
    '✓ Completed trend analysis for time series data',
    '🔍 Identified 3 significant outliers in registration data',
    '📝 Generating executive summary with key insights',
    '✅ Analysis complete - confidence score: 8.7/10',
  ];

  // Real-time status updates
  useEffect(() => {
    if (!runId) return;

    // Connect to WebSocket for real-time updates
    wsRef.current = apiService.connectWebSocket((data) => {
      if (data.run_id === runId) {
        setProgress(data.progress || 0);
        setCurrentAgent(data.current_step || 'Processing');
        
        if (data.message) {
          setLogs(prev => [...prev, data.message]);
        }
        
        if (data.status === 'completed') {
          setCurrentStep(analysisSteps.length - 1);
          setTimeout(() => onComplete(), 1000);
        } else if (data.status === 'failed') {
          setError(data.error || 'Analysis failed');
        }
      }
    });

    // Fallback: Poll status if WebSocket fails
    const pollStatus = async () => {
      try {
        const statusData = await apiService.getStatus(runId);
        setStatus(statusData);
        setProgress(statusData.progress);
        setCurrentAgent(statusData.current_step);
        
        if (statusData.message) {
          setLogs(prev => [...prev, statusData.message]);
        }
        
        if (statusData.status === 'completed') {
          setCurrentStep(analysisSteps.length - 1);
          setTimeout(() => onComplete(), 1000);
          if (statusCheckInterval.current) {
            clearInterval(statusCheckInterval.current);
          }
        } else if (statusData.status === 'failed') {
          setError(statusData.error || 'Analysis failed');
          if (statusCheckInterval.current) {
            clearInterval(statusCheckInterval.current);
          }
        }
      } catch (err) {
        console.error('Status check failed:', err);
      }
    };

    // Poll every 2 seconds
    statusCheckInterval.current = setInterval(pollStatus, 2000);
    pollStatus(); // Initial check

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
      }
    };
  }, [runId, onComplete]);

  // Update current step based on progress
  useEffect(() => {
    const currentStepIndex = analysisSteps.findIndex(step => step.progress >= progress) || 0;
    if (currentStepIndex !== currentStep) {
      setCurrentStep(currentStepIndex);
    }
  }, [progress, currentStep]);

  const getStepIcon = (stepIndex: number) => {
    if (stepIndex < currentStep) return <CheckCircle className="h-5 w-5 text-green-600" />;
    if (stepIndex === currentStep) return <Clock className="h-5 w-5 text-orange-600 animate-pulse" />;
    return <div className="h-5 w-5 rounded-full bg-gray-300"></div>;
  };

  const getAgentIcon = () => {
    switch (currentAgent) {
      case 'Data Ingestion Agent':
        return <Database className="h-6 w-6 text-blue-600" />;
      case 'Schema Analysis Agent':
        return <BarChart3 className="h-6 w-6 text-purple-600" />;
      case 'Data Cleaning Agent':
        return <Activity className="h-6 w-6 text-green-600" />;
      case 'Feature Engineering Agent':
        return <BarChart3 className="h-6 w-6 text-orange-600" />;
      case 'Statistical Analysis Agent':
        return <BarChart3 className="h-6 w-6 text-red-600" />;
      case 'Report Generation Agent':
        return <FileText className="h-6 w-6 text-indigo-600" />;
      default:
        return <CheckCircle className="h-6 w-6 text-green-600" />;
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold text-gray-900">AI Analysis in Progress</h2>
        <p className="text-lg text-gray-600">
          {status?.message || 'Processing your dataset with advanced AI analytics'}
        </p>
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-600" />
              <p className="text-red-800">{error}</p>
            </div>
          </div>
        )}
      </div>

      {/* Progress Overview */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-8">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-gray-900">Analysis Progress</h3>
          <div className="text-2xl font-bold text-orange-600">{progress}%</div>
        </div>
        
        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-3 mb-8">
          <div 
            className="bg-gradient-to-r from-orange-500 to-orange-600 h-3 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          ></div>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {analysisSteps.map((step, index) => (
            <div 
              key={index}
              className={`p-4 rounded-lg border-2 ${
                index < currentStep ? 'border-green-200 bg-green-50' :
                index === currentStep ? 'border-orange-200 bg-orange-50' :
                'border-gray-200 bg-gray-50'
              }`}
            >
              <div className="flex items-center space-x-3">
                {getStepIcon(index)}
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{step.name}</div>
                  <div className="text-sm text-gray-600">{step.agent}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Live Analysis Logs and Current Agent */}
      <div className="grid md:grid-cols-2 gap-8">
        {/* Current Agent Status */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Current Agent</h3>
          <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
            {getAgentIcon()}
            <div className="flex-1">
              <div className="font-semibold text-gray-900">{currentAgent}</div>
              <div className="text-sm text-gray-600 mt-1">
                {currentStep < analysisSteps.length - 1 ? 
                  `Processing ${analysisSteps[currentStep]?.name.toLowerCase()}...` :
                  'Finalizing analysis results'
                }
              </div>
              <div className="mt-2">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-green-600 animate-pulse" />
                  <span className="text-sm text-green-600 font-medium">Active</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Live Logs */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Live Analysis Log</h3>
          <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
            {logs.map((log, index) => (
              <div 
                key={index} 
                className="text-green-400 mb-1 animate-fade-in"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
              </div>
            ))}
            {logs.length > 0 && logs.length < sampleLogs.length && (
              <div className="text-green-400 animate-pulse">▋</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProgressTracker;