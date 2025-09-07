import React, { useState } from 'react';
import Header from './components/Header';
import FileUploader from './components/FileUploader';
import ProgressTracker from './components/ProgressTracker';
import ResultsDashboard from './components/ResultsDashboard';
import RunHistory from './components/RunHistory';
import { BarChart3, History, Upload } from 'lucide-react';
import { apiService, AnalysisResponse } from './services/api';

type AppState = 'upload' | 'analysis' | 'results' | 'history';

function App() {
  const [currentState, setCurrentState] = useState<AppState>('upload');
  const [selectedFile, setSelectedFile] = useState<any>(null);
  const [analysisConfig, setAnalysisConfig] = useState<any>(null);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);

  const handleFileSelect = (file: any) => {
    setSelectedFile(file);
    setCurrentState('analysis');
  };

  const handleConfigChange = async (config: any) => {
    setAnalysisConfig(config);
    
    // Start analysis if we have a file and config
    if (config.file && config.config && config.request) {
      try {
        const response = await apiService.startAnalysis(
          config.file.file_id,
          config.config,
          config.request
        );
        setCurrentRunId(response.run_id);
        setCurrentState('analysis');
      } catch (error) {
        console.error('Failed to start analysis:', error);
        // Handle error - maybe show error message
      }
    }
  };

  const handleAnalysisComplete = () => {
    setCurrentState('results');
  };

  const navigationItems = [
    { id: 'upload', label: 'Upload Data', icon: Upload },
    { id: 'results', label: 'Results', icon: BarChart3 },
    { id: 'history', label: 'History', icon: History },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="container mx-auto px-6">
          <div className="flex space-x-8">
            {navigationItems.map((item) => {
              const IconComponent = item.icon;
              const isActive = currentState === item.id || 
                (item.id === 'results' && currentState === 'analysis');
              
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentState(item.id as AppState)}
                  disabled={item.id === 'results' && !selectedFile}
                  className={`flex items-center space-x-2 py-4 px-2 border-b-2 transition-colors ${
                    isActive
                      ? 'border-orange-500 text-orange-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } ${item.id === 'results' && !selectedFile ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <IconComponent className="h-5 w-5" />
                  <span className="font-medium">{item.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="py-8">
        {currentState === 'upload' && (
          <FileUploader 
            onFileSelect={handleFileSelect}
            onConfigChange={handleConfigChange}
          />
        )}
        
        {currentState === 'analysis' && (
          <ProgressTracker 
            onComplete={handleAnalysisComplete}
            runId={currentRunId || undefined}
          />
        )}
        
        {currentState === 'results' && (
          <ResultsDashboard />
        )}
        
        {currentState === 'history' && (
          <RunHistory />
        )}
      </main>
    </div>
  );
}

export default App;