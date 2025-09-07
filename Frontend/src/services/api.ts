/**
 * API Service for RTGS AI Analyst Backend Integration
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  file_path: string;
  rows: number;
  columns: string[];
  upload_time: string;
}

export interface AnalysisRequest {
  file_id: string;
  config: {
    business_questions: string[];
    key_metrics: string[];
    stakeholders: string;
    time_scope: string;
    geo_scope: string;
    analysis_focus: string;
  };
  request: {
    dataset_name: string;
    domain: string;
    scope: string;
    mode: string;
    sample_rows: number;
    auto_approve: boolean;
    report_format: string;
  };
}

export interface AnalysisResponse {
  run_id: string;
  status: string;
}

export interface RunStatus {
  run_id: string;
  status: string;
  progress: number;
  current_step: string;
  message: string;
  results?: any;
  error?: string;
}

export interface AnalysisResults {
  run_manifest: any;
  traditional_results: any;
  enhanced_results: any;
  artifacts: any;
}

export interface HistoryItem {
  run_id: string;
  filename: string;
  status: string;
  start_time: string;
  progress: number;
}

export class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    };

    const response = await fetch(url, { ...defaultOptions, ...options });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async uploadFile(file: File): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Upload failed: ${response.status}`);
    }

    return response.json();
  }

  async startAnalysis(
    fileId: string,
    config: AnalysisRequest['config'],
    request: AnalysisRequest['request']
  ): Promise<AnalysisResponse> {
    return this.request<AnalysisResponse>('/analyze', {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        config,
        request,
      }),
    });
  }

  async getStatus(runId: string): Promise<RunStatus> {
    return this.request<RunStatus>(`/status/${runId}`);
  }

  async getResults(runId: string): Promise<AnalysisResults> {
    return this.request<AnalysisResults>(`/results/${runId}`);
  }

  async getHistory(): Promise<{ sessions: HistoryItem[] }> {
    return this.request<{ sessions: HistoryItem[] }>('/history');
  }

  // WebSocket connection for real-time updates
  connectWebSocket(onMessage: (data: any) => void): WebSocket {
    const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws';
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    return ws;
  }
}

// Create singleton instance
export const apiService = new ApiService();
