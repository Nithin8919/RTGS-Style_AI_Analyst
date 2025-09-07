import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { sampleDatasets } from '../data/mockData';
import { apiService, FileUploadResponse } from '../services/api';

interface FileUploaderProps {
  onFileSelect: (file: any) => void;
  onConfigChange: (config: any) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileSelect, onConfigChange }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<any>(null);
  const [domain, setDomain] = useState('auto');
  const [scope, setScope] = useState('');
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setUploadError(null);
      setIsUploading(true);
      
      try {
        const response = await apiService.uploadFile(file);
        setUploadedFile(response);
        
        // Generate preview data from uploaded file info
        const mockPreview = [
          { Column: 'Sample Data', Type: 'Preview', Count: response.rows },
          { Column: 'Columns', Type: 'Detected', Count: response.columns.length },
          { Column: 'File', Type: 'Uploaded', Count: 1 },
        ];
        setPreviewData(mockPreview);
      } catch (error) {
        setUploadError(error instanceof Error ? error.message : 'Upload failed');
        setSelectedFile(null);
      } finally {
        setIsUploading(false);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: false,
  });

  const handleSampleDataset = (dataset: any) => {
    setSelectedDataset(dataset);
    setSelectedFile(null);
    setDomain(dataset.domain);
    setScope(dataset.scope);
    setPreviewData(dataset.preview);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile && !selectedDataset) return;
    
    const config = {
      business_questions: ['What are the key trends and patterns?'],
      key_metrics: [],
      stakeholders: 'Government officials',
      time_scope: 'Not specified',
      geo_scope: scope || 'Regional Analysis',
      analysis_focus: 'balanced'
    };
    
    const request = {
      dataset_name: uploadedFile?.filename || selectedDataset?.name || 'Unknown',
      domain: domain,
      scope: scope,
      mode: 'run',
      sample_rows: 500,
      auto_approve: true,
      report_format: 'pdf'
    };
    
    onConfigChange({ config, request, file: uploadedFile || selectedDataset });
    onFileSelect(uploadedFile || selectedDataset);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-8">
        <h2 className="text-3xl font-bold text-gray-900">
          Government Data Intelligence Platform
        </h2>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Upload your government datasets and get AI-powered insights, trends, and policy recommendations
          in minutes with our secure, compliant analysis platform.
        </p>
      </div>

      {/* File Upload Section */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-8">
        <h3 className="text-2xl font-semibold text-gray-900 mb-6">Upload Dataset</h3>
        
        <div {...getRootProps()} className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all duration-200
          ${isDragActive ? 'border-orange-400 bg-orange-50' : 'border-gray-300 hover:border-orange-400 hover:bg-gray-50'}
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}>
          <input {...getInputProps()} disabled={isUploading} />
          {isUploading ? (
            <Loader2 className="h-16 w-16 mx-auto text-orange-600 mb-4 animate-spin" />
          ) : (
            <Upload className="h-16 w-16 mx-auto text-gray-400 mb-4" />
          )}
          {isUploading ? (
            <div>
              <p className="text-xl font-medium text-orange-600">Uploading file...</p>
              <p className="text-gray-500 mt-2">Please wait while we process your file</p>
            </div>
          ) : isDragActive ? (
            <div>
              <p className="text-xl font-medium text-orange-600">Drop your file here</p>
              <p className="text-gray-500 mt-2">CSV, XLS, XLSX formats supported</p>
            </div>
          ) : (
            <div>
              <p className="text-xl font-medium text-gray-900">Drag & drop your data file</p>
              <p className="text-gray-500 mt-2">or click to browse files</p>
              <p className="text-sm text-gray-400 mt-4">Supports CSV, XLS, XLSX up to 50MB</p>
            </div>
          )}
        </div>

        {/* Upload Error */}
        {uploadError && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-600" />
              <p className="text-red-800">{uploadError}</p>
            </div>
          </div>
        )}

        {/* Upload Success */}
        {uploadedFile && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              <p className="text-green-800">
                File uploaded successfully: {uploadedFile.filename} ({uploadedFile.rows.toLocaleString()} rows)
              </p>
            </div>
          </div>
        )}

        {/* Sample Datasets */}
        <div className="mt-8">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Try Sample Datasets</h4>
          <div className="grid md:grid-cols-3 gap-4">
            {sampleDatasets.map((dataset) => (
              <button
                key={dataset.id}
                onClick={() => handleSampleDataset(dataset)}
                className={`p-4 rounded-lg border-2 text-left transition-all duration-200 ${
                  selectedDataset?.id === dataset.id
                    ? 'border-orange-400 bg-orange-50'
                    : 'border-gray-200 hover:border-orange-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <FileText className="h-5 w-5 text-orange-600" />
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    dataset.domain === 'Transport' ? 'bg-blue-100 text-blue-800' :
                    dataset.domain === 'Health' ? 'bg-green-100 text-green-800' :
                    'bg-purple-100 text-purple-800'
                  }`}>
                    {dataset.domain}
                  </span>
                </div>
                <h5 className="font-medium text-gray-900 mb-1">{dataset.name}</h5>
                <p className="text-sm text-gray-600 mb-2">{dataset.scope}</p>
                <div className="text-xs text-gray-500">
                  {dataset.rows.toLocaleString()} rows • {dataset.size}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Configuration Panel */}
        {(selectedFile || selectedDataset) && (
          <div className="mt-8 p-6 bg-gray-50 rounded-lg">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">Analysis Configuration</h4>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Domain</label>
                <select
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                >
                  <option value="auto">Auto-detect (AI-powered)</option>
                  <option value="transport">Transport</option>
                  <option value="health">Health</option>
                  <option value="education">Education</option>
                  <option value="economics">Economics</option>
                  <option value="agriculture">Agriculture</option>
                  <option value="environment">Environment</option>
                  <option value="urban">Urban Development</option>
                  <option value="social">Social Services</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Analysis Scope</label>
                <input
                  type="text"
                  value={scope}
                  onChange={(e) => setScope(e.target.value)}
                  placeholder="e.g., Telangana 2023, Hyderabad District"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
              </div>
            </div>

            {/* File Preview */}
            {previewData.length > 0 && (
              <div className="mt-6">
                <h5 className="text-md font-medium text-gray-900 mb-3">Data Preview (First 3 rows)</h5>
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white border border-gray-200 rounded-md">
                    <thead className="bg-gray-100">
                      <tr>
                        {Object.keys(previewData[0]).map((key) => (
                          <th key={key} className="px-4 py-2 text-left text-sm font-medium text-gray-700">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.map((row, idx) => (
                        <tr key={idx} className="border-t border-gray-200">
                          {Object.values(row).map((value: any, colIdx) => (
                            <td key={colIdx} className="px-4 py-2 text-sm text-gray-900">
                              {value}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={!scope || (!uploadedFile && !selectedDataset)}
              className="mt-6 w-full bg-gradient-to-r from-orange-500 to-orange-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-orange-600 hover:to-orange-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-2"
            >
              <CheckCircle className="h-5 w-5" />
              <span>Start AI Analysis</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUploader;