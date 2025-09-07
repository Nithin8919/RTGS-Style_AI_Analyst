import React, { useState } from 'react';
import { 
  Search, 
  Filter, 
  Star, 
  Calendar, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle,
  XCircle,
  Eye,
  Download,
  Trash2
} from 'lucide-react';
import { runHistory } from '../data/mockData';

const RunHistory: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDomain, setFilterDomain] = useState('All');
  const [filterStatus, setFilterStatus] = useState('All');

  const filteredRuns = runHistory.filter(run => {
    const matchesSearch = run.dataset.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         run.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesDomain = filterDomain === 'All' || run.domain === filterDomain;
    const matchesStatus = filterStatus === 'All' || run.status === filterStatus;
    
    return matchesSearch && matchesDomain && matchesStatus;
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Completed':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'Failed':
        return <XCircle className="h-5 w-5 text-red-600" />;
      case 'Running':
        return <div className="h-5 w-5 border-2 border-orange-600 border-t-transparent rounded-full animate-spin" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 8) return 'text-green-600 bg-green-100';
    if (confidence >= 6) return 'text-yellow-600 bg-yellow-100';
    if (confidence >= 4) return 'text-orange-600 bg-orange-100';
    return 'text-red-600 bg-red-100';
  };

  const getDomainColor = (domain: string) => {
    switch (domain) {
      case 'Transport':
        return 'bg-blue-100 text-blue-800';
      case 'Health':
        return 'bg-green-100 text-green-800';
      case 'Education':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Analysis History</h2>
          <p className="text-lg text-gray-600 mt-1">
            View and manage your previous data analysis runs
          </p>
        </div>
        <div className="flex space-x-3">
          <button className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors">
            New Analysis
          </button>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
        <div className="flex flex-wrap items-center gap-4">
          {/* Search */}
          <div className="flex-1 min-w-64">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search by dataset name or run ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Domain Filter */}
          <div className="flex items-center space-x-2">
            <Filter className="h-5 w-5 text-gray-500" />
            <select
              value={filterDomain}
              onChange={(e) => setFilterDomain(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
            >
              <option value="All">All Domains</option>
              <option value="Transport">Transport</option>
              <option value="Health">Health</option>
              <option value="Education">Education</option>
            </select>
          </div>

          {/* Status Filter */}
          <div>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
            >
              <option value="All">All Status</option>
              <option value="Completed">Completed</option>
              <option value="Failed">Failed</option>
              <option value="Running">Running</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results Summary */}
      <div className="flex items-center justify-between text-sm text-gray-600">
        <span>Showing {filteredRuns.length} of {runHistory.length} analysis runs</span>
        <div className="flex items-center space-x-4">
          <span>Sort by:</span>
          <select className="px-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-orange-500">
            <option>Most Recent</option>
            <option>Highest Confidence</option>
            <option>Dataset Name</option>
          </select>
        </div>
      </div>

      {/* Runs Table */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Run Details</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Dataset</th>
                <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">Domain</th>
                <th className="px-6 py-4 text-center text-sm font-semibold text-gray-900">Status</th>
                <th className="px-6 py-4 text-center text-sm font-semibold text-gray-900">Confidence</th>
                <th className="px-6 py-4 text-center text-sm font-semibold text-gray-900">Duration</th>
                <th className="px-6 py-4 text-center text-sm font-semibold text-gray-900">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredRuns.map((run, idx) => (
                <tr key={idx} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-4">
                    <div className="flex items-center space-x-3">
                      {run.favorite && <Star className="h-4 w-4 text-yellow-500 fill-current" />}
                      <div>
                        <div className="font-medium text-gray-900 text-sm">{run.id}</div>
                        <div className="text-sm text-gray-500 flex items-center space-x-1">
                          <Calendar className="h-3 w-3" />
                          <span>{run.date}</span>
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm font-medium text-gray-900">{run.dataset}</div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDomainColor(run.domain)}`}>
                      {run.domain}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <div className="flex items-center justify-center space-x-2">
                      {getStatusIcon(run.status)}
                      <span className="text-sm font-medium">{run.status}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-center">
                    {run.confidence > 0 ? (
                      <span className={`px-2 py-1 rounded-full text-sm font-medium ${getConfidenceColor(run.confidence)}`}>
                        {run.confidence}/10
                      </span>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-center">
                    <span className="text-sm text-gray-900">{run.duration}</span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center justify-center space-x-2">
                      <button 
                        className="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                        title="View Results"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      {run.status === 'Completed' && (
                        <button 
                          className="p-1 text-gray-400 hover:text-green-600 transition-colors"
                          title="Download Report"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                      )}
                      <button 
                        className="p-1 text-gray-400 hover:text-yellow-600 transition-colors"
                        title="Toggle Favorite"
                      >
                        <Star className={`h-4 w-4 ${run.favorite ? 'fill-current text-yellow-500' : ''}`} />
                      </button>
                      <button 
                        className="p-1 text-gray-400 hover:text-red-600 transition-colors"
                        title="Delete Run"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {filteredRuns.length === 0 && (
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-12 text-center">
          <div className="text-gray-400 mb-4">
            <Search className="h-16 w-16 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No analysis runs found</h3>
          <p className="text-gray-600 mb-6">
            {searchTerm || filterDomain !== 'All' || filterStatus !== 'All'
              ? 'Try adjusting your search criteria or filters.'
              : 'Start your first analysis by uploading a dataset.'}
          </p>
          <button className="px-6 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors">
            New Analysis
          </button>
        </div>
      )}
    </div>
  );
};

export default RunHistory;