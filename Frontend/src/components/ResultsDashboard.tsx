import React from 'react';
import { 
  TrendingUp, 
  Users, 
  MapPin, 
  Calendar, 
  Clock, 
  Star,
  Download,
  BarChart3,
  CheckCircle,
  AlertTriangle,
  XCircle
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Heatmap
} from 'recharts';
import { 
  vehicleRegistrationData, 
  districtData, 
  correlationMatrix,
  hypothesisTests,
  fuelTypeDistribution,
  vehicleTypeDistribution
} from '../data/mockData';

const ResultsDashboard: React.FC = () => {
  const kpiData = [
    { label: 'Total Records', value: '45,230', change: '+12%', icon: Users, color: 'blue' },
    { label: 'Data Completeness', value: '94.2%', change: '+2.1%', icon: CheckCircle, color: 'green' },
    { label: 'Geographic Coverage', value: '23 Districts', change: 'Complete', icon: MapPin, color: 'purple' },
    { label: 'Time Range', value: '12 Months', change: '2023', icon: Calendar, color: 'orange' },
    { label: 'Processing Time', value: '2.4 min', change: 'Optimal', icon: Clock, color: 'indigo' },
    { label: 'Confidence Score', value: '8.7/10', change: 'High', icon: Star, color: 'yellow' },
  ];

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'blue': return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'green': return 'bg-green-50 text-green-700 border-green-200';
      case 'purple': return 'bg-purple-50 text-purple-700 border-purple-200';
      case 'orange': return 'bg-orange-50 text-orange-700 border-orange-200';
      case 'indigo': return 'bg-indigo-50 text-indigo-700 border-indigo-200';
      case 'yellow': return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      default: return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Executive Summary */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Executive Summary</h2>
            <div className="flex items-center space-x-4 mt-2">
              <span className="text-sm font-medium text-gray-600">Run ID: rtgs-transport-20250106-001</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                High Confidence (8.7/10)
              </span>
            </div>
          </div>
          <div className="flex space-x-3">
            <button className="flex items-center space-x-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors">
              <Download className="h-4 w-4" />
              <span>Export PDF</span>
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-orange-50 to-green-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Dataset: Vehicle Registrations - Telangana 2023</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-800 mb-2">Key Findings:</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start space-x-2">
                  <TrendingUp className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                  <span>Vehicle registrations increased 12% YoY in 2023</span>
                </li>
                <li className="flex items-start space-x-2">
                  <MapPin className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                  <span>Rural districts show 35% lower per-capita registrations</span>
                </li>
                <li className="flex items-start space-x-2">
                  <BarChart3 className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
                  <span>Electric vehicle adoption doubled in urban areas</span>
                </li>
                <li className="flex items-start space-x-2">
                  <Users className="h-4 w-4 text-orange-600 mt-0.5 flex-shrink-0" />
                  <span>Young adults (25-35) account for 45% of registrations</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-800 mb-2">Recommendations:</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• Expand EV infrastructure in rural districts</li>
                <li>• Implement targeted registration drives in low-performing areas</li>
                <li>• Develop age-specific vehicle financing programs</li>
                <li>• Create digital-first registration processes</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {kpiData.map((kpi, index) => {
          const IconComponent = kpi.icon;
          return (
            <div key={index} className={`p-6 rounded-xl border-2 ${getColorClasses(kpi.color)}`}>
              <div className="flex items-center justify-between mb-3">
                <IconComponent className="h-8 w-8" />
                <span className="text-sm font-medium">{kpi.change}</span>
              </div>
              <div className="text-2xl font-bold mb-1">{kpi.value}</div>
              <div className="text-sm font-medium">{kpi.label}</div>
            </div>
          );
        })}
      </div>

      {/* Charts Section */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Time Series Chart */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Monthly Registration Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={vehicleRegistrationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="registrations" 
                stroke="#FF9933" 
                strokeWidth={3}
                dot={{ fill: '#FF9933', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-4 p-3 bg-green-50 rounded-lg">
            <p className="text-sm text-green-800">
              <TrendingUp className="h-4 w-4 inline mr-1" />
              Consistent upward trend with 12% overall growth. Peak registration period: Nov-Dec 2023.
            </p>
          </div>
        </div>

        {/* Fuel Type Distribution */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Fuel Type Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={fuelTypeDistribution}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {fuelTypeDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-sm text-blue-800">
              Electric vehicles show 8% adoption rate, doubling from 2022. CNG adoption remains limited.
            </p>
          </div>
        </div>

        {/* Vehicle Type Distribution */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Vehicle Type Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={vehicleTypeDistribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#138808" />
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 p-3 bg-green-50 rounded-lg">
            <p className="text-sm text-green-800">
              Two-wheelers dominate with 63% market share, followed by four-wheelers at 28%.
            </p>
          </div>
        </div>

        {/* Correlation Heatmap */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Variable Correlations</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr>
                  <th className="text-left p-2 font-medium text-gray-700">Variable</th>
                  {Object.keys(correlationMatrix[0]).slice(1).map(key => (
                    <th key={key} className="p-2 text-center font-medium text-gray-700">{key}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationMatrix.map((row, idx) => (
                  <tr key={idx}>
                    <td className="p-2 font-medium text-gray-900">{row.variable}</td>
                    {Object.entries(row).slice(1).map(([key, value], colIdx) => (
                      <td key={colIdx} className="p-2 text-center">
                        <div 
                          className={`px-2 py-1 rounded text-white text-sm font-medium ${
                            Math.abs(value) > 0.7 ? 'bg-red-600' :
                            Math.abs(value) > 0.4 ? 'bg-orange-500' :
                            Math.abs(value) > 0.2 ? 'bg-yellow-500' :
                            'bg-gray-400'
                          }`}
                        >
                          {value.toFixed(2)}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 p-3 bg-purple-50 rounded-lg">
            <p className="text-sm text-purple-800">
              Strong correlation (0.89) between population and registrations. Income and education show moderate positive correlation.
            </p>
          </div>
        </div>
      </div>

      {/* Statistical Analysis Tables */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Hypothesis Tests */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Statistical Hypothesis Tests</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Test</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Variables</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-900">P-Value</th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-900">Significant</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Effect Size</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {hypothesisTests.map((test, idx) => (
                  <tr key={idx}>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{test.test}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">{test.variables}</td>
                    <td className="px-4 py-3 text-center text-sm text-gray-900">{test.pValue}</td>
                    <td className="px-4 py-3 text-center">
                      {test.significant ? (
                        <CheckCircle className="h-5 w-5 text-green-600 mx-auto" />
                      ) : (
                        <XCircle className="h-5 w-5 text-red-600 mx-auto" />
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">{test.effectSize}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Regional Performance */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Top 10 Districts Performance</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Rank</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">District</th>
                  <th className="px-4 py-3 text-right text-sm font-semibold text-gray-900">Total</th>
                  <th className="px-4 py-3 text-right text-sm font-semibold text-gray-900">Per Capita</th>
                  <th className="px-4 py-3 text-right text-sm font-semibold text-gray-900">Growth %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {districtData.map((district, idx) => (
                  <tr key={idx}>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{district.rank}</td>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{district.district}</td>
                    <td className="px-4 py-3 text-sm text-gray-900 text-right">{district.total.toLocaleString()}</td>
                    <td className="px-4 py-3 text-sm text-gray-900 text-right">{district.perCapita}</td>
                    <td className="px-4 py-3 text-sm text-right">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        district.growth > 10 ? 'bg-green-100 text-green-800' :
                        district.growth > 5 ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        +{district.growth}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Report Generation Section */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-8">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Generate Reports</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <button className="flex items-center justify-center space-x-2 p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <Download className="h-5 w-5 text-orange-600" />
            <span>PDF Executive Report</span>
          </button>
          <button className="flex items-center justify-center space-x-2 p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <Download className="h-5 w-5 text-green-600" />
            <span>Excel Data Export</span>
          </button>
          <button className="flex items-center justify-center space-x-2 p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
            <Download className="h-5 w-5 text-blue-600" />
            <span>Interactive HTML Report</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;