import React from 'react';
import { BarChart3, Shield, Database } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-gradient-to-r from-orange-500 to-orange-600 text-white shadow-lg">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Shield className="h-8 w-8" />
              <div>
                <h1 className="text-2xl font-bold tracking-tight">RTGS AI Analyst</h1>
                <p className="text-orange-100 text-sm">Government Data Intelligence Platform</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2 text-orange-100">
              <Database className="h-5 w-5" />
              <span className="text-sm">Secure • Compliant • Trusted</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                <BarChart3 className="h-5 w-5" />
              </div>
              <div className="text-sm">
                <div className="font-medium">Analysis Suite</div>
                <div className="text-orange-100">v2.1.0</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;