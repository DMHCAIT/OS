import React from 'react';
import { Outlet } from 'react-router-dom';

const Layout: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 bg-white shadow-sm">
          <div className="p-4">
            <h1 className="text-xl font-bold text-gray-900">AI Lead Management</h1>
          </div>
          <nav className="mt-4">
            <div className="px-4 space-y-2">
              <a href="/dashboard" className="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">
                Dashboard
              </a>
              <a href="/leads" className="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">
                Leads
              </a>
              <a href="/calls" className="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">
                Calls
              </a>
              <a href="/voice-ai" className="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">
                Voice AI
              </a>
              <a href="/analytics" className="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">
                Analytics
              </a>
              <a href="/settings" className="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">
                Settings
              </a>
            </div>
          </nav>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;