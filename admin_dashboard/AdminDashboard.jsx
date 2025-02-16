import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Gauge } from "@/components/ui/gauge";
import { motion } from "framer-motion";

const AdminDashboard = () => {
  const [adminMode, setAdminMode] = useState(false);
  const [metrics, setMetrics] = useState({
    cognition: 0,
    network: 0,
    security: 0,
    chain: 0,
    treasury: 0,
    users: 0,
  });

  // Function to fetch real-time data from backend API
  const fetchMetrics = async () => {
    try {
      const response = await fetch("https://fracticody1-0-1.onrender.com/api/metrics");
      const data = await response.json();
      setMetrics({
        cognition: data.cognition || 0,
        network: data.network || 0,
        security: data.security || 0,
        chain: data.chain || 0,
        treasury: data.treasury || 0,
        users: data.users || 0,
      });
    } catch (error) {
      console.error("Error fetching metrics:", error);
    }
  };

  // Refresh data every 5 seconds
  useEffect(() => {
    fetchMetrics(); // Fetch immediately
    const interval = setInterval(fetchMetrics, 5000); // Fetch every 5s
    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-6 space-y-6">
      
      {/* Top Panel */}
      <motion.div
        className="w-full max-w-5xl flex justify-between items-center p-4 bg-gray-800 rounded-xl shadow-lg"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h1 className="text-2xl font-semibold">🚀 FractiAdmin 1.0 Dashboard</h1>
        <Button className="bg-blue-500 hover:bg-blue-600" onClick={() => setAdminMode(!adminMode)}>
          {adminMode ? "Exit Admin Mode" : "Enter Admin Mode"}
        </Button>
      </motion.div>

      {/* Metrics Dashboard */}
      <div className="w-full max-w-5xl grid grid-cols-2 md:grid-cols-3 gap-6">
        {Object.entries(metrics).map(([key, value]) => (
          <Card key={key} className="bg-gray-900 p-6 rounded-lg shadow-md hover:shadow-xl transition-shadow duration-300">
            <CardContent className="flex flex-col items-center">
              <h2 className="text-lg font-medium capitalize text-gray-300">{key}</h2>
              <Gauge value={value} className="text-center text-3xl font-bold mt-2" />
              <p className={`mt-1 text-sm font-semibold ${value > 70 ? "text-green-400" : value > 40 ? "text-yellow-400" : "text-red-400"}`}>
                {value}%
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Command Console */}
      <motion.div
        className="w-full max-w-3xl p-4 bg-gray-800 rounded-xl shadow-lg mt-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h3 className="text-lg font-medium text-gray-300">Command Console</h3>
        <input
          type="text"
          className="w-full mt-2 p-3 bg-gray-700 rounded-md outline-none text-white placeholder-gray-400 border border-gray-600 focus:border-blue-500"
          placeholder="Enter command..."
        />
      </motion.div>
    </div>
  );
};

export default AdminDashboard;
