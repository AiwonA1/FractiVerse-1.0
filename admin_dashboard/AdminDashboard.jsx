import React, { useState, useEffect } from "react";

const AdminDashboard = () => {
  const [metrics, setMetrics] = useState({
    cpu: "Loading...",
    memory: "Loading...",
    nodes: "Loading...",
    transactions: "Loading...",
  });

  const [command, setCommand] = useState("");
  const [response, setResponse] = useState("");

  // Fetch Live Data Every 5 Seconds
  useEffect(() => {
    const fetchMetrics = () => {
      fetch("/metrics")
        .then((res) => res.json())
        .then((data) => setMetrics(data));
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  // Send Commands
  const sendCommand = () => {
    fetch("/command", {
      method: "POST",
      body: JSON.stringify({ command }),
      headers: { "Content-Type": "application/json" },
    })
      .then((res) => res.json())
      .then((data) => setResponse(data.response));

    setCommand(""); // Clear input field
  };

  return (
    <div className="dashboard-container">
      <h1>FractiAdmin Dashboard</h1>

      {/* Live Metrics Display */}
      <div className="metrics">
        <p><strong>CPU Usage:</strong> {metrics.cpu}</p>
        <p><strong>Memory Usage:</strong> {metrics.memory}</p>
        <p><strong>Active AI Nodes:</strong> {metrics.nodes}</p>
        <p><strong>FractiChain Transactions:</strong> {metrics.transactions}</p>
      </div>

      {/* Command Input */}
      <div className="command-section">
        <h2>Command Console</h2>
        <input
          type="text"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          placeholder="Enter command..."
        />
        <button onClick={sendCommand}>Submit</button>
        <p className="response">{response}</p>
      </div>
    </div>
  );
};

export default AdminDashboard;
