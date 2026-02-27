import { useState } from "react";
import "../styles/global.css";
import "../styles/dashboard.css";

import UploadBox from "../components/UploadBox";
import MetricsGrid from "../components/MetricsGrid";
import AnomalyTable from "../components/AnomalyTable";
import InsightsBox from "../components/InsightsBox";
import SpendingCharts from "../components/SpendingCharts";
import Chatbot from "../components/Chatbot";

import { analyzeSpending } from "../api/smartspend";

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (csvText, limit) => {
    try {
      setLoading(true);
      setError(null);

      const result = await analyzeSpending(csvText, limit);
      setData(result);

    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <h1 className="title">SmartSpend</h1>
      <p className="subtitle">Analyze your Paytm spending data</p>

      <UploadBox onAnalyze={handleAnalyze} />

      {loading && (
        <div style={{ marginTop: "20px" }}>
          <strong>Analyzing your transactions...</strong>
        </div>
      )}

      {error && (
        <div style={{ marginTop: "20px", color: "red" }}>
          Error: {error}
        </div>
      )}

      {data && (
        <>
          {/* Metrics */}
          <MetricsGrid metrics={data.metrics} />

          {/* Charts */}
          <SpendingCharts metrics={data.metrics} />

          {/* Anomalies */}
          <div style={{ marginTop: "40px", width: "100%", maxWidth: "1000px" }}>
            <AnomalyTable anomalies={data.metrics.anomalies} />
          </div>

          {/* AI Insights */}
          <div style={{ marginTop: "20px", width: "100%", maxWidth: "1000px" }}>
            <InsightsBox text={data.ai_suggestions} />
          </div>

          {/* Chatbot */}
          <div style={{ marginTop: "30px", width: "100%", maxWidth: "1000px" }}>
            <Chatbot metrics={data.metrics} />
          </div>
        </>
      )}
    </div>
  );
}

export default Dashboard;