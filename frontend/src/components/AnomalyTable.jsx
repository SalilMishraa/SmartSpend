function AnomalyTable({ anomalies }) {
    if (!anomalies || anomalies.length === 0) return null;
  
    return (
      <div className="metric-card">
        <h3>Anomalies</h3>
  
        {anomalies.map((a, index) => (
          <div key={index} className="anomaly-row">
            <div>
              <strong>{a.date}</strong> — {a.category}
            </div>
            <div>₹{a.amount.toFixed(2)}</div>
          </div>
        ))}
      </div>
    );
  }
  
  export default AnomalyTable;