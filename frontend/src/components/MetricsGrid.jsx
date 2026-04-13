function MetricsGrid({ metrics }) {
    const topDays = metrics.top_3_days || [];
    const categories = metrics.category_spending || {};
  
    return (
      <div className="metrics-grid">
  
        {/* Total Spent */}
        <div className="metric-card">
          <h3>Total Spent</h3>
          <p>₹{metrics.total_spent.toFixed(2)}</p>
        </div>
  
        {/* Average Daily */}
        <div className="metric-card">
          <h3>Average Daily</h3>
          <p>₹{metrics.avg_daily_spending.toFixed(2)}</p>
        </div>
  
        {/* Top 3 Days */}
        <div className="metric-card">
          <h3>Top 3 Days</h3>
          {topDays.map((day, index) => (
            <div key={index} className="topday-row">
              <span>{day.date}</span>
              <strong>₹{day.amount.toFixed(2)}</strong>
            </div>
          ))}
        </div>
  
        {/* Category Breakdown */}
        <div className="metric-card">
          <h3>Category Breakdown</h3>
          {Object.entries(categories).map(([cat, amt]) => (
            <div key={cat} className="category-row">
              <span>{cat}</span>
              <span>₹{amt.toFixed(2)}</span>
            </div>
          ))}
        </div>
  
      </div>
    );
  }
  
  export default MetricsGrid;