import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
    LineChart,
    Line
  } from "recharts";
  
  function SpendingCharts({ metrics }) {
    const categoryData = Object.entries(metrics.category_spending || {}).map(
      ([category, amount]) => ({
        category,
        amount
      })
    );
  
    const dailyData = metrics.daily_spending || [];
  
    return (
      <div style={{ marginTop: "40px", width: "100%", maxWidth: "1000px" }}>
        
        <div className="metric-card" style={{ marginBottom: "30px" }}>
          <h3>Spending by Category</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="amount" fill="#4f46e5" />
            </BarChart>
          </ResponsiveContainer>
        </div>
  
        <div className="metric-card">
          <h3>Daily Spending Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dailyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="amount" stroke="#2563eb" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
  
      </div>
    );
  }
  
  export default SpendingCharts;