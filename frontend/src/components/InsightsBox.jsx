import ReactMarkdown from "react-markdown";

function InsightsBox({ text }) {
  if (!text) return null;

  return (
    <div className="metric-card">
      <h3>AI Insights</h3>
      <ReactMarkdown
        components={{
          h3: ({ children }) => (
            <h4 style={{ marginTop: "16px", marginBottom: "8px" }}>
              {children}
            </h4>
          ),
          p: ({ children }) => (
            <p style={{ lineHeight: "1.6", fontSize: "14px" }}>
              {children}
            </p>
          )
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

export default InsightsBox;