import { useState } from "react";
import ReactMarkdown from "react-markdown";

function Chatbot({ metrics }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input) return;

    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);

    const response = await fetch("http://127.0.0.1:8000/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: input,
        metrics,
        chat_history: messages
      })
    });

    const data = await response.json();

    setMessages([...newMessages, { role: "assistant", content: data.response }]);
    setInput("");
  };

  return (
    <div className="metric-card" style={{ marginTop: "40px" }}>
      <h3>Ask SmartSpend</h3>

      <div style={{ maxHeight: "300px", overflowY: "auto", marginBottom: "10px" }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: "8px" }}>
            <strong>{msg.role === "user" ? "You" : "SmartSpend"}:</strong>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        ))}
      </div>

      <div className="chat-input-container">
      <input
        className="chat-input"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask about your spending..."
      />
      <button className="chat-button" onClick={sendMessage}>
        ➤
      </button>
    </div>
    </div>
  );
}

export default Chatbot;