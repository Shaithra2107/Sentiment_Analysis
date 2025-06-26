import React, { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();

      if (data.reply) {
        // Store sentiment separately in the bot message
        const botMessage = { 
          sender: "bot", 
          text: data.reply,
          sentiment: data.sentiment
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: "Sorry, something went wrong." },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Failed to connect to the server." },
      ]);
    }

    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Sentiment Chatbot 🤖</h1>
      <div className="chat-box">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`chat-message ${msg.sender === "user" ? "user" : "bot"}`}
          >
            <strong>{msg.sender === "user" ? "You" : "Bot"}:</strong> {msg.text}
            {msg.sentiment && (
              <div
                className={`sentiment ${
                  msg.sentiment.toLowerCase().includes("positive")
                    ? "positive"
                    : msg.sentiment.toLowerCase().includes("negative")
                    ? "negative"
                    : "neutral"
                }`}
              >
                Sentiment: {msg.sentiment}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="chat-message bot">Bot is typing...</div>}
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          autoComplete="off"
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
