import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
const USER_ID = import.meta.env.VITE_USER_ID;

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  // --------------------------------------------------
  // Create session on load
  // --------------------------------------------------
  useEffect(() => {
    fetch(`${API_BASE}/api/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID }),
    })
      .then((res) => res.json())
      .then((data) => setSessionId(data.session_id));
  }, []);

  // --------------------------------------------------
  // Send message
  // --------------------------------------------------
  const sendMessage = async () => {
    if (!input || !sessionId) return;

    setMessages((prev) => [...prev, { role: "user", text: input }]);
    setInput("");
    setLoading(true);

    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: input,
      }),
    });

    const data = await res.json();
    setMessages((prev) => [...prev, { role: "agent", text: data.reply }]);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto" }}>
      <h2>Vertex AI Agent Engine Chat</h2>

      {!sessionId && <p>Creating session…</p>}

      <div style={{
        border: "1px solid #ddd",
        height: 400,
        padding: 12,
        overflowY: "auto"
      }}>
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              textAlign: m.role === "user" ? "right" : "left",
              marginBottom: 8
            }}
          >
            <b>{m.role}:</b> {m.text}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          style={{ flex: 1 }}
          placeholder="Type a message…"
        />
        <button onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}
