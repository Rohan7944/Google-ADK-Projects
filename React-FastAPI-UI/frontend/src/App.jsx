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
    console.log("[UI] App loaded");
    console.log("[UI] Creating session for user:", USER_ID);

    fetch(`${API_BASE}/api/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: USER_ID }),
    })
      .then((res) => {
        console.log("[UI] Session API response status:", res.status);
        return res.json();
      })
      .then((data) => {
        console.log("[UI] Session created:", data.session_id);
        setSessionId(data.session_id);
      })
      .catch((err) => {
        console.error("[UI] Failed to create session", err);
      });
  }, []);

  // --------------------------------------------------
  // Send message
  // --------------------------------------------------
  const sendMessage = async () => {
    if (!input || !sessionId) {
      console.warn("[UI] Cannot send message: missing input or session");
      return;
    }

    console.log("[UI] Sending message:", input);

    setMessages((prev) => [...prev, { role: "user", text: input }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: input,
        }),
      });

      console.log("[UI] Chat API response status:", res.status);

      const data = await res.json();
      console.log("[UI] Agent reply:", data.reply);

      setMessages((prev) => [...prev, { role: "agent", text: data.reply }]);
    } catch (err) {
      console.error("[UI] Chat request failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto" }}>
      <h2>Vertex AI Agent Engine Chat</h2>

      {!sessionId && <p>Creating session…</p>}

      <div
        style={{
          border: "1px solid #ddd",
          height: 400,
          padding: 12,
          overflowY: "auto",
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              textAlign: m.role === "user" ? "right" : "left",
              marginBottom: 8,
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
