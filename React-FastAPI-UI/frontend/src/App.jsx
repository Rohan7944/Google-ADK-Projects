import { useEffect, useState } from "react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const USER_ID = process.env.REACT_APP_USER_ID;

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);

  // -------------------------------------------------
  // Create session on load
  // -------------------------------------------------
  useEffect(() => {
    async function createSession() {
      console.log("Creating session...");
      const res = await fetch(`${BACKEND_URL}/api/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: USER_ID }),
      });

      const data = await res.json();
      console.log("Session created:", data.session_id);
      setSessionId(data.session_id);
    }

    createSession();
  }, []);

  // -------------------------------------------------
  // Send chat message
  // -------------------------------------------------
  async function sendMessage() {
    if (!message.trim()) return;

    setChat([...chat, { role: "user", text: message }]);

    const res = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: message,
      }),
    });

    const data = await res.json();
    setChat((prev) => [
      ...prev,
      { role: "agent", text: data.response },
    ]);

    setMessage("");
  }

  return (
    <div style={{ maxWidth: 600, margin: "40px auto" }}>
      <h2>Vertex AI Agent Engine Chat</h2>

      <div style={{ border: "1px solid #ccc", padding: 10, minHeight: 300 }}>
        {chat.map((c, i) => (
          <div key={i}>
            <strong>{c.role}:</strong> {c.text}
          </div>
        ))}
      </div>

      <input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message..."
        style={{ width: "80%", marginTop: 10 }}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

export default App;