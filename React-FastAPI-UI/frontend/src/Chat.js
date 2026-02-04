import { useState } from "react";
import { sendChatMessage } from "./api";

export default function Chat() {
  const [agentId, setAgentId] = useState("");
  const [projectId, setProjectId] = useState("");
  const [location, setLocation] = useState("us-central1");
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!message.trim()) return;

    const userText = message;
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
    setMessage("");

    try {
      const data = await sendChatMessage({
        user_message: userText,
        agent_id: agentId,
        project_id: projectId,
        location: location,
      });

      setMessages((prev) => [
        ...prev,
        { role: "agent", text: data.response_text },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "agent", text: "⚠️ Error fetching response." },
      ]);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h2 style={styles.title}>Vertex AI Agent Chat</h2>

        {/* Config section */}
        <div style={styles.config}>
          <input
            style={styles.input}
            placeholder="Project ID"
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
          />
          <input
            style={styles.input}
            placeholder="Location (e.g. us-central1)"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          />
          <input
            style={styles.input}
            placeholder="Agent ID"
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
          />
        </div>

        {/* Chat window */}
        <div style={styles.chatWindow}>
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                ...styles.message,
                ...(m.role === "user"
                  ? styles.userMessage
                  : styles.agentMessage),
              }}
            >
              {m.text}
            </div>
          ))}
        </div>

        {/* Input */}
        <div style={styles.inputRow}>
          <input
            style={styles.chatInput}
            placeholder="Type your message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button style={styles.sendButton} onClick={sendMessage}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Styles ---------- */

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #f5f7fa, #e4ecf7)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    fontFamily: "Arial, sans-serif",
  },
  container: {
    width: "100%",
    maxWidth: 800,
    background: "#ffffff",
    borderRadius: 12,
    boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
    padding: 20,
    display: "flex",
    flexDirection: "column",
  },
  title: {
    marginBottom: 10,
    textAlign: "center",
  },
  config: {
    display: "flex",
    gap: 8,
    marginBottom: 10,
  },
  input: {
    flex: 1,
    padding: 8,
    borderRadius: 6,
    border: "1px solid #ccc",
  },
  chatWindow: {
    flex: 1,
    overflowY: "auto",
    background: "#f9fafb",
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  message: {
    maxWidth: "70%",
    padding: "10px 14px",
    borderRadius: 16,
    lineHeight: 1.4,
    wordBreak: "break-word",
  },
  userMessage: {
    alignSelf: "flex-end",
    background: "#0078ff",
    color: "#ffffff",
    borderBottomRightRadius: 4,
  },
  agentMessage: {
    alignSelf: "flex-start",
    background: "#e5e7eb",
    color: "#000000",
    borderBottomLeftRadius: 4,
  },
  inputRow: {
    display: "flex",
    gap: 8,
  },
  chatInput: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    border: "1px solid #ccc",
    fontSize: 14,
  },
  sendButton: {
    padding: "0 16px",
    borderRadius: 8,
    border: "none",
    background: "#0078ff",
    color: "#ffffff",
    fontWeight: "bold",
    cursor: "pointer",
  },
};