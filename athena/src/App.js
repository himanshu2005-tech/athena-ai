import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #ffffff;
    --bg-sidebar: #f7f7f8;
    --bg-input: #f4f4f5;
    --border: #e5e5e7;
    --text: #111111;
    --text-secondary: #666666;
    --text-muted: #aaaaaa;
    --accent: #2563eb;
    --user-bubble: #2563eb;
    --user-text: #ffffff;
    --ai-bubble: #f4f4f5;
    --ai-text: #111111;
    --radius: 18px;
    --font: 'Inter', -apple-system, sans-serif;
  }

  html, body, #root { height: 100%; background: var(--bg); }
  body { font-family: var(--font); color: var(--text); overflow: hidden; -webkit-font-smoothing: antialiased; }

  .shell { display: grid; grid-template-columns: 240px 1fr; height: 100vh; }

  /* SIDEBAR */
  .sidebar {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 16px 12px;
    gap: 4px;
    overflow: hidden;
  }
  .sidebar-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 10px 10px;
  }
  .history-scroll { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
  .history-scroll::-webkit-scrollbar { width: 0; }
  .history-item {
    padding: 9px 12px;
    border-radius: 10px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background 0.1s;
  }
  .history-item:hover { background: var(--border); color: var(--text); }
  .history-empty { font-size: 0.78rem; color: var(--text-muted); padding: 10px 12px; }

  /* OPTIONS SECTION */
  .options-section {
    border-top: 1px solid var(--border);
    padding: 14px 4px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .option-btn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 12px;
    border-radius: 10px;
    border: 1px solid transparent;
    background: none;
    cursor: pointer;
    font-family: var(--font);
    font-size: 0.82rem;
    color: var(--text-secondary);
    transition: all 0.15s;
    text-align: left;
    width: 100%;
    gap: 8px;
  }
  .option-btn:hover { background: var(--border); color: var(--text); }
  .option-btn.active {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: var(--accent);
  }
  .option-label { flex: 1; line-height: 1.3; }
  .option-label small { display: block; font-size: 0.7rem; color: var(--text-muted); margin-top: 1px; font-weight: 400; }
  .option-btn.active .option-label small { color: #93c5fd; }
  .option-check {
    width: 18px; height: 18px;
    border-radius: 5px;
    border: 1.5px solid var(--border);
    background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem;
    flex-shrink: 0;
    transition: all 0.15s;
    color: transparent;
  }
  .option-btn.active .option-check {
    background: var(--accent);
    border-color: var(--accent);
    color: white;
  }

  .clear-btn {
    padding: 9px 12px;
    border-radius: 10px;
    background: none;
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-family: var(--font);
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.1s;
    text-align: left;
    margin-top: 4px;
  }
  .clear-btn:hover { background: var(--border); color: var(--text-secondary); }

  /* MAIN AREA */
  .main { display: flex; flex-direction: column; background: var(--bg); overflow: hidden; }

  .topbar {
    padding: 14px 28px;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; }
  .topbar-pills { margin-left: auto; display: flex; gap: 6px; }
  .topbar-pill {
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.68rem;
    font-weight: 500;
    background: #eff6ff;
    color: var(--accent);
    border: 1px solid #bfdbfe;
    letter-spacing: 0.02em;
  }

  /* CHAT FEED Matrix */
  .feed { flex: 1; overflow-y: auto; padding: 24px 0; display: flex; flex-direction: column; gap: 12px; }
  .feed::-webkit-scrollbar { width: 0; }

  .welcome {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; gap: 10px; text-align: center; padding: 40px;
    animation: fade-up 0.4s ease;
  }
  .welcome h2 { font-size: 1.4rem; font-weight: 600; }
  .welcome p { font-size: 0.85rem; color: var(--text-secondary); max-width: 320px; line-height: 1.7; }

  .msg { display: flex; padding: 4px 28px; animation: fade-up 0.2s ease; width: 100%; }
  .msg.user { justify-content: flex-end; }
  .msg.ai   { justify-content: flex-start; }

  @keyframes fade-up { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

  .bubble {
    max-width: 75%;
    padding: 11px 16px;
    border-radius: var(--radius);
    font-size: 0.875rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .bubble.user { background: var(--user-bubble); color: var(--user-text); border-bottom-right-radius: 4px; }
  .bubble.ai   { background: var(--ai-bubble);   color: var(--ai-text);   border-bottom-left-radius: 4px; }
  .bubble.streaming::after { content: '▋'; color: var(--text-muted); animation: blink 0.8s step-end infinite; }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }

  /* THINKING PROCESS WORKSPACE */
  .thought-box {
    width: 75%;
    background: transparent;
    border-left: 2px solid var(--border);
    padding-left: 14px;
    margin: 6px 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .thought-trigger {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
    font-weight: 500;
  }
  .thought-arrow {
    font-size: 0.6rem;
    transition: transform 0.15s ease;
    transform: rotate(0deg);
    color: var(--text-muted);
  }
  .thought-arrow.expanded { transform: rotate(90deg); }
  .thought-content {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.65;
    font-style: italic;
    padding: 2px 0 4px;
    white-space: pre-wrap;
  }

  .bubble code {
    background: rgba(0,0,0,0.06); border-radius: 4px; padding: 1px 5px;
    font-family: 'SF Mono','Fira Code',monospace; font-size: 0.82em;
  }

  .code-block { margin: 8px 0; border-radius: 10px; border: 1px solid var(--border); overflow: hidden; background: #f8f8f9; }
  .code-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 12px; border-bottom: 1px solid var(--border);
    font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.06em; background: #f4f4f5;
  }
  .copy-btn {
    background: none; border: none; color: var(--text-muted); font-family: var(--font);
    font-size: 0.68rem; cursor: pointer; padding: 2px 6px; border-radius: 4px; transition: all 0.1s;
  }
  .copy-btn:hover { background: var(--border); color: var(--text); }
  .copy-btn.copied { color: #22c55e; }
  pre code {
    display: block; padding: 12px 14px; font-size: 0.78rem; line-height: 1.65;
    overflow-x: auto; color: var(--text); background: none; border: none;
    font-family: 'SF Mono','Fira Code',monospace;
  }

  .typing { display: flex; align-items: center; gap: 4px; padding: 12px 16px; background: var(--ai-bubble); border-radius: var(--radius); border-bottom-left-radius: 4px; width: fit-content; }
  .typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); animation: bounce 1.2s ease infinite; }
  .typing span:nth-child(2) { animation-delay: 0.15s; }
  .typing span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce { 0%,60%,100% { transform:translateY(0); } 30% { transform:translateY(-4px); } }

  .sources { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
  .source-chip {
    padding: 2px 10px; border-radius: 100px; border: 1px solid var(--border);
    background: var(--bg); font-size: 0.68rem; color: var(--text-secondary);
    text-decoration: none; max-width: 200px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; transition: border-color 0.1s;
  }
  .source-chip:hover { border-color: var(--accent); color: var(--accent); }

  /* INPUT BAR */
  .input-bar { padding: 16px 28px 22px; border-top: 1px solid var(--border); }
  .input-wrap {
    display: flex; align-items: flex-end; gap: 8px;
    background: var(--bg-input); border: 1px solid var(--border);
    border-radius: 14px; padding: 10px 12px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .input-wrap:focus-within { border-color: #93b4fd; box-shadow: 0 0 0 3px rgba(37,99,235,0.08); }
  .input-wrap textarea {
    flex: 1; background: none; border: none; outline: none;
    color: var(--text); font-family: var(--font); font-size: 0.875rem;
    line-height: 1.6; resize: none; min-height: 22px; max-height: 140px; overflow-y: auto;
  }
  .input-wrap textarea::placeholder { color: var(--text-muted); }
  .send-btn {
    width: 32px; height: 32px; border-radius: 8px; border: none;
    background: var(--accent); color: white; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; transition: opacity 0.1s; flex-shrink: 0;
  }
  .send-btn:hover:not(:disabled) { opacity: 0.85; }
  .send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .send-btn.stop { background: #ef4444; }

  .error-bar {
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px;
    padding: 8px 14px; font-size: 0.78rem; color: #dc2626;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }
`;

// ─── MODEL REASONING & STRUCTURE PARSER ──────────────────────────────────────
function parseMessageContent(rawText) {
  let thought = "";
  let response = rawText;

  if (rawText.includes("<think>")) {
    const parts = rawText.split("<think>");
    if (parts[1].includes("</think>")) {
      const subParts = parts[1].split("</think>");
      thought = subParts[0];
      response = parts[0] + subParts[1];
    } else {
      thought = parts[1];
      response = parts[0];
    }
  }

  const blocks = [];
  const regex = new RegExp("```(\\w*)\\n([\\s\\S]*?)```", "g");
  let lastIdx = 0, match;

  while ((match = regex.exec(response)) !== null) {
    if (match.index > lastIdx) blocks.push({ type: "text", content: response.slice(lastIdx, match.index) });
    blocks.push({ type: "code", lang: match[1] || "code", content: match[2] });
    lastIdx = match.index + match[0].length;
  }
  if (lastIdx < response.length) blocks.push({ type: "text", content: response.slice(lastIdx) });

  return { thought: thought.trim(), blocks };
}

function CodeBlock({ lang, content }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="code-block">
      <div className="code-header">
        <span>{lang}</span>
        <button className="copy-btn" onClick={copy}>{copied ? "✓ Copied" : "Copy"}</button>
      </div>
      <pre><code>{content}</code></pre>
    </div>
  );
}

function ThoughtBlock({ text, isLive }) {
  const [isOpen, setIsOpen] = useState(true);
  if (!text && !isLive) return null;

  return (
    <div className="thought-box">
      <div className="thought-trigger" onClick={() => setIsOpen(!isOpen)}>
        <span className={`thought-arrow ${isOpen ? "expanded" : ""}`}>▶</span>
        <span>{isLive ? "Thinking Process..." : "Thought Process"}</span>
      </div>
      {isOpen && <div className="thought-content">{text || "Analyzing execution graph parameters..."}</div>}
    </div>
  );
}

function MessageBubble({ rawContent, isUser, isStreaming }) {
  const { thought, blocks } = parseMessageContent(rawContent);

  return (
    <div style={{ width: "100%", display: "flex", flexDirection: "column" }}>
      {!isUser && (thought || (isStreaming && !blocks.length)) && (
        <ThoughtBlock text={thought} isLive={isStreaming && !blocks.length} />
      )}
      
      {(blocks.length > 0 || isUser) && (
        <div className={`bubble ${isUser ? "user" : "ai"}${isStreaming && !thought ? " streaming" : ""}`}>
          {blocks.map((b, i) => 
            b.type === "code" 
              ? <CodeBlock key={i} lang={b.lang} content={b.content} />
              : <span key={i} dangerouslySetInnerHTML={{
                  __html: b.content
                    .replace(/`([^`]+)`/g, "<code>$1</code>")
                    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
                }} />
          )}
        </div>
      )}
    </div>
  );
}

// ─── MAIN SYSTEM COMPONENT ───────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [useStepByStep, setUseStepByStep] = useState(false);
  const [useDeepSearch, setUseDeepSearch] = useState(false);

  const feedRef = useRef(null);
  const textareaRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [messages]);

  const handleInput = (e) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); }
  };

  const addMessage = (role, content, meta = {}) => {
    const id = Date.now() + Math.random();
    setMessages(prev => [...prev, { id, role, content, ...meta }]);
    return id;
  };

  const updateMessage = (id, patch) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, ...patch } : m));
  };

  const sendQuery = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setError(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    addMessage("user", q);
    setLoading(true);

    const aiId = addMessage("ai", "", { streaming: true, typing: true });
    const convHistory = history.slice(-6);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          query: q,
          use_cot: useStepByStep,
          use_flare: useDeepSearch,
          retrieval_k: 6,
          max_search_results: 5,
          conversation_history: convHistory,
        }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";
      let buffer = "";
      let firstToken = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === "token") {
              if (firstToken) { firstToken = false; updateMessage(aiId, { typing: false }); }
              accumulated += evt.content;
              updateMessage(aiId, { content: accumulated, streaming: true });
            } else if (evt.type === "meta") {
              updateMessage(aiId, { streaming: false, content: accumulated, sources: evt.sources || [] });
              setHistory(prev => [...prev, { role: "user", content: q }, { role: "assistant", content: accumulated }].slice(-10));
            } else if (evt.type === "error") {
              throw new Error(evt.message);
            }
          } catch (e) { if (e.message?.startsWith("{")) throw e; }
        }
      }

      updateMessage(aiId, { streaming: false, typing: false });

    } catch (err) {
      if (err.name !== "AbortError") {
        setError("Could not connect to the server.");
        updateMessage(aiId, { streaming: false, typing: false, content: "Something went wrong. Please try again." });
      }
    } finally {
      setLoading(false);
    }
  };

  const clearChat = async () => {
    try { await fetch(`${API_BASE}/clear`, { method: "POST" }); } catch {}
    setMessages([]);
    setHistory([]);
  };

  const interrupt = () => { if (abortRef.current) abortRef.current.abort(); };
  const userMessages = messages.filter(m => m.role === "user");
  const activePills = [useStepByStep && "Step-by-step", useDeepSearch && "Deep search"].filter(Boolean);

  return (
    <>
      <style>{css}</style>
      <div className="shell">

        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-title">Recent</div>
          <div className="history-scroll">
            {userMessages.length === 0
              ? <div className="history-empty">No conversations yet</div>
              : userMessages.map(m => (
                <div key={m.id} className="history-item">
                  {m.content.slice(0, 52)}{m.content.length > 52 ? "…" : ""}
                </div>
              ))
            }
          </div>

          {/* Options */}
          <div className="options-section">
            <button className={`option-btn${useStepByStep ? " active" : ""}`} onClick={() => setUseStepByStep(v => !v)}>
              <div className="option-label">Step-by-step</div>
              <div className="option-check">{useStepByStep ? "✓" : ""}</div>
            </button>
            <button className={`option-btn${useDeepSearch ? " active" : ""}`} onClick={() => setUseDeepSearch(v => !v)}>
              <div className="option-label">Deep search</div>
              <div className="option-check">{useDeepSearch ? "✓" : ""}</div>
            </button>
          </div>

          {userMessages.length > 0 && (
            <button className="clear-btn" onClick={clearChat}>Clear chat</button>
          )}
        </aside>

        {/* Main Window */}
        <main className="main">
          <div className="topbar">
            Chat
            {activePills.length > 0 && (
              <div className="topbar-pills">
                {activePills.map(p => <span key={p} className="topbar-pill">{p}</span>)}
              </div>
            )}
          </div>

          <div className="feed" ref={feedRef}>
            {messages.length === 0 ? (
              <div className="welcome">
                <p style={{fontSize: 20, color: 'black', fontWeight: 500}}>Athena</p>
              </div>
            ) : (
              messages.map(msg => (
                <div key={msg.id}>
                  <div className={`msg ${msg.role === "user" ? "user" : "ai"}`}>
                    {msg.typing ? (
                      <div className="typing"><span /><span /><span /></div>
                    ) : (
                      <MessageBubble 
                        rawContent={msg.content} 
                        isUser={msg.role === "user"} 
                        isStreaming={msg.streaming} 
                      />
                    )}
                  </div>
                  {msg.sources?.length > 0 && (
                    <div className="sources" style={{ padding: "0 28px" }}>
                      {msg.sources.map((src, i) => {
                        try {
                          return <a key={i} href={src} target="_blank" rel="noopener noreferrer" className="source-chip">{new URL(src).hostname}</a>;
                        } catch { return null; }
                      })}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          <div className="input-bar">
            {error && (
              <div className="error-bar">
                {error}
                <button onClick={() => setError(null)} style={{ marginLeft: "auto", background: "none", border: "none", color: "#dc2626", cursor: "pointer" }}>✕</button>
              </div>
            )}
            <div className="input-wrap">
              <textarea
                ref={textareaRef}
                placeholder="Ask Athena..."
                value={input}
                onInput={handleInput}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={loading && !abortRef.current}
              />
              {loading
                ? <button className="send-btn stop" onClick={interrupt}>■</button>
                : <button className="send-btn" onClick={sendQuery} disabled={!input.trim()}>↑</button>
              }
            </div>
          </div>
        </main>

      </div>
    </>
  );
} 