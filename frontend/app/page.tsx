"use client";

import { useState, useCallback, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import ChatWindow from "@/components/ChatWindow";
import { sendMessage } from "@/lib/chatClient";
import type { ChatSession, Message } from "@/lib/types";
import styles from "./page.module.css";

let _counter = 0;
function uid(): string {
  _counter += 1;
  return `${_counter}-${Math.random().toString(36).slice(2, 8)}`;
}

function createSession(): ChatSession {
  return {
    id: uid(),
    title: "New chat",
    createdAt: new Date().toISOString(),
    route: null,
    messages: [],
  };
}

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const s = createSession();
    setSessions([s]);
    setActiveId(s.id);
    setMounted(true);
  }, []);

  const activeSession =
    sessions.find((s) => s.id === activeId) ?? sessions[0] ?? null;

  const updateSession = useCallback(
    (id: string, updater: (s: ChatSession) => ChatSession) => {
      setSessions((prev) => prev.map((s) => (s.id === id ? updater(s) : s)));
    },
    []
  );

  const handleNewChat = useCallback(() => {
    const s = createSession();
    setSessions((prev) => [s, ...prev]);
    setActiveId(s.id);
    setError(null);
  }, []);

  const handleSelect = useCallback((id: string) => {
    setActiveId(id);
    setError(null);
  }, []);

  const handleSend = useCallback(
    async (text: string) => {
      if (!activeSession) return;

      const userMsg: Message = {
        id: uid(),
        role: "user",
        content: text,
        timestamp: new Date().toISOString(),
      };

      updateSession(activeSession.id, (s) => {
        const isFirst = s.messages.length === 0;
        return {
          ...s,
          title: isFirst ? text.slice(0, 48) : s.title,
          messages: [...s.messages, userMsg],
        };
      });

      setIsLoading(true);
      setError(null);

      try {
        const data = await sendMessage(text);

        const assistantMsg: Message = {
          id: uid(),
          role: "assistant",
          content: data.response,
          timestamp: new Date().toISOString(),
        };

        updateSession(activeSession.id, (s) => ({
          ...s,
          route: data.route ?? s.route,
          messages: [...s.messages, assistantMsg],
        }));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong");
      } finally {
        setIsLoading(false);
      }
    },
    [activeSession, updateSession]
  );

  if (!mounted) {
    return null;
  }

  return (
    <div className={styles.layout}>
      <Sidebar
        sessions={sessions}
        activeSessionId={activeId}
        onSelect={handleSelect}
        onNew={handleNewChat}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((c) => !c)}
      />
      <main
        className={styles.main}
        data-sidebar={sidebarCollapsed ? "collapsed" : "open"}
      >
        <ChatWindow
          messages={activeSession?.messages ?? []}
          onSend={handleSend}
          isLoading={isLoading}
          error={error}
          route={activeSession?.route ?? null}
          onDismissError={() => setError(null)}
        />
      </main>
    </div>
  );
}
