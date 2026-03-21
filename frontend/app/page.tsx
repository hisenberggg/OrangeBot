"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import ChatWindow from "@/components/ChatWindow";
import AuthModal from "@/components/AuthModal";
import {
  streamMessage,
  getStoredToken,
  clearAuthSession,
  getProfileDisplayLabel,
  listChats,
  createChat,
  getThreadMessages,
} from "@/lib/chatClient";
import type { ChatSession, Message, ThinkingEvent } from "@/lib/types";
import styles from "./page.module.css";

let _counter = 0;
function uid(): string {
  _counter += 1;
  return `${_counter}-${Math.random().toString(36).slice(2, 8)}`;
}

function serverMessagesToUi(rows: Awaited<ReturnType<typeof getThreadMessages>>): Message[] {
  const out: Message[] = [];
  for (const m of rows) {
    if (m.role !== "user" && m.role !== "assistant") continue;
    out.push({
      id: uid(),
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
      route: m.route ?? undefined,
    });
  }
  return out;
}

export default function Home() {
  const [authed, setAuthed] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [hydrating, setHydrating] = useState(true);
  const [profileDisplayName, setProfileDisplayName] = useState("Account");

  const [streamingThinkingSteps, setStreamingThinkingSteps] = useState<ThinkingEvent[]>([]);
  const [streamingContent, setStreamingContent] = useState("");

  const thinkingRef = useRef<ThinkingEvent[]>([]);
  const contentRef = useRef("");

  const loadSessionsFromServer = useCallback(async () => {
    const threads = await listChats();
    const mapped: ChatSession[] = threads.map((t) => ({
      id: t.id,
      title: t.title || "New chat",
      createdAt: t.created_at,
      updatedAt: t.updated_at,
      route: null,
      messages: [],
    }));
    setSessions(mapped);
    return mapped;
  }, []);

  const hydrateAfterAuth = useCallback(async () => {
    setHydrating(true);
    setError(null);
    try {
      let threads = await loadSessionsFromServer();
      if (threads.length === 0) {
        const t = await createChat();
        threads = [
          {
            id: t.id,
            title: t.title || "New chat",
            createdAt: t.created_at,
            updatedAt: t.updated_at,
            route: null,
            messages: [],
          },
        ];
        setSessions(threads);
      }
      const first = threads[0];
      setActiveId(first.id);
      const msgs = await getThreadMessages(first.id);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === first.id ? { ...s, messages: serverMessagesToUi(msgs) } : s
        )
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load chats");
    } finally {
      setHydrating(false);
    }
  }, [loadSessionsFromServer]);

  useEffect(() => {
    const token = getStoredToken();
    if (token) {
      setAuthed(true);
      hydrateAfterAuth();
    } else {
      setHydrating(false);
    }
    setMounted(true);
  }, [hydrateAfterAuth]);

  useEffect(() => {
    if (authed) {
      setProfileDisplayName(getProfileDisplayLabel());
    } else {
      setProfileDisplayName("Account");
    }
  }, [authed]);

  const activeSession =
    sessions.find((s) => s.id === activeId) ?? sessions[0] ?? null;

  const updateSession = useCallback(
    (id: string, updater: (s: ChatSession) => ChatSession) => {
      setSessions((prev) => prev.map((s) => (s.id === id ? updater(s) : s)));
    },
    []
  );

  const handleAuthed = useCallback(() => {
    setProfileDisplayName(getProfileDisplayLabel());
    setAuthed(true);
    hydrateAfterAuth();
  }, [hydrateAfterAuth]);

  const handleLogout = useCallback(() => {
    clearAuthSession();
    setProfileDisplayName("Account");
    setAuthed(false);
    setSessions([]);
    setActiveId("");
    setError(null);
  }, []);

  const handleNewChat = useCallback(async () => {
    if (!authed) return;
    setError(null);
    try {
      const t = await createChat();
      const s: ChatSession = {
        id: t.id,
        title: t.title || "New chat",
        createdAt: t.created_at,
        updatedAt: t.updated_at,
        route: null,
        messages: [],
      };
      setSessions((prev) => [s, ...prev]);
      setActiveId(s.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create chat");
    }
  }, [authed]);

  const handleSelect = useCallback(
    async (id: string) => {
      setActiveId(id);
      setError(null);
      if (!authed) return;
      try {
        const msgs = await getThreadMessages(id);
        setSessions((prev) =>
          prev.map((s) =>
            s.id === id ? { ...s, messages: serverMessagesToUi(msgs) } : s
          )
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not load messages");
      }
    },
    [authed]
  );

  const handleSend = useCallback(
    async (text: string) => {
      if (!activeSession || !authed) return;

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
      setStreamingThinkingSteps([]);
      setStreamingContent("");
      thinkingRef.current = [];
      contentRef.current = "";

      const sessionId = activeSession.id;

      try {
        await streamMessage(text, sessionId, {
          onThinking: (event) => {
            thinkingRef.current = [...thinkingRef.current, event];
            setStreamingThinkingSteps([...thinkingRef.current]);
          },
          onDelta: (content) => {
            contentRef.current += content;
            setStreamingContent(contentRef.current);
          },
          onError: (message) => {
            setError(message);
          },
          onDone: (response, route) => {
            const assistantMsg: Message = {
              id: uid(),
              role: "assistant",
              content: response,
              timestamp: new Date().toISOString(),
              thinkingSteps: thinkingRef.current,
              route: route ?? undefined,
            };

            updateSession(sessionId, (s) => ({
              ...s,
              route: route ?? s.route,
              updatedAt: new Date().toISOString(),
              messages: [...s.messages, assistantMsg],
            }));

            setStreamingThinkingSteps([]);
            setStreamingContent("");
            thinkingRef.current = [];
            contentRef.current = "";
          },
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong");
      } finally {
        setIsLoading(false);
      }
    },
    [activeSession, authed, updateSession]
  );

  if (!mounted) {
    return null;
  }

  if (!authed) {
    return <AuthModal onAuthed={handleAuthed} />;
  }

  if (hydrating || !activeSession) {
    return (
      <div className={styles.layout}>
        <p style={{ margin: "auto", opacity: 0.8 }}>Loading chats…</p>
      </div>
    );
  }

  return (
    <div className={styles.layout}>
      <Sidebar
        sessions={sessions}
        activeSessionId={activeId}
        onSelect={handleSelect}
        onNew={handleNewChat}
        onLogout={handleLogout}
        displayName={profileDisplayName}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((c) => !c)}
      />
      <main
        className={styles.main}
        data-sidebar={sidebarCollapsed ? "collapsed" : "open"}
      >
        <ChatWindow
          messages={activeSession.messages}
          onSend={handleSend}
          isLoading={isLoading}
          error={error}
          route={activeSession.route ?? null}
          onDismissError={() => setError(null)}
          streamingThinkingSteps={streamingThinkingSteps}
          streamingContent={streamingContent}
        />
      </main>
    </div>
  );
}
