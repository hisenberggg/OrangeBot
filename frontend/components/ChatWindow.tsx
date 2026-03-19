"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import type { Message, ThinkingEvent } from "@/lib/types";
import ThinkingSteps from "./ThinkingSteps";
import styles from "./ChatWindow.module.css";

interface ChatWindowProps {
  messages: Message[];
  onSend: (text: string) => void;
  isLoading: boolean;
  error: string | null;
  route: string | null;
  onDismissError: () => void;
  streamingThinkingSteps: ThinkingEvent[];
  streamingContent: string;
}

export default function ChatWindow({
  messages,
  onSend,
  isLoading,
  error,
  route,
  onDismissError,
  streamingThinkingSteps,
  streamingContent,
}: ChatWindowProps) {
  const [draft, setDraft] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading, streamingContent, streamingThinkingSteps]);

  useEffect(() => {
    if (!isLoading) textareaRef.current?.focus();
  }, [isLoading]);

  const autoResize = useCallback(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, []);

  const handleSend = () => {
    const text = draft.trim();
    if (!text || isLoading) return;
    onSend(text);
    setDraft("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const empty = messages.length === 0 && !isLoading;
  const hasStreamingContent = streamingContent.length > 0;

  return (
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
        <span className={styles.headerTitle}>Chat</span>
        {route && (
          <span className={styles.routeBadge}>
            {route}
          </span>
        )}
      </header>

      {/* Messages */}
      <div className={styles.messages}>
        {empty && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <p className={styles.emptyText}>
              Ask anything about Syracuse University
            </p>
            <p className={styles.emptyHint}>
              Try: &quot;How do I drop a course?&quot;
            </p>
          </div>
        )}

        {messages.map((m) => (
          <div
            key={m.id}
            className={`${styles.bubble} ${
              m.role === "user" ? styles.user : styles.assistant
            }`}
          >
            {m.role === "assistant" && m.thinkingSteps && m.thinkingSteps.length > 0 && (
              <ThinkingSteps steps={m.thinkingSteps} streaming={false} />
            )}
            {m.role === "assistant" && (
              <span className={styles.label}>Answers</span>
            )}
            <div className={styles.bubbleContent}>
              {m.role === "assistant" ? (
                <ReactMarkdown>{m.content}</ReactMarkdown>
              ) : (
                m.content
              )}
            </div>
            <span className={styles.timestamp}>
              {new Date(m.timestamp).toLocaleTimeString(undefined, {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>
        ))}

        {isLoading && (
          <div className={`${styles.bubble} ${styles.assistant}`}>
            {streamingThinkingSteps.length > 0 && (
              <ThinkingSteps steps={streamingThinkingSteps} streaming={true} />
            )}
            <span className={styles.label}>Answers</span>
            {hasStreamingContent ? (
              <div className={styles.bubbleContent}>
                <span>{streamingContent}</span>
                <span className={styles.cursor}>|</span>
              </div>
            ) : (
              <div className={styles.thinking}>
                <span className={styles.dot} />
                <span className={styles.dot} />
                <span className={styles.dot} />
              </div>
            )}
          </div>
        )}

        {error && (
          <div className={styles.errorBox}>
            <span>{error}</span>
            <button className={styles.errorDismiss} onClick={onDismissError}>
              Dismiss
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className={styles.inputArea}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={draft}
          onChange={(e) => {
            setDraft(e.target.value);
            autoResize();
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          rows={1}
          disabled={isLoading}
        />
        <button
          className={styles.sendBtn}
          onClick={handleSend}
          disabled={!draft.trim() || isLoading}
          aria-label="Send"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path
              d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
