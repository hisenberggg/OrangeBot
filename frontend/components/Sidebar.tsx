"use client";

import type { ChatSession } from "@/lib/types";
import styles from "./Sidebar.module.css";

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelect,
  onNew,
  collapsed,
  onToggle,
}: SidebarProps) {
  return (
    <>
      <button
        className={styles.toggleBtn}
        onClick={onToggle}
        aria-label={collapsed ? "Open sidebar" : "Close sidebar"}
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <rect y="3" width="20" height="2" rx="1" fill="currentColor" />
          <rect y="9" width="20" height="2" rx="1" fill="currentColor" />
          <rect y="15" width="20" height="2" rx="1" fill="currentColor" />
        </svg>
      </button>

      <aside className={`${styles.sidebar} ${collapsed ? styles.collapsed : ""}`}>
        <div className={styles.header}>
          <h1 className={styles.title}>Syracuse Answers</h1>
          <button className={styles.closeBtn} onClick={onToggle} aria-label="Close sidebar">
            &times;
          </button>
        </div>

        <button className={styles.newChat} onClick={onNew}>
          + New Chat
        </button>

        <nav className={styles.sessions}>
          {sessions.map((s) => (
            <button
              key={s.id}
              className={`${styles.sessionItem} ${
                s.id === activeSessionId ? styles.active : ""
              }`}
              onClick={() => onSelect(s.id)}
            >
              <span className={styles.sessionTitle}>{s.title}</span>
              <span className={styles.sessionDate}>
                {new Date(s.createdAt).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                })}
              </span>
            </button>
          ))}
        </nav>
      </aside>

      {!collapsed && (
        <div className={styles.overlay} onClick={onToggle} />
      )}
    </>
  );
}
