"use client";

import { useState } from "react";
import { login, setAuthSession, signup } from "@/lib/chatClient";
import styles from "./AuthModal.module.css";

type Mode = "login" | "signup";

interface AuthModalProps {
  onAuthed: () => void;
}

function profileFromTokenResponse(res: {
  first_name?: string | null;
  last_name?: string | null;
}) {
  if (res.first_name == null && res.last_name == null) return null;
  return {
    firstName: res.first_name ?? "",
    lastName: res.last_name ?? "",
  };
}

export default function AuthModal({ onAuthed }: AuthModalProps) {
  const [mode, setMode] = useState<Mode>("login");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === "signup") {
        const res = await signup(
          email.trim(),
          password,
          firstName.trim(),
          lastName.trim()
        );
        setAuthSession(
          res.access_token,
          res.user_id,
          profileFromTokenResponse(res) ?? {
            firstName: firstName.trim(),
            lastName: lastName.trim(),
          }
        );
      } else {
        const res = await login(email.trim(), password);
        setAuthSession(
          res.access_token,
          res.user_id,
          profileFromTokenResponse(res)
        );
      }
      onAuthed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <div className={styles.card}>
        <h2 id="auth-title" className={styles.title}>
          Syracuse Answers
        </h2>
        <p className={styles.sub}>Sign in to load your saved chats.</p>

        <div className={styles.tabs}>
          <button
            type="button"
            className={mode === "login" ? styles.tabActive : styles.tab}
            onClick={() => {
              setMode("login");
              setError(null);
            }}
          >
            Log in
          </button>
          <button
            type="button"
            className={mode === "signup" ? styles.tabActive : styles.tab}
            onClick={() => {
              setMode("signup");
              setError(null);
            }}
          >
            Sign up
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {mode === "signup" ? (
            <>
              <div className={styles.field}>
                <label htmlFor="auth-first-name">First name</label>
                <input
                  id="auth-first-name"
                  type="text"
                  autoComplete="given-name"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  required
                  minLength={1}
                />
              </div>
              <div className={styles.field}>
                <label htmlFor="auth-last-name">Last name</label>
                <input
                  id="auth-last-name"
                  type="text"
                  autoComplete="family-name"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  required
                  minLength={1}
                />
              </div>
            </>
          ) : null}
          <div className={styles.field}>
            <label htmlFor="auth-email">Email</label>
            <input
              id="auth-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className={styles.field}>
            <label htmlFor="auth-password">Password</label>
            <input
              id="auth-password"
              type="password"
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={mode === "signup" ? 6 : 1}
            />
          </div>
          <button type="submit" className={styles.submit} disabled={loading}>
            {loading ? "Please wait…" : mode === "signup" ? "Create account" : "Log in"}
          </button>
        </form>

        {error ? <div className={styles.error}>{error}</div> : null}
      </div>
    </div>
  );
}
