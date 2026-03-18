import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Syracuse Answers Chat",
  description: "Ask questions about Syracuse University policies and procedures",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
