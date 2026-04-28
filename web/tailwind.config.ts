import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx,mdx}",
    "./components/**/*.{ts,tsx}",
    "./content/**/*.{mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        muted: "var(--ink-muted)",
        faint: "var(--ink-faint)",
        paper: "var(--paper)",
        surface: "var(--surface)",
        rule: "var(--rule)",
        saffron: "var(--accent-warm)",
        green: "var(--accent)"
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
        deva: ["var(--font-deva)"],
        serif: ["var(--font-serif)"]
      }
    }
  },
  plugins: [typography]
};

export default config;
