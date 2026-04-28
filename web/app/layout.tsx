import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Instrument_Serif, Tiro_Devanagari_Hindi } from "next/font/google";
import "../styles/globals.css";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

const plexSans = IBM_Plex_Sans({
  subsets: ["latin", "latin-ext"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap"
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin", "latin-ext"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap"
});

const tiroDeva = Tiro_Devanagari_Hindi({
  subsets: ["devanagari", "latin"],
  weight: "400",
  variable: "--font-deva",
  display: "swap"
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  variable: "--font-serif",
  display: "swap"
});

export const metadata: Metadata = {
  metadataBase: new URL("https://sanketra.app"),
  title: {
    default: "Sanketra — PC का कान",
    template: "%s — Sanketra"
  },
  description: "Voice as a universal input layer for your PC. Hindi-first, LAN-only, zero cloud speech-to-text.",
  openGraph: {
    title: "Sanketra — PC का कान",
    description: "Voice as a universal input layer for your PC.",
    url: "https://sanketra.app",
    siteName: "Sanketra",
    images: [{ url: "/og-image.png", width: 1200, height: 630 }],
    locale: "en_IN",
    type: "website"
  },
  twitter: {
    card: "summary_large_image",
    title: "Sanketra — PC का कान",
    description: "Voice as a universal input layer for your PC.",
    images: ["/og-image.png"]
  },
  icons: {
    icon: "/icon.svg"
  }
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en-IN"
      className={`${plexSans.variable} ${plexMono.variable} ${tiroDeva.variable} ${instrumentSerif.variable}`}
    >
      <body>
        <Nav />
        {children}
        <Footer />
      </body>
    </html>
  );
}
