import Link from "next/link";

const tracks = [
  {
    id: "couch",
    eyebrow: "01 · FROM YOUR COUCH",
    icon: "📱",
    title: "Phone as your mic, trackpad, pointer.",
    body: "Sanketra on Android pairs with your PC over Wi-Fi. Hold-to-speak for Hindi dictation, trackpad gestures for cursor control, gyro for pointer, screen mirror for watching from across the room.",
    footer: "Live on Play Store · Closed testing",
    href: "/download#android",
    cta: "Download for Android",
  },
  {
    id: "desk",
    eyebrow: "02 · AT YOUR DESK",
    icon: "💻",
    title: "Global hotkey, desktop mic, anywhere you type.",
    body: "Native menubar app for macOS, Windows and Linux — Cmd+Shift+H in any text field dictates Hindi with your desktop's mic. Transcript history in a dedicated window. Thin client over the same PC server.",
    footer: "Coming v1.3 · For now, localhost:5000 works in any browser",
    href: "/download#ios",
    cta: "See the desktop path",
  },
] as const;

export default function Hero() {
  return (
    <section className="wrap grid gap-10 border-b border-rule pb-16 pt-12">
      <div className="flex flex-col items-center text-center">
        <p className="masthead mb-4">Sanketra · संकेतरा</p>
        <p className="deva serif-italic text-[clamp(2.4rem,6vw,4.2rem)] leading-[1.05] text-ink">
          बोलो, टाइप होगा।
        </p>
        <p className="mt-4 max-w-2xl text-lg text-muted">
          Hindi-first voice input for your PC. Local Whisper, no cloud, no account.
          Pick the flow that matches where you are right now.
        </p>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        {tracks.map((track) => (
          <article
            key={track.id}
            className="card flex flex-col gap-5 p-7 transition hover:border-saffron"
          >
            <div className="flex items-start justify-between">
              <p className="masthead text-saffron">{track.eyebrow}</p>
              <span aria-hidden className="text-4xl leading-none">
                {track.icon}
              </span>
            </div>
            <h2 className="text-2xl font-semibold leading-tight text-ink">
              {track.title}
            </h2>
            <p className="text-muted">{track.body}</p>
            <p className="font-mono text-xs uppercase tracking-[0.12em] text-faint">
              {track.footer}
            </p>
            <Link
              href={track.href}
              className="mt-auto inline-flex items-center gap-2 border-b border-rule pb-2 text-sm font-semibold uppercase tracking-[0.12em] text-ink transition hover:border-saffron hover:text-saffron"
            >
              {track.cta}
              <span aria-hidden>→</span>
            </Link>
          </article>
        ))}
      </div>
    </section>
  );
}
