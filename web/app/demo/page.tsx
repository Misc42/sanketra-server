import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Demo",
  description: "Watch Sanketra dictation, trackpad, and screen mirror demos."
};

const gifs = [
  ["Dictation", "/gifs/dictation.gif", "Speak Hindi or Hinglish into your phone and type into the active PC field."],
  ["Trackpad", "/gifs/trackpad.gif", "Use the phone as a precise LAN trackpad when the laptop is across the room."],
  ["Screen Mirror", "/gifs/screen-mirror.gif", "See the PC screen on your phone and control it without an internet relay."]
] as const;

export default function DemoPage() {
  return (
    <main className="wrap py-16">
      <p className="masthead mb-4">Demo</p>
      <h1 className="section-title">Bolo, move karo, mirror karo.</h1>
      <p className="mt-6 max-w-3xl text-lg text-muted">
        One PC server, one phone, one local Wi-Fi loop. The media files are referenced
        here so Tanay can drop final captures into `public/` before launch.
      </p>
      {/* Hero reel — 30s Remotion-rendered Apple-style piece at public/demo.mp4.
          Muted/autoplay/loop/playsInline so it acts like a motion poster. */}
      <div className="mt-12 overflow-hidden rounded-md border border-rule bg-black/40">
        <video
          className="aspect-video w-full"
          src="/demo.mp4"
          autoPlay
          loop
          muted
          playsInline
          preload="metadata"
        />
      </div>
      <section className="mt-10 grid gap-5 md:grid-cols-3">
        {gifs.map(([label, , caption]) => (
          <article key={label} className="card overflow-hidden">
            <div className="flex aspect-video w-full items-center justify-center bg-black/40 text-center">
              <span className="masthead text-saffron">{label}</span>
            </div>
            <div className="p-5">
              <h2 className="text-xl font-semibold">{label}</h2>
              <p className="mt-2 text-sm text-muted">{caption}</p>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
