import Hero from "@/components/Hero";

const values = [
  ["Hindi typing stops being punishment", "Devanagari keyboard gymnastics disappear. Speak Hindi, Hinglish, or English and let the PC type where focus already is."],
  ["Your voice stays on your Wi-Fi", "Sanketra uses the PC you already own for speech-to-text. No cloud STT bill, no account, no tracking pixels."],
  ["Phone proximity beats laptop mics", "In noisy rooms, the mic near your mouth wins. Phone-as-mic is not a gimmick; it is better signal."],
  ["More than dictation", "Trackpad, screen mirror, PC audio, and dictation sit on one LAN pair-code flow instead of four separate utilities."]
] as const;

export default function HomePage() {
  return (
    <main>
      <Hero />
      <section className="wrap flex flex-col items-center border-b border-rule py-24 text-center">
        <p className="masthead mb-6 text-saffron">The brand line</p>
        <p className="deva text-[clamp(4.4rem,14vw,10rem)] font-normal leading-[0.92] tracking-normal text-saffron">
          PC का कान।
        </p>
        <p className="serif-italic mt-6 max-w-2xl text-[clamp(1.4rem,3vw,2rem)] leading-snug text-muted">
          Voice as a universal input layer for your PC. Bolo, type hoga.
        </p>
      </section>
      <section className="wrap grid gap-10 border-b border-rule py-20 lg:grid-cols-[0.7fr_1fr]">
        <div>
          <p className="masthead mb-4">Why Sanketra</p>
          <h2 className="section-title">Local voice input for Indian PCs.</h2>
        </div>
        <div className="grid gap-5 sm:grid-cols-2">
          {values.map(([title, body]) => (
            <article key={title} className="border-t border-rule pt-5">
              <h3 className="text-xl font-semibold">{title}</h3>
              <p className="mt-3 text-muted">{body}</p>
            </article>
          ))}
        </div>
      </section>
      <section className="wrap pb-16">
        <p className="masthead mb-5 text-center">Demo</p>
        <div className="aspect-video overflow-hidden rounded-md border border-rule bg-black">
          <video
            src="/demo.mp4"
            autoPlay
            loop
            muted
            playsInline
            preload="metadata"
            className="h-full w-full object-cover"
          />
        </div>
      </section>
    </main>
  );
}
