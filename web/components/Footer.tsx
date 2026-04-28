import Link from "next/link";

const links = [
  ["Privacy", "/privacy"],
  ["Terms", "/terms"],
  ["Refunds", "/refund-policy"],
  ["Changelog", "/changelog"],
  ["GitHub", "https://github.com/Misc42/sanketra"],
  ["X", "https://x.com/tanaymisra"]
] as const;

export default function Footer() {
  return (
    <footer className="wrap mt-28 flex flex-col gap-5 border-t border-rule py-10 text-sm text-muted md:flex-row md:items-center md:justify-between">
      <nav className="flex flex-wrap gap-5 font-mono text-[0.72rem] uppercase tracking-[0.14em]">
        {links.map(([label, href]) => (
          <Link key={href} href={href} className="transition hover:text-saffron">
            {label}
          </Link>
        ))}
      </nav>
      <p className="serif-italic text-lg text-ink">
        A{" "}
        <a
          href="https://misc42labs.vercel.app"
          target="_blank"
          rel="noreferrer"
          className="text-ink hover:text-saffron transition underline decoration-rule underline-offset-4"
        >
          Misc42 Labs
        </a>{" "}
        product · made by Tanay Misra
      </p>
    </footer>
  );
}
