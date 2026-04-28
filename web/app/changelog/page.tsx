import type { Metadata } from "next";
import { getChangelogSections } from "@/lib/changelog";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "Changelog",
  description: "Sanketra release notes."
};

export default function ChangelogPage() {
  const sections = getChangelogSections();

  return (
    <main className="narrow py-16">
      <p className="masthead mb-4">Changelog</p>
      <h1 className="section-title">Release notes from the repo.</h1>
      <div className="mt-12 grid gap-10">
        {sections.map((section) => (
          <section key={`${section.version}-${section.date}`} className="border-t border-rule pt-8">
            <p className="masthead mb-3">{section.date || "Release"}</p>
            <h2 className="text-3xl font-semibold tracking-normal">{section.version}</h2>
            <ul className="mt-5 grid gap-3 text-muted">
              {section.bullets.map((bullet) => (
                <li key={bullet.slice(0, 80)} className="border-l-2 border-rule pl-4">
                  {bullet}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </main>
  );
}
