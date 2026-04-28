import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy",
  description: "Sanketra privacy and DPDP statement."
};

export default function PrivacyPage() {
  return (
    <main className="narrow py-16">
      <p className="masthead mb-4">Privacy / DPDP</p>
      <h1 className="section-title">Zero data. Local by design.</h1>
      <section className="mt-12 border-t border-rule pt-8">
        <p className="masthead mb-4">हिंदी</p>
        <div className="prose prose-invert prose-editorial max-w-none">
          <h2 className="deva">हम कोई डेटा collect नहीं करते।</h2>
          <p>
            Sanketra पूरी तरह आपके PC और phone पर, आपके local Wi-Fi network पर चलता है।
            आपकी आवाज, transcripts, pairing code, server address, और settings हमारे पास
            नहीं आते। हमारे पास कोई server नहीं है जो user data receive करे।
          </p>
          <p>
            DPDP के हिसाब से हमारा stance simple है: हम personal data process नहीं करते,
            इसलिए बेचने, share करने, profile बनाने, ads चलाने, या analytics track करने
            का सवाल ही नहीं आता। अगर आप local transcript history enable करते हैं, वह आपके
            PC पर रहती है और आप उसे delete/export कर सकते हैं।
          </p>
        </div>
      </section>
      <section className="mt-12 border-t border-rule pt-8">
        <p className="masthead mb-4">English</p>
        <div className="prose prose-invert prose-editorial max-w-none">
          <h2>We collect ZERO user data.</h2>
          <p>
            Sanketra runs entirely on your PC and your phone, on your local Wi-Fi network.
            We have no servers. Your voice never leaves your Wi-Fi. We do not receive,
            store, sell, rent, or analyze your audio, transcripts, device identifiers,
            pairing codes, or usage patterns.
          </p>
          <p>
            There are no analytics scripts, tracking pixels, third-party session recorders,
            accounts, cloud speech-to-text calls, or advertising identifiers on this website
            or in the product. Payment providers may process payment details only when you
            choose to buy a paid plan.
          </p>
        </div>
      </section>
    </main>
  );
}
