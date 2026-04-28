import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "Sanketra terms and conditions of use."
};

export default function TermsPage() {
  return (
    <main className="narrow py-16">
      <p className="masthead mb-4">Terms of Service</p>
      <h1 className="section-title">Plain language. No traps.</h1>
      <p className="mt-4 text-muted">
        Last updated: 28 April 2026
      </p>

      <section className="mt-12 border-t border-rule pt-8">
        <p className="masthead mb-4">हिंदी</p>
        <div className="prose prose-invert prose-editorial max-w-none">
          <h2 className="deva">1. क्या service है</h2>
          <p>
            Sanketra एक Hindi-first voice-to-text input layer है जो आपके PC पर
            चलती है। यह free है (basic use). Paid SKUs (Phone Pro, Desktop Pro,
            Bundle) advanced features unlock करते हैं — जैसे large Whisper
            model, accent calibration, और extended local history.
          </p>

          <h2 className="deva">2. License</h2>
          <p>
            Paid SKU खरीदने पर आपको एक signed license key मिलती है (email से).
            यह key आप अपने PC पर paste करते हैं — license आपके specific device
            से bind होती है (cryptographic fingerprint), अपनी expiry tak valid
            रहती है (typically 1 साल per purchase). दूसरे device पर use
            करने के लिए license re-bind करना होगा (support से contact करें).
          </p>

          <h2 className="deva">3. Acceptable use</h2>
          <p>
            Sanketra को कानूनी कामों के लिए use करें — content creation,
            accessibility, productivity, education. License key को share /
            re-sell / illegally distribute नहीं करना. Reverse-engineer
            करना, license verification को bypass करना, या malicious purposes
            के लिए use करना prohibited है — ऐसा होने पर license revoke हो
            जाएगी (denylist में add) बिना refund के.
          </p>

          <h2 className="deva">4. Refund & cancellation</h2>
          <p>
            Detailed terms <a href="/refund-policy">/refund-policy</a> page पर हैं.
            Short version: पहली purchase पर 7-day no-questions refund. उसके बाद
            केवल technical defect (हम fix नहीं कर पा रहे) पर refund.
          </p>

          <h2 className="deva">5. Privacy</h2>
          <p>
            हम कोई user data collect नहीं करते. Detailed stance{" "}
            <a href="/privacy">/privacy</a> page पर है. Razorpay payment process
            करते समय कुछ payment-rail data देख सकता है — Razorpay की अपनी
            privacy policy applicable है उस हिस्से पर.
          </p>

          <h2 className="deva">6. Liability</h2>
          <p>
            Sanketra "as-is" provide की जाती है. हम guarantee नहीं देते कि यह
            हर hardware combination पर 100% accurate transcription करेगी —
            speech recognition की inherent limits हैं. किसी भी indirect /
            consequential / incidental loss के लिए हमारी maximum liability
            उस specific user ने जो amount paid किया है उसी तक limited है (last
            12 months में).
          </p>

          <h2 className="deva">7. Governing law</h2>
          <p>
            ये terms India के laws के under govern होते हैं. कोई dispute हो तो
            jurisdiction Indian courts ही होगी.
          </p>

          <h2 className="deva">8. Changes</h2>
          <p>
            हम इन terms को कभी-कभी update कर सकते हैं — material changes हो
            तो आपको email पर inform करेंगे (license owners), और changelog page
            पर notice रहेगा. Continued use = acceptance.
          </p>

          <h2 className="deva">9. Contact</h2>
          <p>
            Sanketra is a <strong>Misc42 Labs</strong> product, operated by
            Tanay Misra (sole proprietor, India). Questions? Email{" "}
            <a href="mailto:tanaymisra97@gmail.com">tanaymisra97@gmail.com</a>.
          </p>
        </div>
      </section>

      <section className="mt-12 border-t border-rule pt-8">
        <p className="masthead mb-4">English</p>
        <div className="prose prose-invert prose-editorial max-w-none">
          <h2>1. The service</h2>
          <p>
            Sanketra is a Hindi-first voice-to-text input layer that runs on
            your PC. It is free for basic use. Paid SKUs (Phone Pro, Desktop
            Pro, Bundle) unlock advanced features such as the large Whisper
            model, accent calibration, and extended local history.
          </p>

          <h2>2. License</h2>
          <p>
            Purchasing a paid SKU gets you a signed license key (delivered by
            email) which you paste into your local Sanketra installation. The
            license is cryptographically bound to your specific device
            fingerprint and is valid until its expiry (typically one year per
            purchase). To re-bind to a different device, contact support.
          </p>

          <h2>3. Acceptable use</h2>
          <p>
            Use Sanketra for lawful purposes — content creation, accessibility,
            productivity, education. Do not share, resell, or illegally
            redistribute license keys. Reverse-engineering, bypassing license
            verification, or any malicious use will result in the license
            being revoked (added to the denylist) without refund.
          </p>

          <h2>4. Refund &amp; cancellation</h2>
          <p>
            See <a href="/refund-policy">/refund-policy</a> for the full
            policy. In short: first-time purchases qualify for a no-questions
            refund within 7 days; after that, only confirmed technical defects
            we cannot fix qualify.
          </p>

          <h2>5. Privacy</h2>
          <p>
            We collect no user data — see <a href="/privacy">/privacy</a> for
            details. Razorpay processes payment-rail data when you buy; their
            own privacy notice applies to that portion of the flow.
          </p>

          <h2>6. Liability</h2>
          <p>
            Sanketra is provided as-is. We do not guarantee 100% transcription
            accuracy across every hardware combination — speech recognition
            has inherent limits. For any indirect, consequential, or
            incidental loss, our maximum liability is limited to the amount
            the specific user paid in the preceding 12 months.
          </p>

          <h2>7. Governing law</h2>
          <p>
            These terms are governed by the laws of India. Any disputes will
            be resolved in Indian courts.
          </p>

          <h2>8. Changes</h2>
          <p>
            We may update these terms occasionally. Material changes will be
            announced to license owners by email and noticed on the changelog
            page. Continued use after a change implies acceptance.
          </p>

          <h2>9. Contact</h2>
          <p>
            Sanketra is a <strong>Misc42 Labs</strong> product, operated by
            Tanay Misra (sole proprietor, India). For questions write to{" "}
            <a href="mailto:tanaymisra97@gmail.com">tanaymisra97@gmail.com</a>.
          </p>
        </div>
      </section>
    </main>
  );
}
