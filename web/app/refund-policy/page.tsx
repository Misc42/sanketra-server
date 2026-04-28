import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Refund Policy",
  description: "Sanketra refund and cancellation policy."
};

export default function RefundPolicyPage() {
  return (
    <main className="narrow py-16">
      <p className="masthead mb-4">Refund Policy</p>
      <h1 className="section-title">Fair, fast, no fine print.</h1>
      <p className="mt-4 text-muted">
        Last updated: 28 April 2026
      </p>

      <section className="mt-12 border-t border-rule pt-8">
        <p className="masthead mb-4">हिंदी</p>
        <div className="prose prose-invert prose-editorial max-w-none">
          <h2 className="deva">1. 7-दिन का money-back</h2>
          <p>
            पहली बार Sanketra खरीद रहे हैं और product expectations match नहीं
            करता, तो <strong>7 दिनों के अंदर</strong> हम पूरा paisa वापस
            करेंगे — कोई reason बताने की ज़रूरत नहीं. Email करें{" "}
            <a href="mailto:tanaymisra97@gmail.com">tanaymisra97@gmail.com</a>{" "}
            par अपनी payment ID (Razorpay receipt में मिलेगी, e.g. <code>pay_XXXXX</code>) के साथ.
          </p>
          <p>
            Refund 5-7 working days में original payment method (UPI / card /
            net banking) पर वापस आ जाएगा. Refund initiate होते ही license
            denylist में move हो जाएगी, paid features unlock नहीं रहेगा.
          </p>

          <h2 className="deva">2. 7 दिन के बाद</h2>
          <p>
            7 दिनों के बाद refund केवल इन cases में मिलेगा:
          </p>
          <ul>
            <li>
              <strong>Technical defect जो हम 14 दिनों में fix नहीं कर सकते</strong> —
              जैसे license verify नहीं हो रही specific OS version पर, या paid
              features के bugs जो undocumented हैं और reproduce होते हैं.
            </li>
            <li>
              <strong>Service permanently discontinued</strong> — अगर हम
              Sanketra बंद कर देते हैं तो जिनकी license अभी active है उनको
              pro-rated refund मिलेगा (बचे हुए महीनों के हिसाब से).
            </li>
            <li>
              <strong>Duplicate payment</strong> — गलती से double-charge हो
              गया तो extra amount तुरंत वापस.
            </li>
          </ul>

          <h2 className="deva">3. Refund नहीं मिलेगा यदि</h2>
          <ul>
            <li>License key share / re-sell / leak की हो</li>
            <li>License verification bypass / reverse-engineer किया हो</li>
            <li>7-day window expire हो चुकी हो AND कोई technical defect नहीं</li>
            <li>"मुझे change of mind है" — 7-day window का whole point यही है</li>
          </ul>

          <h2 className="deva">4. कैसे request करें</h2>
          <ol>
            <li>
              Email <a href="mailto:tanaymisra97@gmail.com">tanaymisra97@gmail.com</a> करें
              subject में "Refund Request — [your payment ID]"
            </li>
            <li>Body में: payment ID, खरीद का date, amount, reason (optional)</li>
            <li>Reply 24-48 hours में, refund initiate 3 working days में</li>
            <li>Razorpay के through original payment method पर 5-7 working days में paisa</li>
          </ol>
        </div>
      </section>

      <section className="mt-12 border-t border-rule pt-8">
        <p className="masthead mb-4">English</p>
        <div className="prose prose-invert prose-editorial max-w-none">
          <h2>1. 7-day money-back</h2>
          <p>
            If this is your first Sanketra purchase and the product doesn&apos;t
            match your expectations, we&apos;ll refund the full amount{" "}
            <strong>within 7 days</strong> — no questions asked. Email{" "}
            <a href="mailto:tanaymisra97@gmail.com">tanaymisra97@gmail.com</a>{" "}
            with your Razorpay payment ID (looks like <code>pay_XXXXX</code>,
            visible on the receipt).
          </p>
          <p>
            Refunds land back on the original payment method (UPI / card /
            net banking) within 5-7 working days. Once a refund is initiated,
            the corresponding license is moved to the denylist — paid
            features stop unlocking immediately.
          </p>

          <h2>2. After 7 days</h2>
          <p>After the 7-day window, refunds are issued only for:</p>
          <ul>
            <li>
              <strong>Technical defects we cannot fix in 14 days</strong> —
              for example a license that fails to verify on a specific OS
              version, or undocumented and reproducible bugs in paid
              features.
            </li>
            <li>
              <strong>Permanent discontinuation of the service</strong> — if
              we shut Sanketra down, every active-license holder gets a
              pro-rated refund for the unused months.
            </li>
            <li>
              <strong>Duplicate payment</strong> — accidental double-charges
              are returned immediately.
            </li>
          </ul>

          <h2>3. Refunds are not available if</h2>
          <ul>
            <li>You&apos;ve shared, resold, or leaked the license key</li>
            <li>You&apos;ve attempted to bypass or reverse-engineer license verification</li>
            <li>The 7-day window has expired AND there is no technical defect</li>
            <li>
              Change of mind — that&apos;s exactly what the 7-day window is for
            </li>
          </ul>

          <h2>4. How to request a refund</h2>
          <ol>
            <li>
              Email{" "}
              <a href="mailto:tanaymisra97@gmail.com">tanaymisra97@gmail.com</a>{" "}
              with the subject line <code>Refund Request — [payment ID]</code>
            </li>
            <li>
              Include in the body: payment ID, purchase date, amount, reason
              (optional)
            </li>
            <li>
              We&apos;ll reply in 24-48 hours and initiate the refund within 3
              working days
            </li>
            <li>
              Razorpay returns the money to your original payment method in 5-7
              working days
            </li>
          </ol>

          <h2>5. Operator</h2>
          <p>
            Sanketra is a <strong>Misc42 Labs</strong> product, operated by
            Tanay Misra (sole proprietor, India). Bank statements and UPI
            receipts will identify the merchant accordingly.
          </p>
        </div>
      </section>
    </main>
  );
}
