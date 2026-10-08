/* demo-library.js - SAMPLE data behind the hero's search bar.
 *
 * A public page cannot see a visitor's recordings, so the search runs over
 * this invented library instead. Every word below is made up for the demo.
 *
 * TODO(hero-video): when the real hero loop is recorded, replace the
 * `featured` project's `segments` with that take's transcript.json
 * "segments" ({t, dur, text}, seconds on the video's own clock) so picking a
 * result seeks the background video to the moment it was said. NEVER paste a
 * real personal take's transcript here: this file ships publicly.
 */
window.AUTOCINE_DEMO = {
  featured: "checkout-walkthrough",
  projects: [
    {
      id: "checkout-walkthrough",
      name: "Checkout flow walkthrough",
      recorded: "Sep 24",
      duration: 38.0,
      clicks: 14,
      segments: [
        { t: 0.0,  dur: 4.2, text: "Quick walkthrough of the new checkout flow before we ship it." },
        { t: 4.2,  dur: 5.1, text: "Start on the cart page and click Proceed to checkout." },
        { t: 9.3,  dur: 5.6, text: "Shipping address autofills from the saved profile, so I just confirm it." },
        { t: 14.9, dur: 4.8, text: "Here's the promo code field, it validates as you type." },
        { t: 19.7, dur: 6.0, text: "Pick express shipping and watch the total update on the right." },
        { t: 25.7, dur: 5.4, text: "Card details, then click Place order." },
        { t: 31.1, dur: 6.9, text: "And the confirmation page shows the order number and the delivery estimate." }
      ]
    },
    {
      id: "api-quickstart",
      name: "API quickstart in the terminal",
      recorded: "Sep 22",
      duration: 312.4,
      clicks: 41,
      segments: [
        { t: 3.0,   dur: 6.5, text: "First install the CLI and log in with your API key." },
        { t: 42.8,  dur: 7.2, text: "Create a project, then paste the key into the env file." },
        { t: 96.1,  dur: 6.0, text: "Now run the first request and look at the JSON that comes back." },
        { t: 158.4, dur: 8.1, text: "If you get a 401 here, the key is missing from your environment." },
        { t: 221.9, dur: 6.6, text: "Pagination works with a cursor, not page numbers." },
        { t: 287.0, dur: 7.3, text: "That's it, you can export the whole collection as CSV from the dashboard." }
      ]
    },
    {
      id: "pricing-review",
      name: "Design review: pricing page",
      recorded: "Sep 19",
      duration: 486.0,
      clicks: 57,
      segments: [
        { t: 12.4,  dur: 6.8, text: "Let's look at the pricing page, starting with the plan comparison table." },
        { t: 74.0,  dur: 7.5, text: "The annual toggle is hard to find, I'd move it above the cards." },
        { t: 151.3, dur: 5.9, text: "On mobile the table scrolls sideways, which nobody will discover." },
        { t: 233.8, dur: 8.2, text: "The enterprise card needs a contact button, not a checkout button." },
        { t: 340.5, dur: 6.4, text: "FAQ answers are fine, just shorten the refund policy one." },
        { t: 452.2, dur: 7.1, text: "Overall close, let's ship after the toggle and the mobile fix." }
      ]
    },
    {
      id: "settings-bug",
      name: "Bug repro: settings won't save",
      recorded: "Sep 17",
      duration: 94.6,
      clicks: 22,
      segments: [
        { t: 2.1,  dur: 5.0, text: "Repro for the settings bug, this happens every time." },
        { t: 18.7, dur: 6.3, text: "Change the notification email and click Save." },
        { t: 31.0, dur: 5.8, text: "The toast says saved, but reload the page and it's back to the old value." },
        { t: 55.4, dur: 7.4, text: "The network tab shows the request returns a 200 with an empty body." },
        { t: 80.2, dur: 6.1, text: "Same thing in Safari, so it's not a browser cache issue." }
      ]
    },
    {
      id: "metrics-weekly",
      name: "Weekly metrics review",
      recorded: "Sep 12",
      duration: 603.8,
      clicks: 33,
      segments: [
        { t: 8.0,   dur: 6.2, text: "Weekly numbers, signups are up eleven percent." },
        { t: 121.5, dur: 7.0, text: "Retention dipped in week two, mostly from the onboarding change." },
        { t: 264.9, dur: 6.8, text: "Click into the funnel chart, the drop is on the verify email step." },
        { t: 398.2, dur: 7.6, text: "Support tickets about export are down since the fix shipped." },
        { t: 540.0, dur: 6.9, text: "Action items: revert the onboarding copy and rerun the email test." }
      ]
    }
  ],
  // Example queries the placeholder cycles through (each must hit something).
  examples: [
    "where I clicked Place order",
    "the part about pricing",
    "401 error",
    "export as CSV",
    "promo code"
  ]
};
