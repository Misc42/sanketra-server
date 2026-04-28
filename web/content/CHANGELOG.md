# Changelog

## v1.2.0 — Apr 21 (in flight, ship sprint)
**7-agent parallel ship sprint**: One-shot push for v1.2 — Chrome
extension v1, public Vercel marketing site (sanketra.app), PC server
dashboard at `localhost:5000/dashboard` with SQLite history (date-wise
+ session-wise transcripts, FTS5 search, opt-out logging, JSON/TXT/MD
export, real-time WS push), Android v1.2 polish (custom vocab CRUD UI,
accent calibration first-run flow, OEM-specific battery walkthrough
screens for Xiaomi/OPPO/Samsung), and ConnectionController extraction
out of MainViewModel (~600 LOC). Server gains `/api/vocab`,
`/api/accent`, `/api/history/*`, `/ws-dashboard`. Two new webpages:
public marketing on Vercel + private dashboard on PC server. Zero
mega-dependency: app works without either webpage. Reference docs
live at `docs/PRODUCT_VISION.html`, `docs/CHROME_EXTENSION.html`,
`docs/ANDROID_POLISH_V1.2.html`, `docs/DASHBOARD_AND_WEB.html`. Full
bug-by-bug log: `BUGS_AND_FIXES.md`.

## v1.1.4 follow-ups — Apr 20 (same day)
**Mar 24 audit P1/P2 second batch + release tooling**: `connectToServer`
guard now short-circuits on either `trackpadConnected` or
`trackpadConnecting` — LaunchedEffect re-fires during a slow connect
(DNS, TLS, TOFU prompt) used to cancel + restart the connectionJob in
a tight loop, server saw a burst of accept/disconnect, client wedged
in CONNECTING. AudioForegroundService companion `_serviceState` is
process-wide static; without an onDestroy reset, a service-stop-
without-process-death (mic perm revoke, OEM service-only kill) left
the next launch reading `isRecording=true` for a service that wasn't
running. `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` permission added to
manifest — Xiaomi MIUI / OPPO ColorOS silently drop the system dialog
intent without it, the persistent battery banner becomes a dead end.
ScreenWebSocket frame-copy GC: old path allocated twice per frame
(full ByteArray + trimmed copyOfRange = 1.8-6 MB/s of garbage at
30fps); new path reads header bytes directly off the okio.ByteString
indexer, only allocates the payload — config messages allocate
nothing. Light-mode WCAG AA contrast: cream paper backdrop (Y ≈ 0.89)
made mid-tone foregrounds fail AA — textTertiary 3.1:1 → 5.0:1
(#8A8576 → #6E6857), accentGreen 3.6:1 → 4.6:1 (#2BA34A → #1F8A38),
recordingRed 4.0:1 → 5.1:1. Dark mode untouched; brand stays visibly
green not jaundiced. Verified both modes on emulator. macOS Gatekeeper
bypass: `docs/install/sanketra-mac-install.command.zip` wraps the
install command so Safari/Chrome quarantine the .zip not the .command,
install page now leads with green Download button. `RELEASE.md` is
single-page source-of-truth for cutting Play Store release (5 GitHub
Secrets, manual fallback, keystore generation, sanketra repo sync).

**Mar 24 audit P1 — H264Decoder leak + Doze reconnect**: H264Decoder.stop()
and tryRecover() both had `mc.stop(); mc.release()` in the same try block —
if stop() threw (codec already in error state) release() would be skipped,
leaking the native MediaCodec instance from Android's fixed-size pool.
Split into separate try blocks. Doze recovery: ON_START in
ProcessLifecycleOwner now detects stale connection (server stored,
trackpad WS disconnected) and triggers `connectToServer(currentServer)`
inline — fixes the "open app after sleep, UI looks connected but next
action fails" bug from Mar 24 audit.

**UX recovery batch — cert one-tap re-pair + persistent battery banner**:
the cert-mismatch dialog used to bounce the user back to ServerList on
Accept; user then had to scan / find / re-pair (~4 taps). Now Accept
clears the TOFU fingerprint and re-invokes `connectToServer(currentServer)`
inline — if the saved session token still works, the user is back on
MainScreen in one tap; if not, the existing session-expired cascade
handles it. The battery-optimization snackbar (10s, then gone forever)
is replaced with a persistent warm-amber banner at the top of MainScreen
that stays put until the user fixes the OEM-killer setting or dismisses
it for the session. Picks up the OEM-specific guidance string for
Samsung/Xiaomi/OPPO/OnePlus/Vivo/Huawei.

**Server: `_accept_authenticated_ws` helper** dedupes the
`accept → origin → auth → register → close` boilerplate that lived in 4
WS handlers (audio stream, trackpad, screen mirror, audio output).
Variants had drifted (different log labels, different reason texts);
single source kills future drift. server_async.py: 3953 → 3892 lines.
88 tests still pass.

**Hindi i18n batch 2** (~50 new strings): cert dialog, disconnect
dialog, compute-switch dialog, battery banner, settings sheet labels,
permission prompts, server-list dialogs (offline/forget/uninstall),
section headers (RUNNING/NEW/SAVED/DEVICES → चल रहे हैं/नए/सेव/डिवाइस),
auth screen mastheads + bodies, MainScreen mastheads. Verified Hindi
rendering on emulator: section headers, action labels, server cards
all render Devanagari cleanly. ~90 of ~200 hardcoded strings extracted;
remainder are low-traffic edge dialogs.

**Editorial pass — MainScreen + ServerList rows**: ServerList group labels
(RUNNING / NEW / SAVED / DEVICES) and SET UP action labels switched to
`EyebrowStyle` / `MastheadStyle` (wide-tracked monospace, print-broadsheet
section headers). Card subtitles ('Ubuntu · SSH') use `TerminalStyle` so
they read like shell metadata instead of generic small caps. MainScreen
TopAppBar gains a green `CONNECTED` masthead label above the server name —
print-magazine "section header + publication name" feel. Connecting view
opens with a saffron `ESTABLISHING` masthead + hairline rule + serif
italic "Connecting…" — feels like the system is composing itself, not
just waiting.

**Editorial pass — Onboarding / Auth / MicControls**: each onboarding page
now opens with a saffron `EyebrowStyle` masthead label (`01 · YOUR PHONE`,
`02 · YOUR HANDS`, `03 · YOUR NETWORK`) sitting above a 48dp hairline rule —
print-broadsheet step indicator, kills the generic title-then-body slop.
AuthScreen Password / SetupRequired / Reset / Error views all gain an
`AuthMasthead` (saffron eyebrow + rule, red for ERROR) so each branch reads
as a labelled section, not a context-free headline. MicControls idle hint
renders as a serif italic quote (`"Hold to record"`) — feels like a calm
whisper rather than a system label. Active states (HOLDING / LOCKED /
PROCESSING) keep `labelSmall` so the urgency reads correctly.

**Language recovery (escape hatch from picker miss-tap)**: the first-launch
LanguagePickerScreen committed the user to a language with no way back. Two
new entry points fix that. (1) ServerListScreen top bar now has a wide-tracked
mono `HI`/`EN` toggle that always shows the OTHER language — one tap, Activity
recreates with the new locale. (2) SettingsSheet (reachable post-pair) gains a
`LANGUAGE  ·  भाषा` masthead row with हिंदी/English chips — same effect, more
discoverable inside the running session. Both routes call
`AppCompatDelegate.setApplicationLocales(...)` directly; no ViewModel state
needed because per-app locale is the global source of truth. Verified on
emulator: pick English on first launch → tap HI in top bar → flips to Hindi
→ tap EN → flips back. Activity recreate is clean (no flicker, no state loss
for ServerList).

**Install page editorial redesign** (`docs/install/index.html`, 158 → 590 lines):
the GitHub Pages page every new user lands on after typing tinyurl.com/sanketra.
Editorial direction with Devanagari display typography as the hero — huge
"बोलो, टाइप होगा।" set in Tiro Devanagari Hindi, italic "टाइप" in saffron-gold.
Terminal-aesthetic install card with `$`-prefix on the curl command, mono
labels, 3-step "what happens next" journey with serial-number watermarks and
a pulsing 4-digit pair-code skeleton. Trust strip: open source / Wi-Fi-only /
no account / no ads. Bilingual Hinglish microcopy ("Install karo", "Code
dikhega", "Phone par daalo"). Verified at desktop (1440px) and mobile (412px)
via Playwright headless chromium. Sync to `Misc42/sanketra` repo still pending
(GitHub Pages source).

**App: editorial palette + StatusBar masthead**: surfaces shifted from cool
iOS gray (#0A0A0B) to warm purple-black (#0F0E14); text from blue-shifted
white (#F5F5F7) to warm paper-white (#F4EFE6); cream-paper light mode
(#F5F1E8) instead of stark white. New `accentSaffron` (#E8B339) for editorial
display moments; green stays primary. New `rule` color for hairline dividers.
warningAmber softened (#FFD60A → #FFB800). textTertiary tightened to #8E8678
(WCAG AA 5.0:1 on dark surface). Five new non-Material display styles
(MastheadStyle, EyebrowStyle, DisplayItalicStyle, QuoteStyle, TerminalStyle).
StatusBar rebuilt: wide-tracked mono masthead row (`TRACK · AUDIO · SCREEN ·
SPEAKING ... GPU`) with hairline rule, transcript renders as serif italic
quote with smart quotes for final / ellipsis for partial. Foundation in place
for the rest of the screens; MainScreen layout, MicControls, ServerList, etc.
still pending editorial pass.

AuthController extraction (`domain/auth/AuthController.kt`): kills the 4-way
re-auth duplication in MainViewModel. `applyNewToken()` is now the single owner
of the WS-reconnect cascade. handleSessionExpired collapses ~140 → ~25 lines.
ffmpeg encoder probe extended: NVENC → **VAAPI** (`/dev/dri/renderD128`) → Intel
QSV → libx264. VAAPI bypasses the broken CUDA context entirely (3-6 ms encode
vs libx264's 8-15 ms). `auth_core.py` extracted with pure rate-limit + session
primitives; `tests/test_auth_logic.py` now tests real production code, not the
old reimplementation that filtered attempts in a 60 s window vs production's
600 s. 77 → 88 tests. Sync workflow updated to copy VERSION + static/.

## v1.1.4 (vC 10104) — Apr 20
Hindi-first onboarding: language picker as first screen (हिंदी / English), `values-hi/strings.xml` with conversational translations, Devanagari-tuned typography (lineHeight +10%, letterSpacing 0). MainActivity migrated to AppCompatActivity for per-app locale infra. Audio receive-loop rearchitected: WS receive runs in a separate task draining into asyncio.Queue, processor consumes — kills the syllable-clipping bug where `await run_inference` blocked frame intake. Server races fixed: `_model_load_lock` (no concurrent OOM), `_pair_mutex` (atomic pair transaction), `load_whisper` returns actual device (no more "GPU active" lie when CPU fallback hit). VAD fallback now actually works (`EnergyVADIterator` wrapper, server stops rejecting EnergyVAD with 4500). Android: `SavedStateHandle` actually injected (was defaulted = dead code), FGS keeps MICROPHONE permanent (kills Samsung/Xiaomi WifiLock handover race), AudioFocus TRANSIENT_EXCLUSIVE pauses Spotify, DisposableEffect disconnect race removed (no more recording death on nav-back), mic threshold 300→400ms. EasySetup timeout 600s→90s with actionable error. requirements.txt: dropped flask/flask-sock, added argon2-cffi (was imported, undeclared). Argon2 RFC 9106 LOW_MEMORY profile. Sudoers runtime mutation removed (read-only check). VERSION file at root, single source for Python+Android. server_async.py 5762→3892 lines: HTML extracted to `static/index.html`. 22 files modified, 5 new. 77 Python tests + Android unit tests pass.

## v1.1.3 (vC 11) — Mar 6
Theme: system light/dark, 390+ colors migrated. Screen mirror: pinch-zoom, auto-landscape, cursor overlay 30Hz, adaptive quality, PC audio, multi-monitor. Input: soft keyboard, haptic fix, natural scroll. Network: subnet-aware discovery, structured logging. NDK symbols.

## v1.1.2 (vC 10) — Feb 23
Auth fix (kills old service), dual config path, SSH auto-recovery. GPU auto-repair, CPU/GPU switch. Clean uninstall all OS. Auto-disconnect 5s bg, watchdog 10s.

## v1.1.1 (vC 9) — Feb 21
Settings UX, model download overlay, LAN dedup, ethernet pref, watchdog.

## v1.1.0 (vC 8) — Feb 20
77 bugs (14P0). JsonStore atomic, runBlocking ANR, Surface OOM, TCP_NODELAY, pre-alloc buffers. 109 Android + 77 Python tests.

## Earlier
v1.0.6 (Feb 20): cross-plat audits. v1.0.4: cross-plat. v1.0.2: mirror fix. v1.0.0: initial.
