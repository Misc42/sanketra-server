/* ==========================================================================
   Sanketra Dashboard — vanilla JS
   Owns: view routing, API fetch, inline vocab editor, WS push, toasts.
   Assumes endpoints from DASHBOARD_AND_WEB.html §3.4 (may 404 during server buildout —
   we degrade gracefully instead of exploding).
   ========================================================================== */

(function () {
  'use strict';

  // ---------- small utils ------------------------------------------------
  const $  = (sel, ctx = document) => ctx.querySelector(sel);
  const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

  const API = {
    history: {
      byDate: (d) => `/api/history/by-date/${d}`,
      sessions: (page = 0, size = 50) => `/api/history/sessions?page=${page}&size=${size}`,
      search: (q) => `/api/history/search?q=${encodeURIComponent(q)}`,
      del: (id) => `/api/history/sessions/${id}`,
      delTranscript: (id) => `/api/history/transcripts/${id}`,
      delAll: '/api/history',
      export: (fmt) => `/api/history/export?fmt=${fmt}`,
      settings: '/api/history/settings',
    },
    vocab: '/api/vocab',
    accent: '/api/accent',
    devices: '/api/auth/devices',
    version: '/version',
  };

  const DEBOUNCE = {
    search: 300,
    wsRefresh: 500,
    vocabSave: 400,
  };

  const state = {
    currentView: 'today',
    allPage: 0,
    allPageSize: 50,
    allTotalPages: null,
    vocabEntries: [],
    collapsedSessions: new Set(),
    ws: null,
    wsReconnectTimer: null,
    wsRefreshTimer: null,
  };

  // ---------- date helpers ----------------------------------------------
  function todayISO() {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }
  function daysAgoISO(n) {
    const d = new Date();
    d.setDate(d.getDate() - n);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }
  function formatDateLong(iso) {
    try {
      const d = new Date(iso + 'T00:00:00');
      return d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
    } catch (_) { return iso; }
  }
  function formatTime(ms) {
    try {
      return new Date(ms).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });
    } catch (_) { return '—'; }
  }
  function formatClock() {
    return new Date().toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });
  }

  // ---------- HTML escaping ---------------------------------------------
  const escapeHtml = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  function highlightMatch(text, query) {
    if (!query || !text) return escapeHtml(text);
    const q = query.trim();
    if (!q) return escapeHtml(text);
    const safe = escapeHtml(text);
    try {
      const re = new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
      return safe.replace(re, '<mark>$1</mark>');
    } catch (_) { return safe; }
  }

  function isHindi(s) {
    return /[\u0900-\u097F]/.test(String(s || ''));
  }

  // ---------- toast ------------------------------------------------------
  function toast(msg, kind = '') {
    const wrap = $('#toast-wrap');
    if (!wrap) return;
    const el = document.createElement('div');
    el.className = 'toast' + (kind ? ' ' + kind : '');
    el.textContent = msg;
    wrap.appendChild(el);
    setTimeout(() => { el.classList.add('out'); }, 2600);
    setTimeout(() => { el.remove(); }, 3000);
  }

  // ---------- auth token bootstrap (Codex F-Apr21-05) -------------------
  // Dashboard runs at https://localhost:5000/dashboard?token=<TOKEN>. The
  // server's /dashboard route appends the token from a same-LAN session.
  // For now: read token from URL ?token= or from localStorage cache.
  function getToken() {
    const fromQuery = new URLSearchParams(location.search).get('token');
    if (fromQuery) {
      try { localStorage.setItem('sanketra_dashboard_token', fromQuery); } catch (_) {}
      return fromQuery;
    }
    try { return localStorage.getItem('sanketra_dashboard_token') || ''; } catch (_) { return ''; }
  }
  const TOKEN = getToken();

  // ---------- fetch with graceful degrade + auth ------------------------
  async function fetchJSON(url, opts = {}) {
    const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    if (TOKEN) headers['Authorization'] = `Bearer ${TOKEN}`;
    // Also append ?token= for endpoints that prefer query-param auth (legacy).
    let withToken = url;
    if (TOKEN && !url.includes('token=')) {
      withToken = url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(TOKEN);
    }
    const res = await fetch(withToken, {
      ...opts,
      credentials: 'same-origin',
      headers,
    });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) return res.json();
    return res.text();
  }

  function renderError(container, err, retry) {
    if (!container) return;
    const isNotFound = err && err.status === 404;
    container.innerHTML = `
      <div class="error-block">
        <strong>${isNotFound ? 'Endpoint not ready' : 'Server unreachable'}</strong>
        ${isNotFound
          ? 'This API endpoint isn\'t wired yet on the server side. Your data is fine — the dashboard is ahead of the backend.'
          : 'Could not reach the Whisper server. Check if it\'s running.'}
        ${retry ? '<div class="error-retry"><button class="btn" id="err-retry">Retry</button></div>' : ''}
      </div>
    `;
    if (retry) {
      const btn = container.querySelector('#err-retry');
      if (btn) btn.addEventListener('click', retry);
    }
  }

  // ---------- view routing ----------------------------------------------
  function setView(view) {
    state.currentView = view;
    $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
    $$('.view').forEach(v => {
      const on = v.id === `view-${view}`;
      v.hidden = !on;
      v.classList.toggle('active', on);
    });
    // lazy-load
    loadView(view);
    try { localStorage.setItem('sanketra:lastView', view); } catch (_) {}
  }

  function loadView(view) {
    switch (view) {
      case 'today':    loadToday();    break;
      case 'week':     loadWeek();     break;
      case 'all':      loadAll();      break;
      case 'vocab':    loadVocab();    break;
      case 'settings': loadSettings(); break;
      case 'search':   /* waits for user input */ break;
    }
  }

  // ---------- transcript + session render -------------------------------
  function renderTranscript(t, query) {
    const hi = isHindi(t.text);
    const quoted = query ? highlightMatch(t.text, query) : escapeHtml(t.text);
    return `
      <div class="transcript" data-transcript-id="${t.id}">
        <div class="transcript-quote${hi ? ' hi' : ''}">${quoted || '<em>(empty)</em>'}</div>
        <div class="transcript-meta">
          <span class="tm-time">${formatTime(t.created_at)}</span>
          <span class="tm-sep">·</span>
          <span>${t.word_count || 0} words</span>
          ${t.app_context ? `<span class="tm-sep">·</span><span>${escapeHtml(t.app_context)}</span>` : ''}
          <span class="transcript-actions">
            <button class="icon-btn copy-t" data-text="${escapeHtml(t.text || '')}">Copy</button>
            <button class="icon-btn danger del-t" data-id="${t.id}">Delete</button>
          </span>
        </div>
      </div>
    `;
  }

  // Codex F-Apr21-06: group flat transcript list by session_id.
  function groupTranscriptsBySession(transcripts) {
    if (!Array.isArray(transcripts) || !transcripts.length) return [];
    const map = new Map();
    for (const t of transcripts) {
      const sid = t.session_id != null ? t.session_id : 'unknown';
      if (!map.has(sid)) {
        map.set(sid, {
          id: sid,
          started_at: t.created_at || 0,
          ended_at: t.created_at || 0,
          client_kind: t.client_kind || 'unknown',
          client_name: t.client_name || '',
          language: t.language || 'hi',
          transcripts: [],
        });
      }
      const sess = map.get(sid);
      sess.transcripts.push(t);
      if ((t.created_at || 0) < sess.started_at) sess.started_at = t.created_at;
      if ((t.created_at || 0) > sess.ended_at) sess.ended_at = t.created_at;
    }
    // Sort sessions by started_at descending (newest first)
    return Array.from(map.values()).sort((a, b) => b.started_at - a.started_at);
  }

  function renderSession(s) {
    const clientKind = (s.client_kind || 'web').toLowerCase();
    const collapsed = state.collapsedSessions.has(s.id) ? ' collapsed' : '';
    const wordTotal = (s.transcripts || []).reduce((a, t) => a + (t.word_count || 0), 0);
    return `
      <article class="session${collapsed}" data-session-id="${s.id}">
        <header class="session-head" data-toggle-session>
          <div class="session-head-left">
            <span class="session-chevron">▾</span>
            <span class="session-time">${formatTime(s.started_at)}</span>
            <span class="session-client ${escapeHtml(clientKind)}">${escapeHtml(s.client_name || clientKind)}</span>
          </div>
          <div class="session-stats">
            <strong>${(s.transcripts || []).length}</strong> transcripts · <strong>${wordTotal}</strong> words
            <button class="icon-btn danger del-s" data-id="${s.id}" style="margin-left:12px">Delete</button>
          </div>
        </header>
        <div class="session-body">
          ${(s.transcripts || []).map(t => renderTranscript(t)).join('')}
        </div>
      </article>
    `;
  }

  // ---------- TODAY view -------------------------------------------------
  async function loadToday() {
    const body = $('#today-body');
    const dateEl = $('#today-date');
    const statsEl = $('#today-stats');
    if (dateEl) dateEl.textContent = formatDateLong(todayISO());
    body.innerHTML = `<div class="skeleton-stack" aria-busy="true">
      <div class="skel skel-q"></div>
      <div class="skel skel-q"></div>
      <div class="skel skel-q short"></div>
    </div>`;
    try {
      const data = await fetchJSON(API.history.byDate(todayISO()));
      // Codex F-Apr21-06: server returns {date, transcripts: [...]}, NOT
      // {sessions}. Group transcripts by session_id on the client so existing
      // renderSession can be reused without server-side change.
      const transcripts = (data && data.transcripts) || [];
      const sessions = groupTranscriptsBySession(transcripts);
      if (!sessions.length) {
        body.innerHTML = `
          <div class="empty-quote">
            <em>Aaj kuch nahi bola.</em> Try dictation.
            <span class="cue">Long-press the mic on your phone to begin</span>
          </div>`;
        if (statsEl) statsEl.innerHTML = `<span class="stat"><strong>0</strong> sessions</span><span class="stat-sep">·</span><span class="stat"><strong>0</strong> words</span>`;
        return;
      }
      body.innerHTML = sessions.map(renderSession).join('');
      const wordTotal = sessions.reduce((a, s) => a + (s.transcripts || []).reduce((x, t) => x + (t.word_count || 0), 0), 0);
      if (statsEl) statsEl.innerHTML = `<span class="stat"><strong>${sessions.length}</strong> sessions</span><span class="stat-sep">·</span><span class="stat"><strong>${wordTotal}</strong> words</span>`;
    } catch (err) {
      renderError(body, err, loadToday);
    }
  }

  // ---------- WEEK view --------------------------------------------------
  async function loadWeek() {
    const body = $('#week-body');
    body.innerHTML = `<div class="empty-placeholder">Loading past 7 days…</div>`;
    try {
      const days = [];
      for (let i = 0; i < 7; i++) days.push(daysAgoISO(i));
      const results = await Promise.all(days.map(d =>
        fetchJSON(API.history.byDate(d)).catch(e => ({ _err: e, _date: d }))
      ));
      const anyWorked = results.some(r => !r._err);
      if (!anyWorked) {
        renderError(body, results[0]._err, loadWeek);
        return;
      }
      const parts = days.map((d, i) => {
        const r = results[i];
        // Codex F-Apr21-06: server returns {transcripts}, group on client.
        const transcripts = (r && !r._err && r.transcripts) || [];
        const sessions = groupTranscriptsBySession(transcripts);
        if (!sessions.length) return '';
        const dayLabel = i === 0 ? 'Today' : (i === 1 ? 'Yesterday' : formatDateLong(d));
        const wordTotal = sessions.reduce((a, s) => a + (s.transcripts || []).reduce((x, t) => x + (t.word_count || 0), 0), 0);
        return `
          <section class="day-group">
            <header class="day-header">
              <span>${escapeHtml(dayLabel)}</span>
              <span class="day-header-count">${sessions.length} sessions · ${wordTotal} words</span>
            </header>
            ${sessions.map(renderSession).join('')}
          </section>
        `;
      }).filter(Boolean);
      if (!parts.length) {
        body.innerHTML = `<div class="empty-quote"><em>Pichle 7 din mein kuch nahi.</em> Fresh start.</div>`;
        return;
      }
      body.innerHTML = parts.join('');
    } catch (err) {
      renderError(body, err, loadWeek);
    }
  }

  // ---------- ALL view (paginated) --------------------------------------
  async function loadAll() {
    const body = $('#all-body');
    const prev = $('#all-prev');
    const next = $('#all-next');
    const info = $('#all-pageinfo');
    body.innerHTML = `<div class="empty-placeholder">Loading page ${state.allPage + 1}…</div>`;
    try {
      const data = await fetchJSON(API.history.sessions(state.allPage, state.allPageSize));
      const sessions = (data && data.sessions) || [];
      const total = (data && data.total) || sessions.length;
      state.allTotalPages = Math.max(1, Math.ceil(total / state.allPageSize));
      if (!sessions.length) {
        body.innerHTML = `<div class="empty-quote"><em>Archive empty.</em> Dictate something.</div>`;
      } else {
        body.innerHTML = sessions.map(renderSession).join('');
      }
      if (info) info.textContent = `Page ${state.allPage + 1} of ${state.allTotalPages}`;
      if (prev) prev.disabled = state.allPage <= 0;
      if (next) next.disabled = state.allPage + 1 >= state.allTotalPages;
    } catch (err) {
      renderError(body, err, loadAll);
    }
  }

  // ---------- SEARCH view -----------------------------------------------
  let searchTimer = null;
  async function runSearch(q) {
    const body = $('#search-body');
    const meta = $('#search-meta');
    if (!q || !q.trim()) {
      body.innerHTML = `<div class="empty-quote"><em>Type something above to search your archive.</em></div>`;
      if (meta) meta.textContent = '—';
      return;
    }
    if (meta) meta.textContent = 'Searching…';
    body.innerHTML = `<div class="empty-placeholder">Searching…</div>`;
    try {
      const data = await fetchJSON(API.history.search(q));
      const hits = (data && data.results) || [];
      if (meta) meta.textContent = `${hits.length} ${hits.length === 1 ? 'match' : 'matches'}`;
      if (!hits.length) {
        body.innerHTML = `
          <div class="empty-quote">
            <em>Nothing found.</em>
            <span class="cue">Try a shorter query or different spelling</span>
          </div>`;
        return;
      }
      // Hits are flat transcripts — render as standalone quotes, no session wrapper.
      body.innerHTML = hits.map(t => `
        <div class="transcript" data-transcript-id="${t.id}" style="margin-bottom:28px">
          <div class="transcript-quote${isHindi(t.text) ? ' hi' : ''}">${highlightMatch(t.text, q) || '<em>(empty)</em>'}</div>
          <div class="transcript-meta">
            <span class="tm-time">${formatTime(t.created_at)}</span>
            <span class="tm-sep">·</span>
            <span>${t.word_count || 0} words</span>
            ${t.app_context ? `<span class="tm-sep">·</span><span>${escapeHtml(t.app_context)}</span>` : ''}
            <span class="transcript-actions">
              <button class="icon-btn copy-t" data-text="${escapeHtml(t.text || '')}">Copy</button>
              <button class="icon-btn danger del-t" data-id="${t.id}">Delete</button>
            </span>
          </div>
        </div>
      `).join('');
    } catch (err) {
      renderError(body, err, () => runSearch(q));
      if (meta) meta.textContent = 'error';
    }
  }

  // ---------- VOCAB view -------------------------------------------------
  async function loadVocab() {
    const body = $('#vocab-body');
    const count = $('#vocab-count');
    body.innerHTML = `<div class="empty-placeholder">Loading dictionary…</div>`;
    try {
      const data = await fetchJSON(API.vocab);
      const entries = (data && data.entries) || [];
      state.vocabEntries = entries;
      if (count) count.textContent = `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}`;
      renderVocabRows(entries);
    } catch (err) {
      renderError(body, err, loadVocab);
    }
  }

  function renderVocabRows(entries) {
    const body = $('#vocab-body');
    if (!entries.length) {
      body.innerHTML = `<div class="empty-quote" style="margin:20px"><em>No custom words yet.</em><span class="cue">Click "Add word" above</span></div>`;
      return;
    }
    body.innerHTML = entries.map((e, idx) => vocabRowHTML(e, idx)).join('');
  }

  function vocabRowHTML(entry, idx) {
    return `
      <div class="vocab-row" data-idx="${idx}">
        <input class="vocab-input text"     value="${escapeHtml(entry.text || '')}"      placeholder="Word" aria-label="Word">
        <input class="vocab-input phonetic" value="${escapeHtml(entry.phonetic || '')}"  placeholder="Phonetic (e.g. saa-ket-ra)" aria-label="Phonetic">
        <input class="vocab-input weight"   value="${escapeHtml(String(entry.weight != null ? entry.weight : 1.0))}" inputmode="decimal" aria-label="Weight">
        <button class="icon-btn danger vocab-del" data-idx="${idx}">Delete</button>
      </div>
    `;
  }

  function collectVocabFromDOM() {
    return $$('#vocab-body .vocab-row').map(row => {
      const text = row.querySelector('.text').value.trim();
      const phonetic = row.querySelector('.phonetic').value.trim();
      const weight = parseFloat(row.querySelector('.weight').value);
      return { text, phonetic, weight: isNaN(weight) ? 1.0 : weight };
    }).filter(e => e.text);
  }

  async function saveVocab() {
    const entries = collectVocabFromDOM();
    try {
      // Codex F-Apr21-07: server PATCH expects {add, remove}, not {entries}.
      // Use POST for full-replacement which the server's POST handler accepts.
      await fetchJSON(API.vocab, {
        method: 'POST',
        body: JSON.stringify({ entries }),
      });
      const count = $('#vocab-count');
      if (count) count.textContent = `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}`;
      state.vocabEntries = entries;
      toast('Saved', 'ok');
    } catch (err) {
      toast(err.status === 404 ? 'Endpoint not wired yet' : 'Save failed', 'err');
    }
  }

  // ---------- SETTINGS view ---------------------------------------------
  async function loadSettings() {
    loadLoggingSetting();
    loadAccentStatus();
    loadDevices();
  }

  async function loadLoggingSetting() {
    const toggle = $('#logging-toggle');
    const label = $('#logging-label');
    try {
      const data = await fetchJSON(API.history.settings);
      const on = !!(data && data.logging_enabled);
      toggle.checked = on;
      label.textContent = on ? 'Logging: ON' : 'Logging: OFF';
    } catch (err) {
      toggle.checked = true;
      label.textContent = (err.status === 404) ? 'Not configured' : 'Error';
    }
  }

  async function loadAccentStatus() {
    const el = $('#accent-status');
    if (!el) return;
    try {
      const data = await fetchJSON(API.accent);
      const hasProfile = data && data.has_profile;
      const last = data && data.calibrated_at ? new Date(data.calibrated_at).toLocaleDateString() : null;
      const dialect = (data && data.dialect) || '—';
      el.innerHTML = `
        <div class="accent-line saff">Status: <strong>${hasProfile ? 'Calibrated' : 'Not calibrated'}</strong></div>
        ${last ? `<div class="accent-line">Last calibration: <strong>${escapeHtml(last)}</strong></div>` : ''}
        <div class="accent-line">Dialect estimate: <strong>${escapeHtml(dialect)}</strong></div>
      `;
    } catch (err) {
      el.innerHTML = `<div class="accent-line">Status: <strong>${err.status === 404 ? 'Endpoint not ready' : 'Error'}</strong></div>`;
    }
  }

  async function loadDevices() {
    const el = $('#device-list');
    if (!el) return;
    try {
      const data = await fetchJSON(API.devices);
      const devices = (data && data.devices) || [];
      if (!devices.length) {
        el.innerHTML = `<div class="accent-line">No devices paired yet.</div>`;
        return;
      }
      el.innerHTML = devices.map(d => `
        <div class="device-row">
          <div>
            <div class="device-name">${escapeHtml(d.name || d.kind || 'Unknown')}</div>
            <div class="device-meta">${escapeHtml(d.kind || '')} · paired ${d.paired_at ? new Date(d.paired_at).toLocaleDateString() : '—'}</div>
          </div>
        </div>
      `).join('');
    } catch (err) {
      el.innerHTML = `<div class="accent-line">${err.status === 404 ? 'Endpoint not ready' : 'Could not load'}</div>`;
    }
  }

  // ---------- WebSocket push --------------------------------------------
  function connectWS() {
    try {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      // Codex F-Apr21-05: pass token as query-param so server can authenticate
      // the WS handshake (browsers can't add Authorization header to WS).
      const tokenSuffix = TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : '';
      const url = `${proto}//${location.host}/ws-dashboard${tokenSuffix}`;
      const ws = new WebSocket(url);
      state.ws = ws;
      ws.onopen = () => {
        setWsStatus('live', 'live');
      };
      ws.onclose = () => {
        setWsStatus('offline', '');
        clearTimeout(state.wsReconnectTimer);
        state.wsReconnectTimer = setTimeout(connectWS, 3000);
      };
      ws.onerror = () => {
        setWsStatus('error', 'err');
      };
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (_) { return; }
        if (msg && msg.type === 'transcript') {
          scheduleRefresh();
        }
      };
    } catch (err) {
      setWsStatus('unavailable', 'err');
    }
  }

  function scheduleRefresh() {
    clearTimeout(state.wsRefreshTimer);
    state.wsRefreshTimer = setTimeout(() => {
      loadView(state.currentView);
    }, DEBOUNCE.wsRefresh);
  }

  function setWsStatus(text, cls) {
    const el = $('#ws-status');
    if (!el) return;
    el.innerHTML = `<span class="ws-dot ${cls || ''}"></span>${text}`;
  }

  // ---------- event wiring ----------------------------------------------
  function wireTabs() {
    $$('.tab').forEach(t => t.addEventListener('click', () => setView(t.dataset.view)));
  }

  function wirePrivacy() {
    $('#privacy-more').addEventListener('click', () => openModal('#privacy-modal'));
  }

  function wireSearch() {
    const input = $('#search-input');
    input.addEventListener('input', () => {
      clearTimeout(searchTimer);
      const q = input.value;
      searchTimer = setTimeout(() => runSearch(q), DEBOUNCE.search);
    });
  }

  function wireAllPagination() {
    $('#all-prev').addEventListener('click', () => {
      if (state.allPage > 0) { state.allPage--; loadAll(); }
    });
    $('#all-next').addEventListener('click', () => {
      if (state.allTotalPages == null || state.allPage + 1 < state.allTotalPages) {
        state.allPage++;
        loadAll();
      }
    });
  }

  function wireVocabControls() {
    $('#vocab-add').addEventListener('click', () => {
      const body = $('#vocab-body');
      // Remove empty placeholder if present
      const placeholder = body.querySelector('.empty-quote, .empty-placeholder');
      if (placeholder) placeholder.remove();
      const row = document.createElement('div');
      row.className = 'vocab-row adding';
      row.innerHTML = `
        <input class="vocab-input text" value="" placeholder="Word" aria-label="Word" autofocus>
        <input class="vocab-input phonetic" value="" placeholder="Phonetic (e.g. saa-ket-ra)" aria-label="Phonetic">
        <input class="vocab-input weight" value="1.0" inputmode="decimal" aria-label="Weight">
        <button class="icon-btn danger vocab-del">Delete</button>
      `;
      body.insertBefore(row, body.firstChild);
      row.querySelector('.text').focus();
      setTimeout(() => row.classList.remove('adding'), 1000);
    });

    const body = $('#vocab-body');
    let vocabTimer = null;
    body.addEventListener('blur', (e) => {
      if (e.target.classList && e.target.classList.contains('vocab-input')) {
        clearTimeout(vocabTimer);
        vocabTimer = setTimeout(saveVocab, DEBOUNCE.vocabSave);
      }
    }, true);
    body.addEventListener('keydown', (e) => {
      if (e.target.classList && e.target.classList.contains('vocab-input') && e.key === 'Enter') {
        e.target.blur();
      }
    });
    body.addEventListener('click', (e) => {
      const del = e.target.closest('.vocab-del');
      if (del) {
        const row = del.closest('.vocab-row');
        if (row) {
          const word = row.querySelector('.text').value.trim() || 'this entry';
          openConfirm({
            eyebrow: 'Delete',
            title: `Remove "${word}"?`,
            body: 'Is shabd ko dictionary se hata denge.',
            go: async () => {
              row.remove();
              await saveVocab();
              const remaining = $$('#vocab-body .vocab-row');
              if (!remaining.length) renderVocabRows([]);
            }
          });
        }
      }
    });
  }

  function wireSettings() {
    $('#logging-toggle').addEventListener('change', async (e) => {
      const on = !!e.target.checked;
      const label = $('#logging-label');
      try {
        await fetchJSON(API.history.settings, {
          method: 'POST',
          body: JSON.stringify({ logging_enabled: on }),
        });
        label.textContent = on ? 'Logging: ON' : 'Logging: OFF';
        toast(on ? 'Logging enabled' : 'Logging disabled', 'ok');
      } catch (err) {
        e.target.checked = !on;
        toast('Could not update setting', 'err');
      }
    });

    $$('[data-export]').forEach(btn => btn.addEventListener('click', () => {
      const fmt = btn.dataset.export;
      const a = document.createElement('a');
      a.href = API.history.export(fmt);
      a.download = `sanketra-history-${todayISO()}.${fmt === 'md' ? 'md' : fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }));

    $('#clear-all').addEventListener('click', () => {
      openConfirm({
        eyebrow: 'Danger',
        title: 'Clear EVERYTHING?',
        body: 'All sessions + transcripts will be deleted permanently. <strong>No undo.</strong><br><br>Type your consent — click the red button again to confirm.',
        go: () => {
          // Double-confirm: close, then re-open with different copy
          openConfirm({
            eyebrow: 'Last chance',
            title: 'Really sure?',
            body: 'Isske baad wapas nahi aa sakta. <em>All history wiped.</em>',
            go: async () => {
              try {
                // Codex F-Apr21-07: server requires ?confirm=yes guard.
              await fetchJSON(API.history.delAll + '?confirm=yes', { method: 'DELETE' });
                toast('All history cleared', 'ok');
                loadView(state.currentView);
              } catch (err) {
                toast(err.status === 404 ? 'Endpoint not wired yet' : 'Could not clear', 'err');
              }
            }
          });
        }
      });
    });
  }

  function wireGlobalActions() {
    // Transcript actions + session toggles (delegated)
    document.addEventListener('click', async (e) => {
      // Copy transcript
      const copyBtn = e.target.closest('.copy-t');
      if (copyBtn) {
        const text = copyBtn.dataset.text || '';
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.classList.add('copied');
          const orig = copyBtn.textContent;
          copyBtn.textContent = 'Copied';
          setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.textContent = orig;
          }, 1400);
        } catch (_) {
          toast('Copy failed — permission denied', 'err');
        }
        return;
      }

      // Delete transcript
      const delT = e.target.closest('.del-t');
      if (delT) {
        const id = delT.dataset.id;
        openConfirm({
          eyebrow: 'Delete transcript',
          title: 'Remove this one?',
          body: 'Sirf ye ek line hategi. Baaki session intact rahega.',
          go: async () => {
            try {
              await fetchJSON(API.history.delTranscript(id), { method: 'DELETE' });
              const node = document.querySelector(`[data-transcript-id="${id}"]`);
              if (node) node.remove();
              toast('Removed', 'ok');
            } catch (err) {
              toast(err.status === 404 ? 'Endpoint not wired yet' : 'Could not delete', 'err');
            }
          }
        });
        return;
      }

      // Delete session
      const delS = e.target.closest('.del-s');
      if (delS) {
        e.stopPropagation();
        const id = delS.dataset.id;
        openConfirm({
          eyebrow: 'Delete session',
          title: 'Delete entire session?',
          body: 'Sab transcripts is session ke delete ho jayenge.',
          go: async () => {
            try {
              await fetchJSON(API.history.del(id), { method: 'DELETE' });
              const node = document.querySelector(`[data-session-id="${id}"]`);
              if (node) node.remove();
              toast('Session deleted', 'ok');
            } catch (err) {
              toast(err.status === 404 ? 'Endpoint not wired yet' : 'Could not delete', 'err');
            }
          }
        });
        return;
      }

      // Toggle session collapse
      const sessHead = e.target.closest('[data-toggle-session]');
      if (sessHead) {
        // Don't collapse if we clicked the delete button inside the header
        if (e.target.closest('.del-s')) return;
        const sess = sessHead.closest('.session');
        if (sess) {
          sess.classList.toggle('collapsed');
          const id = sess.dataset.sessionId;
          if (sess.classList.contains('collapsed')) state.collapsedSessions.add(id);
          else state.collapsedSessions.delete(id);
        }
      }

      // Modal close
      if (e.target.matches('[data-close]')) {
        const modal = e.target.closest('.modal');
        if (modal) modal.hidden = true;
      }
    });

    // Esc to close modals
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        $$('.modal').forEach(m => { m.hidden = true; });
      }
    });
  }

  // ---------- modal ------------------------------------------------------
  function openModal(sel) {
    const m = $(sel);
    if (m) m.hidden = false;
  }
  function closeModal(sel) {
    const m = $(sel);
    if (m) m.hidden = true;
  }
  function openConfirm(opts) {
    const m = $('#confirm-modal');
    $('#cmod-eyebrow').textContent = opts.eyebrow || 'Confirm';
    $('#cmod-title').textContent = opts.title || 'Sure?';
    $('#cmod-body').innerHTML = opts.body || '';
    const go = $('#cmod-go');
    const fresh = go.cloneNode(true);
    go.parentNode.replaceChild(fresh, go);
    fresh.addEventListener('click', () => {
      m.hidden = true;
      if (opts.go) opts.go();
    });
    m.hidden = false;
  }

  // ---------- clock + version -------------------------------------------
  function startClock() {
    const el = $('#clock');
    function tick() { if (el) el.textContent = formatClock(); }
    tick();
    setInterval(tick, 30_000);
  }

  async function loadVersion() {
    try {
      const v = await fetchJSON(API.version);
      const el = $('#version-tag');
      if (el) el.textContent = `v${v.version || v}`;
    } catch (_) {
      const el = $('#version-tag');
      if (el) el.textContent = '';
    }
  }

  // ---------- boot -------------------------------------------------------
  function boot() {
    wireTabs();
    wirePrivacy();
    wireSearch();
    wireAllPagination();
    wireVocabControls();
    wireSettings();
    wireGlobalActions();
    startClock();
    loadVersion();

    // Restore last view (skip search — requires input)
    let last = 'today';
    try {
      const saved = localStorage.getItem('sanketra:lastView');
      if (saved && saved !== 'search' && $(`#view-${saved}`)) last = saved;
    } catch (_) {}
    setView(last);

    connectWS();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
