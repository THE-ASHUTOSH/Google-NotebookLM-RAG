// Frontend logic: upload a document, then chat with it. Plain DOM, no framework.

const $ = (id) => document.getElementById(id);

const els = {
  dropzone: $('dropzone'),
  file: $('file'),
  chunkSize: $('chunkSize'),
  chunkOverlap: $('chunkOverlap'),
  topK: $('topK'),
  statusPanel: $('statusPanel'),
  docName: $('docName'),
  docMeta: $('docMeta'),
  clearChat: $('clearChat'),
  messages: $('messages'),
  emptyState: $('emptyState'),
  composer: $('composer'),
  question: $('question'),
  send: $('send'),
  toast: $('toast'),
  chatTitle: $('chatTitle'),
};

// Wire slider value readouts.
for (const id of ['chunkSize', 'chunkOverlap', 'topK']) {
  const out = $(id + 'Out');
  els[id].addEventListener('input', () => (out.textContent = els[id].value));
}

let docLoaded = false;

// ---------- Toast ----------
let toastTimer;
function toast(msg, kind = '') {
  els.toast.textContent = msg;
  els.toast.className = 'toast ' + kind;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), 3500);
}

// ---------- Upload ----------
// The dropzone is a <label> wrapping the hidden file input, so clicking it
// opens the file picker natively — no manual els.file.click() (that would open
// the dialog twice). Drag-and-drop is handled separately below.
['dragover', 'dragenter'].forEach((e) =>
  els.dropzone.addEventListener(e, (ev) => {
    ev.preventDefault();
    els.dropzone.classList.add('drag');
  })
);
['dragleave', 'drop'].forEach((e) =>
  els.dropzone.addEventListener(e, (ev) => {
    ev.preventDefault();
    els.dropzone.classList.remove('drag');
  })
);
els.dropzone.addEventListener('drop', (ev) => {
  const f = ev.dataTransfer.files?.[0];
  if (f) uploadFile(f);
});
els.file.addEventListener('change', () => {
  if (els.file.files[0]) uploadFile(els.file.files[0]);
});

async function uploadFile(file) {
  const dz = els.dropzone;
  const original = dz.innerHTML;
  dz.innerHTML =
    `<span class="dz-text">Processing <b>${escapeHtml(file.name)}</b>…</span>` +
    `<div class="progress"><i></i></div>`;

  const form = new FormData();
  form.append('file', file);
  form.append('chunkSize', els.chunkSize.value);
  form.append('chunkOverlap', els.chunkOverlap.value);
  form.append('topK', els.topK.value);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');

    docLoaded = true;
    els.statusPanel.hidden = false;
    els.docName.textContent = data.docName;
    els.docMeta.textContent =
      `${data.numChunks} chunks · ${Number(data.chars).toLocaleString()} chars` +
      (data.reused ? ' · reused index' : '');
    els.chatTitle.textContent = `Chat · ${data.docName}`;
    els.emptyState?.remove();
    els.messages.innerHTML = '';
    els.question.disabled = false;
    els.send.disabled = false;
    els.question.focus();
    toast(data.reused ? 'Reused existing index' : 'Document indexed', 'ok');
  } catch (err) {
    toast(err.message, 'err');
  } finally {
    dz.innerHTML = original;
  }
}

// ---------- Clear chat ----------
els.clearChat.addEventListener('click', async () => {
  await fetch('/api/reset', { method: 'POST' });
  els.messages.innerHTML = '';
  toast('Chat cleared');
});

// ---------- Chat ----------
els.composer.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const q = els.question.value.trim();
  if (!q || !docLoaded) return;
  els.question.value = '';

  addMessage('user', q);
  const thinking = addThinking();
  setBusy(true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    thinking.remove();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    addAssistant(q, data);
  } catch (err) {
    thinking.remove();
    addMessage('assistant', '⚠ ' + err.message);
    toast(err.message, 'err');
  } finally {
    setBusy(false);
    els.question.focus();
  }
});

function setBusy(busy) {
  els.send.disabled = busy || !docLoaded;
  els.question.disabled = busy || !docLoaded;
}

// ---------- Rendering ----------
function addMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  els.messages.appendChild(wrap);
  scroll();
  return wrap;
}

function addThinking() {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  wrap.innerHTML = `<div class="bubble thinking">Retrieving · grading · (rewriting if needed) · answering…</div>`;
  els.messages.appendChild(wrap);
  scroll();
  return wrap;
}

function addAssistant(question, data) {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = data.answer;
  wrap.appendChild(bubble);

  if (data.subQuestions?.length > 1) {
    const note = document.createElement('div');
    note.className = 'rewrite-note';
    note.textContent = `🧩 Broken into ${data.subQuestions.length} sub-questions: ` + data.subQuestions.join(' · ');
    wrap.appendChild(note);
  }

  if (data.rewritten) {
    const note = document.createElement('div');
    note.className = 'rewrite-note';
    note.textContent = `🔁 Query rewritten to: ${data.queryUsed}`;
    wrap.appendChild(note);
  }

  if (data.sources?.length) wrap.appendChild(renderSources(data.sources));
  if (data.trace) wrap.appendChild(renderTrace(data.trace));

  els.messages.appendChild(wrap);
  scroll();
}

function renderSources(sources) {
  const d = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = `Sources used (${sources.length})`;
  d.appendChild(sum);
  sources.forEach((s, i) => {
    const box = document.createElement('div');
    box.className = 'source';
    const head = document.createElement('div');
    head.className = 'src-head';
    head.textContent = `[#${i + 1}] similarity ${s.score.toFixed(3)} · chunk #${
      s.metadata?.chunk_index ?? '?'
    }`;
    const body = document.createElement('div');
    body.className = 'src-text';
    body.textContent = s.text;
    box.append(head, body);
    d.appendChild(box);
  });
  return d;
}

function renderTrace(trace) {
  const d = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = `🔍 Trace · ${Math.round(trace.duration_ms)} ms · ${
    trace.events.length
  } steps · ${trace.tokens_in} in / ${trace.tokens_out} out tokens`;
  d.appendChild(sum);

  for (const ev of trace.events) {
    const line = document.createElement('div');
    line.className = 'trace-line' + (ev.error ? ' err' : '');
    const extras = [];
    for (const k of [
      'num_subquestions', 'attempts', 'num_documents', 'num_relevant', 'pool_size',
      'grounded', 'verify_retries', 'sub_questions', 'answer_chars', 'tokens_in', 'tokens_out',
    ]) {
      if (ev.summary[k] !== undefined) extras.push(`${k}=${ev.summary[k]}`);
    }
    if (ev.summary.query_used) extras.push(`query="${ev.summary.query_used}"`);
    line.innerHTML =
      `<b>${ev.node}</b> · ${Math.round(ev.duration_ms)} ms` +
      (extras.length ? ' · ' + extras.join(' · ') : '') +
      (ev.error ? ` · ❌ ${escapeHtml(ev.error)}` : '');
    d.appendChild(line);
  }
  return d;
}

function scroll() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------- Restore status on load ----------
(async () => {
  try {
    const res = await fetch('/api/status');
    const s = await res.json();
    if (!s.hasKey) toast('Server missing GEMINI_API_KEY — see .env.example', 'err');
    if (s.docName) {
      docLoaded = true;
      els.statusPanel.hidden = false;
      els.docName.textContent = s.docName;
      els.docMeta.textContent = `${s.numChunks} chunks`;
      els.chatTitle.textContent = `Chat · ${s.docName}`;
      els.topK.value = s.topK;
      $('topKOut').textContent = s.topK;
      els.emptyState?.remove();
      els.question.disabled = false;
      els.send.disabled = false;
    }
  } catch {
    /* server not ready yet */
  }
})();
