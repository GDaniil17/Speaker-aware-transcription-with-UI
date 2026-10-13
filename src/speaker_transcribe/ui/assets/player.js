// Media player + editable transcript, rendered inside a Streamlit component.
// Globals injected before this script: SEGS, SPEAKERS, PALETTE, LABELS.
// Editing and exporting happen in the browser, so they never rerun the app.
const player = document.getElementById('player');
// Video the browser can't decode (e.g. HEVC): switch to the MP3 copy the
// server encodes in the background, waiting for it if it isn't ready yet.
// Same element, so timestamp seeking keeps working.
function audioFallback() {
  const alt = player && player.dataset.fallback;
  if (!alt || player.dataset.fellBack) return;
  player.dataset.fellBack = '1';
  player.style.height = '54px';
  player.style.background = 'transparent';
  const note = document.createElement('div');
  note.style.cssText = 'opacity:.7;font-size:.85rem;margin:6px 0;';
  note.textContent = LABELS.fallbackPreparing;
  player.after(note);
  const swap = () => fetch(alt, {method: 'HEAD', cache: 'no-store'}).then((r) => {
    // A missing static file comes back as the app's HTML page, not a 404.
    if (r.ok && (r.headers.get('content-type') || '').startsWith('audio/')) {
      player.src = alt;
      note.textContent = LABELS.fallbackPlaying;
    } else {
      setTimeout(swap, 2000);
    }
  }).catch(() => setTimeout(swap, 2000));
  swap();
}
if (player) {
  player.addEventListener('error', audioFallback);
  if (player.error) audioFallback();
}
const list = document.getElementById('transcript');
const caption = document.getElementById('caption');
const spkBar = document.getElementById('speakers');
let spkSeq = SPEAKERS.length;
let rows = [];
let activeIdx = -1;

function getSpk(key) {
  for (const s of SPEAKERS) if (s.key === key) return s;
  return { key: key, name: key, color: '#888888' };
}
function pad(n, l) { return String(n).padStart(l, '0'); }
function fmt(t) {
  t = Math.max(0, t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  const ss = s.toFixed(1).padStart(4, '0');
  return h ? h + ':' + pad(m, 2) + ':' + ss : pad(m, 2) + ':' + ss;
}
function tsCode(t, sep) {
  t = Math.max(0, t);
  let h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  let ms = Math.round((t - Math.floor(t)) * 1000);
  if (ms === 1000) { s += 1; ms = 0; }
  return pad(h, 2) + ':' + pad(m, 2) + ':' + pad(s, 2) + sep + pad(ms, 3);
}
function seek(t) {
  if (!player) return;
  player.currentTime = t;
  const p = player.play();
  if (p && p.catch) p.catch(() => {});
}

function newSpeaker(name) {
  spkSeq += 1;
  const sp = {
    key: 'spk_' + spkSeq, name: name,
    color: PALETTE[SPEAKERS.length % PALETTE.length],
  };
  SPEAKERS.push(sp);
  return sp;
}

function renderSpeakerBar() {
  spkBar.innerHTML = '';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = LABELS.speakers;
  spkBar.appendChild(lbl);
  SPEAKERS.forEach((sp) => {
    const chip = document.createElement('span');
    chip.className = 'spkchip';
    const col = document.createElement('input');
    col.type = 'color';
    col.value = sp.color;
    const nm = document.createElement('input');
    nm.type = 'text';
    nm.value = sp.name;
    nm.style.color = sp.color;
    col.addEventListener('input', () => {
      sp.color = col.value;
      nm.style.color = sp.color;
      renderTranscript();
    });
    nm.addEventListener('input', () => {
      sp.name = nm.value || sp.key;
      renderTranscript();
    });
    chip.appendChild(col);
    chip.appendChild(nm);
    spkBar.appendChild(chip);
  });
  const add = document.createElement('button');
  add.id = 'addspk';
  add.textContent = '+ ' + LABELS.addSpeaker;
  add.addEventListener('click', () => {
    newSpeaker(LABELS.speaker + ' ' + (spkSeq + 1));
    render();
  });
  spkBar.appendChild(add);
}

function makeSelect(seg) {
  const sel = document.createElement('select');
  sel.className = 'spk';
  SPEAKERS.forEach((sp) => {
    const o = document.createElement('option');
    o.value = sp.key;
    o.textContent = sp.name;
    o.style.background = sp.color;
    o.style.color = '#ffffff';
    sel.appendChild(o);
  });
  const nw = document.createElement('option');
  nw.value = '__new__';
  nw.textContent = '+ ' + LABELS.newSpeaker;
  sel.appendChild(nw);
  sel.value = seg.spk;
  sel.style.background = getSpk(seg.spk).color;
  sel.addEventListener('change', () => {
    if (sel.value === '__new__') {
      const name = prompt(LABELS.newSpeakerPrompt);
      if (name) seg.spk = newSpeaker(name).key;
      render();
      return;
    }
    seg.spk = sel.value;
    sel.style.background = getSpk(seg.spk).color;
    renderTranscript();
  });
  return sel;
}

function renderTranscript() {
  list.innerHTML = '';
  rows = [];
  SEGS.forEach((seg) => {
    const row = document.createElement('div');
    row.className = 'line';
    const ts = document.createElement('span');
    ts.className = 'ts';
    ts.textContent = fmt(seg.start);
    ts.title = LABELS.jumpTo + ' ' + fmt(seg.start);
    ts.addEventListener('click', () => seek(seg.start));
    const tx = document.createElement('span');
    tx.className = 'tx';
    tx.textContent = seg.text;
    tx.contentEditable = 'true';
    tx.spellcheck = true;
    tx.addEventListener('input', () => {
      seg.text = tx.textContent || '';
    });
    row.appendChild(ts);
    row.appendChild(makeSelect(seg));
    row.appendChild(tx);
    list.appendChild(row);
    rows.push(row);
  });
  activeIdx = -1;
}

function render() { renderSpeakerBar(); renderTranscript(); }

function currentIndex(t) {
  let idx = -1;
  for (let i = 0; i < SEGS.length; i++) {
    if (t >= SEGS[i].start && t < SEGS[i].end) return i;
    if (t >= SEGS[i].start) idx = i;
  }
  return idx;
}
function sync(scroll) {
  if (!player) return;
  const idx = currentIndex(player.currentTime);
  if (idx !== activeIdx) {
    if (rows[activeIdx]) rows[activeIdx].classList.remove('active');
    activeIdx = idx;
    if (rows[idx]) {
      rows[idx].classList.add('active');
      if (scroll) rows[idx].scrollIntoView({ block: 'nearest' });
    }
  }
  if (caption) {
    const seg = SEGS[idx];
    caption.innerHTML = seg ? '<span>' + seg.text.replace(/</g, '&lt;') + '</span>' : '';
  }
}
if (player) {
  player.addEventListener('timeupdate', () => sync(true));
  ['seeked', 'play', 'pause', 'loadedmetadata'].forEach(
    (ev) => player.addEventListener(ev, () => sync(false))
  );
}

function withSpk() { return document.getElementById('ex_spk').checked; }
function toTxt() {
  const spk = withSpk();
  const out = [];
  let prev = null;
  SEGS.forEach((s) => {
    if (spk) {
      const n = getSpk(s.spk).name;
      if (n !== prev) { out.push('\n[' + n + ']'); prev = n; }
    }
    out.push(s.text);
  });
  return out.join('\n').trim() + '\n';
}
function toSrt() {
  const spk = withSpk();
  return SEGS.map((s, i) =>
    (i + 1) + '\n' + tsCode(s.start, ',') + ' --> ' + tsCode(s.end, ',') +
    '\n' + (spk ? '[' + getSpk(s.spk).name + '] ' : '') + s.text + '\n'
  ).join('\n');
}
function toVtt() {
  const spk = withSpk();
  const out = ['WEBVTT\n'];
  SEGS.forEach((s) => {
    out.push(
      tsCode(s.start, '.') + ' --> ' + tsCode(s.end, '.') +
      '\n' + (spk ? '<v ' + getSpk(s.spk).name + '>' : '') + s.text + '\n'
    );
  });
  return out.join('\n');
}
function toJson() {
  const spk = withSpk();
  const data = SEGS.map((s) => spk
    ? { start: s.start, end: s.end, speaker: getSpk(s.spk).name, text: s.text }
    : { start: s.start, end: s.end, text: s.text });
  return JSON.stringify(data, null, 2) + '\n';
}
function download(name, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const EXPORTS = { ex_txt: toTxt, ex_srt: toSrt, ex_vtt: toVtt, ex_json: toJson };
Object.keys(EXPORTS).forEach((id) => {
  document.getElementById(id).addEventListener('click', () => {
    download('transcript.' + id.slice(3), EXPORTS[id]());
  });
});

render();
