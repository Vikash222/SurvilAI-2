async function get(path) { const r = await fetch(path); if (!r.ok) throw new Error(await r.text()); return r.json(); }

function esc(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function load() {
  try {
    const [health, cameras, people, events] = await Promise.all([get('/api/health'), get('/api/cameras'), get('/api/people'), get('/api/events')]);
    document.querySelector('#health').textContent = health.status === 'ok' ? '● Local / Online' : 'Offline';
    document.querySelector('#camera-count').textContent = cameras.length;
    document.querySelector('#people-count').textContent = people.length;
    document.querySelector('#event-count').textContent = events.length;
    const body = document.querySelector('#events');
    body.innerHTML = events.length ? events.map(e => `<tr><td>${esc(e.occurred_at)}</td><td><span class="event">${esc(e.event_type)}</span></td><td>${esc(e.camera_id)}</td><td>${esc(e.person_id)}</td><td>${e.confidence == null ? '—' : Number(e.confidence).toFixed(3)}</td><td>${esc(e.track_id)}</td></tr>`).join('') : '<tr><td colspan="6">No events recorded.</td></tr>';
  } catch (err) {
    document.querySelector('#health').textContent = '● Dashboard error';
    document.querySelector('#events').innerHTML = `<tr><td colspan="6">${esc(err.message)}</td></tr>`;
  }
}

load();
setInterval(load, 5000);
