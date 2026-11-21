import './style.css';
import { Motion } from './motion.js';
import { Viewer } from './viewer.js';

const el = Object.fromEntries([...document.querySelectorAll('[id]')].map((node) => [node.id, node]));
const text = {
  queued: '대기 중', loading: 'Kimodo 모델 로드 중', generating: 'Kimodo 추론 중',
  exporting: '파일 저장 중', ready: '생성 완료', failed: '생성 실패',
};
let viewer;
let selectedId;
let readySelection;
let takes = [];
let pendingId;
let accepting = false;
let connected = false;
let constraints = [];
let constraintsRevision = 0;
let pollTimer;
let refreshRevision = 0;

function message(node, value, error = false) {
  node.textContent = value;
  node.hidden = !value;
  node.dataset.error = String(error);
}

const PENDING = ['queued', 'loading', 'generating', 'exporting'];
const busy = (status) => PENDING.includes(status);

function updateFrame(frame, playing) {
  el.timeline.value = Math.floor(frame);
  el.timecode.textContent = `${(frame / viewer.motion.fps).toFixed(2)} s`;
  el.play.textContent = playing ? '일시정지' : '재생';
}

try {
  viewer = new Viewer(el.viewport, updateFrame);
} catch (error) {
  console.error(error);
  message(el['empty-stage'], '3D 미리보기에 WebGL 2가 필요합니다.', true);
  el['vrm-file'].disabled = true;
  el['reset-view'].disabled = true;
  el['skeleton-visible'].disabled = el['root-motion'].disabled = true;
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function updateGenerate() {
  el.generate.disabled = !connected || accepting || Boolean(pendingId);
}

function renderTakes() {
  el['take-count'].textContent = takes.length;
  el['take-list'].replaceChildren(...takes.map((take, index) => {
    const row = document.createElement('div');
    row.className = 'take-item';
    row.dataset.takeId = take.id;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'take-main';
    button.setAttribute('aria-pressed', String(take.id === selectedId));
    const title = document.createElement('span');
    title.className = 'take-title';
    title.textContent = take.prompt;
    title.title = take.prompt;
    const meta = document.createElement('span');
    meta.className = 'take-meta';
    meta.dataset.error = String(take.status === 'failed');
    const duration = take.frame_count ? take.frame_count / take.fps : take.duration;
    const status = take.status === 'ready' ? `${take.fps} fps` : text[take.status];
    meta.textContent = `${duration.toFixed(1)}초 · 시드 ${take.seed} · ${status}`;
    button.append(title, meta);
    const move = (direction, label, disabled) => {
      const node = document.createElement('button');
      node.type = 'button';
      node.className = 'take-move';
      node.dataset.move = String(direction);
      node.textContent = direction < 0 ? '↑' : '↓';
      node.setAttribute('aria-label', label);
      node.disabled = disabled;
      return node;
    };
    const inactive = busy(take.status);
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'take-delete';
    del.dataset.delete = 'true';
    del.textContent = '삭제';
    del.setAttribute('aria-label', '이 테이크 삭제');
    del.disabled = inactive;
    row.append(
      button,
      move(-1, '이전 순서로', index === 0 || inactive),
      move(1, '다음 순서로', index === takes.length - 1 || inactive),
      del,
    );
    return row;
  }));
  if (!takes.length) {
    const empty = document.createElement('p');
    empty.className = 'message';
    empty.textContent = '저장된 테이크가 없습니다.';
    el['take-list'].append(empty);
  }
}

function takeUrl(id, format = '') {
  return `/api/takes/${id}${format ? '/' + format : ''}`;
}

async function selectTake(take, manual = false) {
  if (manual) readySelection = undefined;
  selectedId = take.id;
  renderTakes();
  el['take-actions'].hidden = false;
  el.exports.replaceChildren();
  if (take.status !== 'ready') {
    message(el['preview-status'], take.error || text[take.status], take.status === 'failed');
    return;
  }
  for (const format of ['json', 'bvh', 'npz']) {
    const link = document.createElement('a');
    link.href = takeUrl(take.id, format) + '?download=1';
    link.textContent = format.toUpperCase();
    link.setAttribute('aria-label', `${link.textContent} 내려받기`);
    el.exports.append(link);
  }
  if (!viewer) return;
  message(el['preview-status'], '테이크 불러오는 중');
  try {
    const raw = await api(takeUrl(take.id, 'json'));
    if (selectedId !== take.id) return;
    const motion = new Motion(raw);
    viewer.setMotion(motion);
    el['empty-stage'].hidden = true;
    el.timeline.max = motion.frameCount - 1;
    el.timeline.disabled = el.play.disabled = motion.frameCount < 2;
    el['take-info'].textContent = `${take.id.slice(0, 8)} · ${motion.frameCount} 프레임 · ${motion.fps} fps`;
    message(el['preview-status'], '');
  } catch (error) {
    if (selectedId === take.id) message(el['preview-status'], error.message, true);
  }
}

async function refresh() {
  const revision = ++refreshRevision;
  clearTimeout(pollTimer);
  let nextPoll = 5000;
  try {
    const updated = await api('/api/takes');
    if (revision !== refreshRevision) return;
    const initial = !connected;
    connected = true;
    el.connection.dataset.state = 'online';
    el.connection.textContent = '서버 연결됨';
    const prior = pendingId;
    takes = updated;
    pendingId = takes.find((take) => busy(take.status))?.id;
    if (pendingId) {
      nextPoll = 900;
      const pending = takes.find((take) => take.id === pendingId);
      message(el['generation-status'], text[pending.status]);
    }
    const finished = takes.find((take) => take.id === prior && ['ready', 'failed'].includes(take.status));
    const fromHistory = initial && !selectedId && takes.find((take) => take.status === 'ready');
    const selected = takes.find((take) => take.id === selectedId);
    renderTakes();
    if (finished) {
      message(el['generation-status'], finished.error || text[finished.status], finished.status === 'failed');
      if (readySelection === finished.id || selectedId === finished.id) await selectTake(finished);
    } else if (fromHistory) {
      await selectTake(fromHistory);
    } else if (selected && selected.status !== 'ready') {
      message(el['preview-status'], selected.error || text[selected.status], selected.status === 'failed');
    }
  } catch (error) {
    if (revision !== refreshRevision) return;
    connected = false;
    el.connection.dataset.state = 'offline';
    el.connection.textContent = '서버 연결 끊김';
    nextPoll = 2500;
  } finally {
    if (revision === refreshRevision) {
      updateGenerate();
      clearTimeout(pollTimer);
      pollTimer = setTimeout(refresh, nextPoll);
    }
  }
}

el['generate-form'].addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!connected || accepting || pendingId) return;
  accepting = true;
  updateGenerate();
  message(el['generation-status'], '생성 요청 중');
  try {
    const take = await api('/api/takes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: el.prompt.value,
        duration: el.duration.valueAsNumber,
        seed: el.seed.value === '' ? null : el.seed.valueAsNumber,
        steps: el.steps.valueAsNumber,
        constraints,
      }),
    });
    takes = [take, ...takes.filter(({ id }) => id !== take.id)];
    pendingId = readySelection = take.id;
    renderTakes();
    await refresh();
  } catch (error) {
    message(el['generation-status'], error.message, true);
  } finally {
    accepting = false;
    updateGenerate();
  }
});

function clearPreview() {
  el['take-info'].textContent = '재생 중인 테이크 없음';
  el.timeline.max = 0;
  el.timeline.value = 0;
  el.timeline.disabled = el.play.disabled = true;
  el.timecode.textContent = '0.00 s';
  if (viewer) viewer.clearMotion();
  el['empty-stage'].hidden = Boolean(viewer?.vrm);
  message(el['preview-status'], '');
}

async function deleteTake(id) {
  const take = takes.find((entry) => entry.id === id);
  if (!take || busy(take.status)) return;
  if (!confirm('이 테이크를 삭제할까요?')) return;
  try {
    await api(takeUrl(id), { method: 'DELETE' });
    if (selectedId === id) {
      selectedId = undefined;
      el['take-actions'].hidden = true;
      clearPreview();
    }
    await refresh();
  } catch (error) {
    message(el['preview-status'], error.message, true);
  }
}

async function moveTake(id, direction) {
  const index = takes.findIndex((take) => take.id === id);
  const target = index + direction;
  if (index < 0 || target < 0 || target >= takes.length || busy(takes[index].status)) return;
  [takes[index], takes[target]] = [takes[target], takes[index]];
  renderTakes();
  try {
    await api('/api/takes/reorder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: takes.map((take) => take.id) }),
    });
  } catch (error) {
    message(el['preview-status'], error.message, true);
    await refresh();
  }
}

el['take-list'].addEventListener('click', (event) => {
  const row = event.target.closest('[data-take-id]');
  if (!row) return;
  const id = row.dataset.takeId;
  if (event.target.closest('[data-delete]')) {
    deleteTake(id);
    return;
  }
  const move = event.target.closest('[data-move]');
  if (move) {
    moveTake(id, Number(move.dataset.move));
    return;
  }
  const take = takes.find((entry) => entry.id === id);
  if (take) selectTake(take, true);
});

function setConstraints(value, name) {
  constraints = value;
  message(el['constraints-name'], name);
  el['clear-constraints'].hidden = !value.length;
}

el['constraints-file'].addEventListener('change', async (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  const revision = ++constraintsRevision;
  try {
    if (file.size > 2 * 1024 * 1024) throw new Error('제약 파일은 2 MB까지 불러올 수 있습니다.');
    const value = JSON.parse(await file.text());
    if (!Array.isArray(value) || value.some((item) => typeof item?.type !== 'string')) {
      throw new Error('Kimodo constraints.json 형식이 필요합니다.');
    }
    if (revision === constraintsRevision) {
      setConstraints(value, value.length ? `${file.name} · ${value.length}개 제약` : '제약 없음');
    }
  } catch (error) {
    if (revision === constraintsRevision) message(el['generation-status'], error.message, true);
  }
});

el['clear-constraints'].addEventListener('click', () => {
  constraintsRevision++;
  setConstraints([], '');
});

el.reuse.addEventListener('click', async () => {
  const take = takes.find(({ id }) => id === selectedId);
  if (!take) return;
  const revision = ++constraintsRevision;
  try {
    const value = take.constraint_count ? await api(takeUrl(take.id, 'constraints')) : [];
    if (revision !== constraintsRevision || selectedId !== take.id) return;
    el.prompt.value = take.prompt;
    el.duration.value = take.duration;
    el.steps.value = take.steps;
    el.seed.value = take.seed;
    setConstraints(value, value.length ? `저장된 제약 · ${value.length}개` : '');
    el.prompt.focus();
  } catch (error) {
    if (revision === constraintsRevision) message(el['preview-status'], error.message, true);
  }
});

el['vrm-file'].addEventListener('change', async (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  event.target.disabled = el['clear-avatar'].disabled = true;
  message(el['avatar-status'], 'VRM 불러오는 중');
  try {
    await viewer.loadVrm(file);
    el['empty-stage'].hidden = true;
    el['avatar-name'].hidden = el['clear-avatar'].hidden = false;
    el['avatar-name'].textContent = file.name;
    message(el['avatar-status'], '');
  } catch (error) {
    message(el['avatar-status'], error.message, true);
  } finally {
    event.target.disabled = el['clear-avatar'].disabled = false;
  }
});

el['clear-avatar'].addEventListener('click', () => {
  viewer.clearVrm();
  viewer.resetView();
  el['empty-stage'].hidden = Boolean(viewer.motion);
  el['avatar-name'].hidden = el['clear-avatar'].hidden = true;
  message(el['avatar-status'], '');
});
el.play.addEventListener('click', () => viewer.togglePlay());
el.timeline.addEventListener('input', () => viewer.seek(el.timeline.valueAsNumber));
el.speed.addEventListener('change', () => { if (viewer) viewer.speed = Number(el.speed.value); });
el.loop.addEventListener('change', () => { if (viewer) viewer.loop = el.loop.checked; });
el['reset-view'].addEventListener('click', () => viewer.resetView());
el['skeleton-visible'].addEventListener('change', () => {
  viewer.showSkeleton = el['skeleton-visible'].checked;
  viewer.updateVisibility();
  viewer.resetView();
});
el['root-motion'].addEventListener('change', () => {
  viewer.rootMotion = el['root-motion'].checked;
  viewer.updateVisibility();
  viewer.drawPose();
});

refresh();
