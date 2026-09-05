const $ = (id) => document.getElementById(id);

const canvas = $('canvasArea');
const ctx = canvas.getContext('2d');
const cx = canvas.width / 2;
const cy = canvas.height / 2;

const state = {
  draggingTarget: false,
  lastThetas: [0, 0, 0, 0, 0],
  viewScale: 1,
  viewZoom: 1,
  canvasPointers: new Map(),
  pinchStartDistance: 0,
  pinchStartZoom: 1,
};
const STD_JOINT_MAX = 5;
const DEFAULT_LINKS = [120, 100, 80, 65, 50];
const DEFAULT_ANGLES = [20, 20, 10, 0, 0];
const LEARNING_GUIDES = {
  FK: {
    label: 'FK 2D',
    summary: 'Move the joint angles to calculate the end-effector position from known robot geometry.',
    steps: [
      ['Read inputs', 'The solver uses each link length and joint angle in order from the base.'],
      ['Accumulate angle', 'Each link direction is the sum of the previous joint rotations.'],
      ['Add link vectors', 'X and Y are built by adding L cos(theta) and L sin(theta) for every link.'],
      ['Check condition', 'The Jacobian warning shows whether the arm is near a sensitive pose.'],
    ],
  },
  IK: {
    label: 'IK 2D',
    summary: 'Drag the target or type X/Y to find joint angles that place the end-effector near that point.',
    steps: [
      ['Check reach', 'The target must be inside the minimum and maximum reach zone.'],
      ['Solve angles', 'Two links use an analytic elbow solution; longer chains use CCD iteration.'],
      ['Compare error', 'Position error shows the distance between the target and final hand position.'],
      ['Choose posture', 'For two links, elbow up/down selects the alternate valid solution.'],
    ],
  },
  DH: {
    label: 'DH',
    summary: 'Build a 3D robot from Standard DH parameters and inspect the transform, frames, and Jacobian.',
    steps: [
      ['Fill DH row', 'Each joint row defines theta, d, a, and alpha for one transform.'],
      ['Multiply transforms', 'The final pose is T01 x T12 x ... x Tn from base to tool.'],
      ['Read the matrix', 'The last column is tool position; the first 3x3 block is orientation.'],
      ['Inspect Jacobian', 'Rank and condition number warn about singular or near-singular poses.'],
    ],
  },
  IK3D: {
    label: 'IK 3D',
    summary: 'Set an X/Y/Z target and let the damped least-squares solver move the DH robot toward it.',
    steps: [
      ['Set target', 'The orange marker is the desired 3D tool position.'],
      ['Measure error', 'The solver compares target position with the current end-effector position.'],
      ['Update joints', 'Damped least squares uses the position Jacobian to adjust joint angles.'],
      ['Confirm result', 'Small position error means the target is reached or closely approximated.'],
    ],
  },
};

const deg2rad = (d) => d * Math.PI / 180;
const rad2deg = (r) => r * 180 / Math.PI;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const normalizeAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function selectedValue(name) {
  return [...document.getElementsByName(name)].find((el) => el.checked).value;
}
function mode() { return selectedValue('mode'); }
function linkCount() { return Number(selectedValue('linkCount')); }
function elbowMode() { return selectedValue('elbow'); }
function numVal(id) { const v = Number($(id).value); return Number.isFinite(v) ? v : 0; }

function links() {
  return Array.from({ length: linkCount() }, (_, i) => clamp(numVal(`L${i + 1}`), 10, 240));
}

// ── Standard 2D FK ──
function fk(lv, angles) {
  let x = 0, y = 0, angle = 0;
  const joints = [{ x: 0, y: 0 }];
  for (let i = 0; i < lv.length; i++) {
    angle += angles[i];
    x += lv[i] * Math.cos(angle);
    y += lv[i] * Math.sin(angle);
    joints.push({ x, y });
  }
  return { x, y, joints };
}

// Position Jacobian singular values for a planar revolute chain.
function jacobianMetrics(lv, angles) {
  const cumulative = [];
  let angle = 0;
  for (let i = 0; i < lv.length; i++) {
    angle += angles[i] || 0;
    cumulative.push(angle);
  }

  // J is 2 x N. Build J*J^T directly to avoid a matrix dependency.
  let jxx = 0, jxy = 0, jyy = 0;
  for (let j = 0; j < lv.length; j++) {
    let dx = 0, dy = 0;
    for (let k = j; k < lv.length; k++) {
      dx -= lv[k] * Math.sin(cumulative[k]);
      dy += lv[k] * Math.cos(cumulative[k]);
    }
    jxx += dx * dx;
    jxy += dx * dy;
    jyy += dy * dy;
  }

  const trace = jxx + jyy;
  const det = Math.max(0, jxx * jyy - jxy * jxy);
  const discriminant = Math.sqrt(Math.max(0, trace * trace - 4 * det));
  const sigmaMax = Math.sqrt(Math.max(0, (trace + discriminant) / 2));
  const sigmaMin = Math.sqrt(Math.max(0, (trace - discriminant) / 2));
  const condition = sigmaMin < 1e-9 ? Infinity : sigmaMax / sigmaMin;
  const normalized = sigmaMin / Math.max(1, lv.reduce((sum, value) => sum + value, 0));
  const level = normalized < 0.005 ? 'singular' : normalized < 0.03 ? 'near' : 'stable';
  return { sigmaMin, sigmaMax, condition, normalized, level };
}

function updateSingularity(lv, angles) {
  const result = jacobianMetrics(lv, angles);
  const labels = { stable: 'Stable', near: 'Near Singularity', singular: 'Singular' };
  const notes = {
    stable: 'Position Jacobian is well-conditioned.',
    near: 'Motion sensitivity is high. Joint velocity may rise quickly.',
    singular: 'A Cartesian motion direction is lost at this configuration.',
  };
  const label = labels[result.level];
  $('singularityCard').className = `mini-card singularity-card ${result.level}`;
  $('singularityBadge').textContent = label;
  $('singularityState').textContent = label;
  $('singularityState').className = result.level;
  $('sigmaMin').textContent = result.sigmaMin.toFixed(3);
  $('conditionNumber').textContent = Number.isFinite(result.condition) ? result.condition.toFixed(1) : '∞';
  $('singularityNote').textContent = notes[result.level];
  $('singularityBarFill').style.width = `${clamp(result.normalized / 0.12 * 100, 2, 100)}%`;
  return result;
}

function ik2(l1, l2, x, y, elbow) {
  const r = Math.hypot(x, y);
  if (r > l1 + l2 || r < Math.abs(l1 - l2)) return { ok: false };
  const cos2 = clamp((r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1);
  const sin2 = Math.sqrt(Math.max(0, 1 - cos2 * cos2));
  const theta2 = Math.atan2(elbow === 'up' ? -sin2 : sin2, cos2);
  const theta1 = Math.atan2(y, x) - Math.atan2(l2 * Math.sin(theta2), l1 + l2 * Math.cos(theta2));
  return { ok: true, thetas: [theta1, theta2] };
}

function ccdIk(lv, tx, ty, init) {
  const th = init.slice(0, lv.length);
  while (th.length < lv.length) th.push(0);
  for (let iter = 0; iter < 90; iter++) {
    for (let j = lv.length - 1; j >= 0; j--) {
      const pts = fk(lv, th).joints;
      const cur = Math.atan2(pts[pts.length - 1].y - pts[j].y, pts[pts.length - 1].x - pts[j].x);
      const tgt = Math.atan2(ty - pts[j].y, tx - pts[j].x);
      th[j] = normalizeAngle(th[j] + normalizeAngle(tgt - cur));
    }
    const e = fk(lv, th);
    if (Math.hypot(e.x - tx, e.y - ty) < 0.35) break;
  }
  return { ok: true, thetas: th.map(normalizeAngle) };
}

// ══════════════════════════════════════
//  DH Defaults & State
// ══════════════════════════════════════
const DH_DEFAULTS = [
  { theta: 0,   d: 80,  a: 0,   alpha: -90 },
  { theta: 0,   d: 0,   a: 120, alpha: 0   },
  { theta: 0,   d: 0,   a: 100, alpha: 0   },
  { theta: 0,   d: 80,  a: 0,   alpha: -90 },
  { theta: 0,   d: 0,   a: 60,  alpha: 90  },
  { theta: 0,   d: 40,  a: 0,   alpha: 0   },
  { theta: 0,   d: 30,  a: 0,   alpha: 90  },
];
const stateDH = { jointCount: 3, dh: DH_DEFAULTS.slice(0, 3).map(d => ({ ...d })) };

// ── DH Matrix Math ──
function dhMat(thetaDeg, d, a, alphaDeg) {
  const t = deg2rad(thetaDeg), al = deg2rad(alphaDeg);
  const ct = Math.cos(t), st = Math.sin(t), ca = Math.cos(al), sa = Math.sin(al);
  return [
    [ct, -st*ca,  st*sa, a*ct],
    [st,  ct*ca, -ct*sa, a*st],
    [0,   sa,     ca,    d   ],
    [0,   0,      0,     1   ],
  ];
}
function mulMat(A, B) {
  const C = Array.from({length:4}, ()=>Array(4).fill(0));
  for (let i=0;i<4;i++) for (let j=0;j<4;j++) for (let k=0;k<4;k++) C[i][j]+=A[i][k]*B[k][j];
  return C;
}
function fkDH(dhParams) {
  let T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const joints = [{ x:0, y:0, z:0 }];
  const matrices = [];
  for (const d of dhParams) {
    T = mulMat(T, dhMat(d.theta, d.d, d.a, d.alpha));
    matrices.push(T);
    joints.push({ x: T[0][3], y: T[1][3], z: T[2][3] });
  }
  const end = joints[joints.length - 1];
  return { x: end.x, y: end.y, z: end.z, joints, matrices };
}

// Full geometric Jacobian for the current all-revolute Standard-DH chain.
// Linear rows are normalized by a characteristic length for condition metrics,
// so millimetres and radians do not distort the singular-value comparison.
function dhJacobianMetrics(dhParams, fkResult) {
  const n = dhParams.length;
  const end = [fkResult.x, fkResult.y, fkResult.z];
  const J = Array.from({ length: 6 }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const Tprev = i === 0 ? null : fkResult.matrices[i - 1];
    const origin = Tprev ? [Tprev[0][3], Tprev[1][3], Tprev[2][3]] : [0, 0, 0];
    const z = Tprev ? [Tprev[0][2], Tprev[1][2], Tprev[2][2]] : [0, 0, 1];
    const r = [end[0] - origin[0], end[1] - origin[1], end[2] - origin[2]];
    J[0][i] = z[1] * r[2] - z[2] * r[1];
    J[1][i] = z[2] * r[0] - z[0] * r[2];
    J[2][i] = z[0] * r[1] - z[1] * r[0];
    J[3][i] = z[0]; J[4][i] = z[1]; J[5][i] = z[2];
  }

  const length = Math.max(1, dhParams.reduce((s, d) => s + Math.abs(d.a) + Math.abs(d.d), 0));
  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) =>
    J.reduce((sum, row, k) => sum + (k < 3 ? row[i] / length : row[i]) * (k < 3 ? row[j] / length : row[j]), 0)));

  // Jacobi eigenvalue iteration for the small symmetric J^T J matrix (N <= 7).
  for (let iter = 0; iter < 80; iter++) {
    let p = 0, q = 1, largest = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (Math.abs(A[i][j]) > largest) { largest = Math.abs(A[i][j]); p = i; q = j; }
    }
    if (largest < 1e-12) break;
    const angle = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
    const c = Math.cos(angle), s = Math.sin(angle);
    const app = c*c*A[p][p] - 2*s*c*A[p][q] + s*s*A[q][q];
    const aqq = s*s*A[p][p] + 2*s*c*A[p][q] + c*c*A[q][q];
    for (let k = 0; k < n; k++) if (k !== p && k !== q) {
      const akp = A[k][p], akq = A[k][q];
      A[k][p] = A[p][k] = c*akp - s*akq;
      A[k][q] = A[q][k] = s*akp + c*akq;
    }
    A[p][p] = app; A[q][q] = aqq; A[p][q] = A[q][p] = 0;
  }
  const singularValues = A.map((row, i) => Math.sqrt(Math.max(0, row[i]))).sort((a,b) => b-a);
  const expectedRank = Math.min(6, n);
  const sigmaMax = singularValues[0] || 0;
  // For redundant chains (N > 6), ignore the expected joint-space null value.
  const sigmaMin = singularValues[expectedRank - 1] || 0;
  const condition = sigmaMin < 1e-6 ? Infinity : sigmaMax / sigmaMin;
  const rank = Math.min(6, singularValues.filter(v => v > Math.max(1e-6, sigmaMax * 1e-4)).length);
  const ratio = sigmaMax > 0 ? sigmaMin / sigmaMax : 0;
  const level = rank < expectedRank || ratio < 0.005 ? 'singular' : ratio < 0.03 ? 'near' : 'stable';
  return { J, sigmaMin, condition, rank, expectedRank, level };
}

function inverse3(M) {
  const [a,b,c] = M[0], [d,e,f] = M[1], [g,h,i] = M[2];
  const A=e*i-f*h, B=c*h-b*i, C=b*f-c*e;
  const D=f*g-d*i, E=a*i-c*g, F=c*d-a*f;
  const G=d*h-e*g, H=b*g-a*h, I=a*e-b*d;
  const det = a*A + b*D + c*G;
  if (Math.abs(det) < 1e-12) return null;
  return [[A,B,C],[D,E,F],[G,H,I]].map(row => row.map(v => v / det));
}

function cloneDhParams(dhParams) {
  return dhParams.map(d => ({ ...d }));
}

// Position-only 3D IK using damped least squares:
// dq = Jv^T (Jv Jv^T + lambda^2 I)^-1 error.
function solveIK3DOnParams(dhParams, target) {
  const n = dhParams.length;
  if (dhParams.every(d => Math.abs(d.theta) < 0.001)) {
    const seed = [10,-20,25,15,-10,10,5];
    dhParams.forEach((d, idx) => { d.theta = seed[idx]; });
  }
  let error = Infinity, iterations = 0;
  const lambda = 6;
  for (; iterations < 160; iterations++) {
    const res = fkDH(dhParams);
    const err = [target.x-res.x, target.y-res.y, target.z-res.z];
    error = Math.hypot(...err);
    if (error < 0.5) break;
    const J = dhJacobianMetrics(dhParams, res).J.slice(0, 3);
    const B = Array.from({length:3}, (_, r) => Array.from({length:3}, (_, c) =>
      J[r].reduce((sum, value, k) => sum + value * J[c][k], 0) + (r === c ? lambda * lambda : 0)));
    const inv = inverse3(B);
    if (!inv) break;
    const y = inv.map(row => row.reduce((sum, value, k) => sum + value * err[k], 0));
    for (let joint = 0; joint < n; joint++) {
      const dq = J.reduce((sum, row, r) => sum + row[joint] * y[r], 0);
      dhParams[joint].theta = clamp(dhParams[joint].theta + rad2deg(clamp(dq, -0.22, 0.22)), -180, 180);
    }
  }
  return { error, iterations, converged: error < 1 };
}

function solveIK3D(target) {
  return solveIK3DOnParams(stateDH.dh, target);
}

// ══════════════════════════════════════
//  2D Drawing
// ══════════════════════════════════════
const LINK_COLORS = ['#18202b','#df3f3f','#0f9f8f','#7c3aed','#2563eb','#ea580c','#db2777'];

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fbfdff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#e8eef5'; ctx.lineWidth = 1;
  for (let i = 0; i <= canvas.width; i += 30) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke();
  }
  ctx.strokeStyle = '#9aa9ba'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height); ctx.stroke();
}

function drawReachZone(lv) {
  if (!$('showReach').checked) return;
  const mx = lv.reduce((s,l)=>s+l,0), lg = Math.max(...lv);
  const mn = Math.max(0, lg - (mx - lg));
  ctx.save(); ctx.translate(cx, cy);
  ctx.fillStyle = $('showWorkspaceFill').checked ? 'rgba(37,99,235,0.06)' : 'rgba(37,99,235,0)';
  ctx.strokeStyle = 'rgba(37,99,235,0.22)'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(0, 0, mx * state.viewScale, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  if (mn > 0) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath(); ctx.arc(0, 0, mn * state.viewScale, 0, Math.PI*2); ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = 'rgba(183,121,31,0.35)';
    ctx.beginPath(); ctx.arc(0, 0, mn * state.viewScale, 0, Math.PI*2); ctx.stroke();
  }
  ctx.restore();
}

function solve2DForTarget(lv, tx, ty, initialThetas) {
  if (lv.length === 2) {
    const sol = ik2(lv[0], lv[1], tx, ty, elbowMode());
    return sol.ok ? sol.thetas : initialThetas.slice(0, 2);
  }
  return ccdIk(lv, tx, ty, initialThetas).thetas;
}

function drawTrajectory(lv, target) {
  if (!$('showTrajectory').checked || mode() !== 'IK' || !target) return null;
  const start = { x: numVal('trajStartX'), y: numVal('trajStartY') };
  const steps = clamp(numVal('trajStepsIn'), 8, 48);
  const path = [];
  let seed = state.lastThetas.slice(0, lv.length);
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const x = start.x + (target.x - start.x) * t;
    const y = start.y + (target.y - start.y) * t;
    seed = solve2DForTarget(lv, x, y, seed);
    const joints = fk(lv, seed).joints;
    const end = joints[joints.length - 1];
    path.push({ x: end.x, y: end.y });
  }

  ctx.save();
  path.forEach((p, i) => {
    const isStart = i === 0;
    const isEnd = i === path.length - 1;
    ctx.fillStyle = isStart ? '#ffffff' : '#0f9f8f';
    ctx.strokeStyle = '#0f9f8f';
    ctx.lineWidth = isStart || isEnd ? 2 : 0;
    ctx.beginPath();
    ctx.arc(cx + p.x * state.viewScale, cy - p.y * state.viewScale, isStart || isEnd ? 5 : 2.6, 0, Math.PI * 2);
    ctx.fill();
    if (isStart || isEnd) ctx.stroke();
  });
  ctx.restore();
  return path;
}

function drawObstacle() {
  if (!$('showObstacle').checked) return null;
  const obstacle = {
    x: numVal('obstacleX'),
    y: numVal('obstacleY'),
    r: clamp(numVal('obstacleR'), 10, 120),
  };
  const px = cx + obstacle.x * state.viewScale;
  const py = cy - obstacle.y * state.viewScale;
  ctx.save();
  ctx.fillStyle = 'rgba(223,63,63,0.08)';
  ctx.strokeStyle = 'rgba(223,63,63,0.58)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py, obstacle.r * state.viewScale, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#df3f3f';
  ctx.beginPath();
  ctx.arc(px, py, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  return obstacle;
}

function distancePointToSegment(point, a, b) {
  const vx = b.x - a.x, vy = b.y - a.y;
  const wx = point.x - a.x, wy = point.y - a.y;
  const len2 = vx * vx + vy * vy;
  const t = len2 > 0 ? clamp((wx * vx + wy * vy) / len2, 0, 1) : 0;
  const px = a.x + vx * t, py = a.y + vy * t;
  return Math.hypot(point.x - px, point.y - py);
}

function pathClearance(points, obstacle) {
  if (!points || points.length < 2 || !obstacle) return Infinity;
  let minDistance = Infinity;
  for (let i = 0; i < points.length - 1; i++) {
    minDistance = Math.min(minDistance, distancePointToSegment(obstacle, points[i], points[i + 1]) - obstacle.r);
  }
  return minDistance;
}

function collisionStatus(points, obstacle, path = null) {
  if (!obstacle) return { level: 'clear', label: 'Off', distance: Infinity };
  const linkDistance = pathClearance(points, obstacle);
  const pathDistance = pathClearance(path, obstacle);
  const minDistance = Math.min(linkDistance, pathDistance);
  const nearClearance = Math.max(24, obstacle.r * 0.75);
  if (minDistance <= 0) return { level: 'crossing', label: 'Crossing', distance: minDistance };
  if (minDistance <= nearClearance) return { level: 'near', label: 'Near', distance: minDistance };
  return { level: 'clear', label: 'Clear', distance: minDistance };
}

function updatePlanarMatrix(angles, end) {
  const phi = angles.reduce((sum, value) => sum + value, 0);
  const c = Math.cos(phi), s = Math.sin(phi);
  const f = v => v.toFixed(3).padStart(7);
  $('planarMatrix').textContent =
    `${f(c)} ${f(-s)} ${f(end.x)}\n` +
    `${f(s)} ${f(c)} ${f(end.y)}\n` +
    `${f(0)} ${f(0)} ${f(1)}`;
}

function drawTarget(x, y, ok) {
  const px = cx + x * state.viewScale, py = cy - y * state.viewScale;
  ctx.save();
  ctx.strokeStyle = ok ? '#2563eb' : '#df3f3f';
  ctx.fillStyle = ok ? 'rgba(37,99,235,0.14)' : 'rgba(223,63,63,0.14)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 11, 0, Math.PI*2); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(px-17,py); ctx.lineTo(px+17,py);
  ctx.moveTo(px,py-17); ctx.lineTo(px,py+17); ctx.stroke();
  ctx.restore();
}

function drawArm2D(points, count) {
  ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (let i = 0; i < count; i++) {
    ctx.strokeStyle = LINK_COLORS[i % LINK_COLORS.length];
    ctx.lineWidth = Math.max(5, 13 - i * 2);
    ctx.beginPath();
    ctx.moveTo(cx + points[i].x * state.viewScale, cy - points[i].y * state.viewScale);
    ctx.lineTo(cx + points[i+1].x * state.viewScale, cy - points[i+1].y * state.viewScale);
    ctx.stroke();
  }
  points.forEach((p, i) => {
    ctx.fillStyle = i === points.length-1 ? '#2563eb' : '#ffffff';
    ctx.strokeStyle = i === 0 ? '#18202b' : '#435266'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx + p.x * state.viewScale, cy - p.y * state.viewScale, i === points.length-1 ? 9 : 8, 0, Math.PI*2);
    ctx.fill(); ctx.stroke();
  });
  const end = points[points.length - 1];
  ctx.fillStyle = '#18202b';
  ctx.font = '700 12px Inter, system-ui, sans-serif';
  ctx.fillText('EE', cx + end.x * state.viewScale + 12, cy - end.y * state.viewScale + 18);
  ctx.restore();
  return points[points.length - 1];
}

function setStatus(msg, t = '') {
  $('statusText').textContent = msg;
  $('reachStatus').className = `status-card ${t}`.trim();
}

function updateLearningGuide() {
  const guide = LEARNING_GUIDES[mode()];
  if (!guide) return;
  $('guideMode').textContent = guide.label;
  $('guideSummary').textContent = guide.summary;
  $('guideSteps').innerHTML = guide.steps.map(([title, detail], index) => `
    <div class="guide-step">
      <span>${index + 1}</span>
      <div><strong>${title}</strong><p>${detail}</p></div>
    </div>
  `).join('');
}

function buildJointOutput(count) {
  const c = $('jointOutput'); c.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const r = document.createElement('div'); r.className = 'joint-row';
    r.innerHTML = `<span>&theta;${i+1}</span><strong id="thOut${i}">0.00 deg</strong>`;
    c.appendChild(r);
  }
  const errRow = document.createElement('div');
  errRow.className = 'joint-row position-error-row';
  errRow.innerHTML = `<span>Position Error</span><strong id="jointPositionError">0.00</strong>`;
  c.appendChild(errRow);
}

function syncSlider(id, v) { $(`${id}In`).value = v; $(`${id}Range`).value = v; }

// ══════════════════════════════════════
//  DH Table & Sliders UI
// ══════════════════════════════════════
function buildDHTable() {
  const tb = $('dhBody'); tb.innerHTML = '';
  stateDH.dh.forEach((dh, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="dh-label">${i+1}</td>
      <td class="dh-cell joint-var"><input type="number" data-i="${i}" data-p="theta" value="${fmt2(dh.theta)}" step="0.01"></td>
      <td class="dh-cell"><input type="number" data-i="${i}" data-p="d" value="${fmt2(dh.d)}" step="0.01"></td>
      <td class="dh-cell"><input type="number" data-i="${i}" data-p="a" value="${fmt2(dh.a)}" step="0.01"></td>
      <td class="dh-cell"><input type="number" data-i="${i}" data-p="alpha" value="${fmt2(dh.alpha)}" step="0.01"></td>`;
    tb.appendChild(tr);
  });
  tb.querySelectorAll('input').forEach(inp => inp.addEventListener('input', e => {
    const i = +e.target.dataset.i, p = e.target.dataset.p;
    stateDH.dh[i][p] = Number(e.target.value) || 0;
    if (p === 'theta') { const r = $(`dhR${i}`), n = $(`dhN${i}`); if(r) r.value = stateDH.dh[i].theta; if(n) n.value = stateDH.dh[i].theta; }
    update();
  }));
}

function syncDHTable() {
  $('dhBody').querySelectorAll('input').forEach(inp => {
    const i = +inp.dataset.i, p = inp.dataset.p;
    inp.value = fmt2(p === 'theta' ? stateDH.dh[i].theta : stateDH.dh[i][p]);
  });
}

function buildDHSliders() {
  const c = $('dhSliders'); c.innerHTML = '';
  for (let i = 0; i < stateDH.jointCount; i++) {
    const d = document.createElement('div'); d.className = 'slider-field';
    d.innerHTML = `<label>&theta;${i+1}</label>
      <input type="range" id="dhR${i}" min="-180" max="180" step="0.01" value="${fmt2(stateDH.dh[i].theta)}">
      <input type="number" id="dhN${i}" step="0.01" value="${fmt2(stateDH.dh[i].theta)}">`;
    c.appendChild(d);
    const rng = d.querySelector(`#dhR${i}`), num = d.querySelector(`#dhN${i}`);
    rng.addEventListener('input', e => { stateDH.dh[i].theta = +e.target.value; num.value = fmt2(stateDH.dh[i].theta); syncDHTable(); update(); });
    num.addEventListener('input', e => { stateDH.dh[i].theta = +e.target.value || 0; rng.value = fmt2(clamp(stateDH.dh[i].theta,-180,180)); syncDHTable(); update(); });
  }
}

// ══════════════════════════════════════
//  3D Scene (Three.js)
// ══════════════════════════════════════
let scene, camera, renderer, orbit, targetMarker, trajectory3DGroup;
let armGroup, gridHelper, axesHelper;
const linkMeshes = [], jointMeshes = [], frameArrows = [];
const THREE_COLORS = [0x1e293b, 0xdc2626, 0x0d9488, 0x7c3aed, 0x2563eb, 0xea580c, 0xdb2777];

function init3D() {
  const c = $('threeContainer');
  // Force reflow so container has dimensions after display:none → block
  void c.offsetHeight;
  const w = c.clientWidth || 700, h = c.clientHeight || 560;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8faff);

  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 5000);
  camera.position.set(200, 180, 200);

  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  c.appendChild(renderer.domElement);

  // Minimal orbit: drag=rotate, right-drag/shift-drag=pan, scroll/pinch=zoom
  orbit = { target: new THREE.Vector3(0, 50, 0), dragging: false, button: -1,
    prev: {x:0,y:0}, spherical: new THREE.Spherical(), sphericalDelta: new THREE.Spherical(),
    panOffset: new THREE.Vector3(), rotateSpeed: 0.005, panSpeed: 0.5, zoomSpeed: 1.1,
    pointers: new Map(), pinchStartDistance: 0, pinchStartRadius: 0 };

  function updateOrbitCamera() {
    const offset = new THREE.Vector3().setFromSpherical(orbit.spherical);
    camera.position.copy(orbit.target).add(offset);
    camera.lookAt(orbit.target);
  }
  orbit.update = updateOrbitCamera;

  function sphericalFromCamera() {
    const offset = new THREE.Vector3().subVectors(camera.position, orbit.target);
    orbit.spherical.setFromVector3(offset);
  }
  sphericalFromCamera();

  const el = renderer.domElement;
  function rememberOrbitPointer(e) {
    orbit.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  }
  function orbitPinchDistance() {
    const pts = [...orbit.pointers.values()];
    if (pts.length < 2) return 0;
    return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
  }
  function endOrbitPointer(e) {
    orbit.pointers.delete(e.pointerId);
    orbit.dragging = orbit.pointers.size > 0;
    if (orbit.pointers.size < 2) orbit.pinchStartDistance = 0;
  }
  el.addEventListener('pointerdown', e => {
    e.preventDefault();
    rememberOrbitPointer(e);
    el.setPointerCapture(e.pointerId);
    if (orbit.pointers.size === 2) {
      orbit.dragging = false;
      orbit.pinchStartDistance = orbitPinchDistance();
      orbit.pinchStartRadius = orbit.spherical.radius;
      return;
    }
    orbit.dragging = true; orbit.button = e.button;
    orbit.prev = { x: e.clientX, y: e.clientY };
  });
  el.addEventListener('pointermove', e => {
    e.preventDefault();
    if (!orbit.pointers.has(e.pointerId)) return;
    rememberOrbitPointer(e);
    if (orbit.pointers.size >= 2 && orbit.pinchStartDistance > 0) {
      const distance = orbitPinchDistance();
      if (distance > 0) {
        orbit.spherical.radius = clamp(orbit.pinchStartRadius * (orbit.pinchStartDistance / distance), 30, 1500);
        updateOrbitCamera();
      }
      return;
    }
    if (!orbit.dragging) return;
    const dx = e.clientX - orbit.prev.x, dy = e.clientY - orbit.prev.y;
    orbit.prev = { x: e.clientX, y: e.clientY };
    if (orbit.button === 0 && e.shiftKey) { // shift+left = pan (CAD-style)
      const pan = new THREE.Vector3();
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      pan.addScaledVector(right, -dx * orbit.panSpeed);
      pan.addScaledVector(up, dy * orbit.panSpeed);
      orbit.target.add(pan);
    } else if (orbit.button === 0) { // left = rotate
      orbit.spherical.theta -= dx * orbit.rotateSpeed;
      orbit.spherical.phi -= dy * orbit.rotateSpeed;
      orbit.spherical.phi = clamp(orbit.spherical.phi, 0.05, Math.PI - 0.05);
    } else if (orbit.button === 2) { // right = pan
      const pan = new THREE.Vector3();
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      pan.addScaledVector(right, -dx * orbit.panSpeed);
      pan.addScaledVector(up, dy * orbit.panSpeed);
      orbit.target.add(pan);
    }
    updateOrbitCamera();
  });
  el.addEventListener('pointerup', endOrbitPointer);
  el.addEventListener('pointercancel', endOrbitPointer);
  el.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1 / orbit.zoomSpeed : orbit.zoomSpeed;
    orbit.spherical.radius = clamp(orbit.spherical.radius * factor, 30, 1500);
    updateOrbitCamera();
  }, { passive: false });
  el.addEventListener('contextmenu', e => e.preventDefault());

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const dl = new THREE.DirectionalLight(0xffffff, 0.8);
  dl.position.set(200, 300, 150); scene.add(dl);

  // Grid + axes
  gridHelper = new THREE.GridHelper(600, 20, 0xcbd5e1, 0xe2e8f0);
  scene.add(gridHelper);
  axesHelper = new THREE.AxesHelper(120);
  scene.add(axesHelper);

  armGroup = new THREE.Group();
  scene.add(armGroup);

  targetMarker = new THREE.Group();
  const targetSphere = new THREE.Mesh(
    new THREE.SphereGeometry(10, 18, 18),
    new THREE.MeshBasicMaterial({ color: 0xf97316, wireframe: true })
  );
  targetMarker.add(targetSphere);
  targetMarker.add(new THREE.AxesHelper(28));
  targetMarker.visible = false;
  scene.add(targetMarker);

  trajectory3DGroup = new THREE.Group();
  trajectory3DGroup.visible = false;
  scene.add(trajectory3DGroup);

  updateOrbitCamera();

  window.addEventListener('resize', () => {
    const w2 = c.clientWidth, h2 = c.clientHeight;
    camera.aspect = w2 / h2; camera.updateProjectionMatrix();
    renderer.setSize(w2, h2);
  });

  (function loop() { requestAnimationFrame(loop); renderer.render(scene, camera); })();
}

function clearArm3D() {
  while (armGroup.children.length) armGroup.remove(armGroup.children[0]);
  linkMeshes.length = 0; jointMeshes.length = 0; frameArrows.length = 0;
}

function buildArm3D() {
  clearArm3D();
  const n = stateDH.jointCount;
  for (let i = 0; i < n; i++) {
    const jm = new THREE.Mesh(new THREE.CylinderGeometry(6, 6, 18, 20),
      new THREE.MeshStandardMaterial({ color: 0xb0b8c4, metalness: 0.2, roughness: 0.6, transparent: true, opacity: 0.85 }));
    armGroup.add(jm); jointMeshes.push(jm);

    const lm = new THREE.Mesh(new THREE.CylinderGeometry(4, 5, 1, 14),
      new THREE.MeshStandardMaterial({ color: THREE_COLORS[i], metalness: 0.15, roughness: 0.7 }));
    armGroup.add(lm); linkMeshes.push(lm);

    const fg = new THREE.Group();
    fg.add(new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(), 25, 0xff0000, 6, 4));
    fg.add(new THREE.ArrowHelper(new THREE.Vector3(0,1,0), new THREE.Vector3(), 25, 0x00ff00, 6, 4));
    fg.add(new THREE.ArrowHelper(new THREE.Vector3(0,0,1), new THREE.Vector3(), 25, 0x0000ff, 6, 4));
    armGroup.add(fg); frameArrows.push(fg);
  }
  // End-effector
  const ee = new THREE.Mesh(new THREE.SphereGeometry(9, 20, 20),
    new THREE.MeshStandardMaterial({ color: 0x2563eb, metalness: 0.3, roughness: 0.5 }));
  armGroup.add(ee); jointMeshes.push(ee);

  const ef = new THREE.Group();
  ef.add(new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(), 35, 0xff0000, 8, 5));
  ef.add(new THREE.ArrowHelper(new THREE.Vector3(0,1,0), new THREE.Vector3(), 35, 0x00ff00, 8, 5));
  ef.add(new THREE.ArrowHelper(new THREE.Vector3(0,0,1), new THREE.Vector3(), 35, 0x0000ff, 8, 5));
  armGroup.add(ef); frameArrows.push(ef);
}

function buildIK3DTrajectory(startDh, target) {
  if (!$('showTrajectory3D') || !$('showTrajectory3D').checked) return [];
  const workingDh = cloneDhParams(startDh);
  const start = {
    x: numVal('trajStartX3D'),
    y: numVal('trajStartY3D'),
    z: numVal('trajStartZ3D'),
  };
  const steps = clamp(numVal('trajSteps3DIn'), 8, 48);
  const path = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const waypoint = {
      x: start.x + (target.x - start.x) * t,
      y: start.y + (target.y - start.y) * t,
      z: start.z + (target.z - start.z) * t,
    };
    solveIK3DOnParams(workingDh, waypoint);
    const end = fkDH(workingDh);
    path.push({ x: end.x, y: end.y, z: end.z });
  }
  return path;
}

function renderIK3DTrajectory(path) {
  if (!trajectory3DGroup) return;
  while (trajectory3DGroup.children.length) trajectory3DGroup.remove(trajectory3DGroup.children[0]);
  trajectory3DGroup.visible = path && path.length > 1;
  if (!trajectory3DGroup.visible) return;

  path.forEach((p, i) => {
    const isEnd = i === path.length - 1;
    const isStart = i === 0;
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(isStart || isEnd ? 5 : 3, 12, 12),
      new THREE.MeshBasicMaterial({ color: isStart ? 0xffffff : 0x0d9488 })
    );
    dot.position.set(p.x, p.z, -p.y);
    trajectory3DGroup.add(dot);
  });
}

function updateArm3D() {
  if (renderer) {
    const c = $('threeContainer');
    const w = c.clientWidth || 700;
    const h = c.clientHeight || 560;
    if (renderer.domElement.width !== Math.round(w * window.devicePixelRatio) ||
        renderer.domElement.height !== Math.round(h * window.devicePixelRatio)) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
  }
  const res = fkDH(stateDH.dh);
  const joints = res.joints;
  const matrices = res.matrices;
  const n = stateDH.jointCount;

  for (let i = 0; i < n; i++) {
    jointMeshes[i].position.set(joints[i].x, joints[i].z, -joints[i].y);

    const zi = i === 0
      ? [0, 0, 1]
      : [matrices[i-1][0][2], matrices[i-1][1][2], matrices[i-1][2][2]];
    const zAxis = new THREE.Vector3(zi[0], zi[2], -zi[1]);
    if (zAxis.length() > 0.001) {
      zAxis.normalize();
      jointMeshes[i].quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), zAxis);
    }

    const s = joints[i], e = joints[i+1];
    const sx = s.x, sy = s.z, sz = -s.y;
    const ex = e.x, ey = e.z, ez = -e.y;
    const mx = (sx+ex)/2, my = (sy+ey)/2, mz = (sz+ez)/2;
    linkMeshes[i].position.set(mx, my, mz);

    const dx = ex-sx, dy = ey-sy, dz = ez-sz;
    const len = Math.hypot(dx, dy, dz);
    if (len > 0.01) {
      linkMeshes[i].scale.set(1, len, 1);
      const dir = new THREE.Vector3(dx, dy, dz).normalize();
      const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0), dir);
      linkMeshes[i].quaternion.copy(quat);
    }

    if (frameArrows[i]) {
      frameArrows[i].position.set(sx, sy, sz);
    }
  }

  const ee = joints[n];
  jointMeshes[n].position.set(ee.x, ee.z, -ee.y);
  if (frameArrows[n]) frameArrows[n].position.set(ee.x, ee.z, -ee.y);

  return res;
}

// ══════════════════════════════════════
//  Visibility & Update
// ══════════════════════════════════════
let threeInit = false;

function updateVisibility() {
  const m = mode();
  const isDH = m === 'DH';
  const isIK3D = m === 'IK3D';
  const is3D = isDH || isIK3D;
  const isIk = m === 'IK';
  const isPositionIK = isIk || isIK3D;
  const count = linkCount();
  const isMulti = count > 2;

  document.body.classList.toggle('is-2d-mode', !is3D);
  document.body.classList.toggle('is-3d-mode', is3D);
  $('controlsStd').style.display = is3D ? 'none' : '';
  $('controlsDH').style.display = is3D ? '' : 'none';
  $('ik3DTargetPanel').style.display = isIK3D ? '' : 'none';
  $('ik3DTrajectoryPanel').style.display = isIK3D ? '' : 'none';
  $('dhJointAnglesPanel').style.display = isDH ? '' : 'none';
  $('dhTransferPanel').style.display = isDH ? '' : 'none';
  $('panel2D').style.display = is3D ? 'none' : '';
  $('panel3D').style.display = is3D ? '' : 'none';
  $('endXMetric').style.display = isPositionIK ? 'none' : '';
  $('endYMetric').style.display = isPositionIK ? 'none' : '';
  $('endZMetric').style.display = is3D && !isIK3D ? '' : 'none';
  $('reachMetricCard').style.display = isPositionIK ? 'none' : '';
  $('errorMetricLabel').textContent = isPositionIK ? 'Position Error' : 'Error';
  $('jointOutputCard').style.display = isPositionIK ? '' : 'none';
  $('dhMatrixCard').style.display = is3D ? '' : 'none';
  $('dhSingularityStatusCard').style.display = is3D ? '' : 'none';
  $('dhJacobianCard').style.display = is3D ? '' : 'none';
  $('planarMatrixCard').style.display = is3D ? 'none' : '';
  $('singularityMetric').style.display = is3D || isIk ? 'none' : '';
  $('singularityCard').style.display = is3D || isIk ? 'none' : '';
  $('workspaceMetricCard').style.display = is3D ? 'none' : '';
  $('collisionMetricCard').style.display = is3D ? 'none' : '';

  $('ikPanel').classList.toggle('hidden', !isIk);
  $('trajectoryPanel').classList.toggle('hidden', !isIk);
  $('obstaclePanel').classList.toggle('hidden', is3D);
  $('fkPanel').classList.toggle('hidden', isIk || is3D);
  for (let i = 3; i <= STD_JOINT_MAX; i++) {
    $(`L${i}Field`).classList.toggle('hidden', count < i);
    $(`theta${i}Field`).classList.toggle('hidden', count < i);
  }
  $('elbowPanel').classList.toggle('hidden', !isIk || isMulti);
  $('ccdHint').classList.toggle('hidden', !isIk || !isMulti);

  if (is3D) {
    $('dhModeLabel').textContent = isIK3D ? '3D Position Inverse Kinematics' : 'DH Forward Kinematics';
    $('dhSolverLabel').textContent = isIK3D
      ? `${stateDH.jointCount}-joint Damped Least Squares — Drag to rotate, scroll to zoom`
      : `${stateDH.jointCount}-joint DH chain — Drag to rotate, scroll to zoom`;
    $('solverNote').textContent = isIK3D
      ? 'Damped Least Squares position IK (orientation unconstrained)'
      : 'Forward kinematics using DH convention (3D)';
  } else {
    $('modeLabel').textContent = isIk ? 'Inverse Kinematics' : 'Forward Kinematics';
    $('solverLabel').textContent = isIk ? (isMulti ? `CCD ${count}-link` : 'Analytic 2-link') : `${count}-link joint angle solve`;
    $('solverNote').textContent = isIk ? (isMulti ? `CCD inverse kinematics (${count} links)` : 'Analytic inverse kinematics') : `Forward kinematics (${count} links)`;
  }
  updateLearningGuide();
}

function update() {
  updateVisibility();
  const m = mode();

  if (m === 'DH' || m === 'IK3D') {
    if (!threeInit) { init3D(); threeInit = true; buildArm3D(); }
    let ikResult = null;
    if (m === 'IK3D') {
      const target = { x:numVal('xTarget3D'), y:numVal('yTarget3D'), z:numVal('zTarget3D') };
      const trajectoryStart = cloneDhParams(stateDH.dh);
      ikResult = solveIK3D(target);
      syncDHTable();
      targetMarker.visible = true;
      targetMarker.position.set(target.x, target.z, -target.y);
      renderIK3DTrajectory(buildIK3DTrajectory(trajectoryStart, target));
    } else {
      if (targetMarker) targetMarker.visible = false;
      renderIK3DTrajectory([]);
    }
    const res = updateArm3D();
    const n = stateDH.jointCount;

    for (let i = 0; i < n; i++) {
      const el = $(`thOut${i}`);
      if (el) el.textContent = `${stateDH.dh[i].theta.toFixed(2)} deg`;
    }

    $('endX').textContent = res.x.toFixed(2);
    $('endY').textContent = res.y.toFixed(2);
    $('endZ').textContent = res.z.toFixed(2);
    $('errorMetric').textContent = ikResult ? ikResult.error.toFixed(2) : '0.0';
    if ($('jointPositionError')) $('jointPositionError').textContent = ikResult ? ikResult.error.toFixed(2) : '0.00';
    const maxR = stateDH.dh.reduce((s,d) => s + Math.abs(d.a) + Math.abs(d.d), 0);
    $('reachMetric').textContent = `0-${maxR.toFixed(0)}`;

    const T = res.matrices[res.matrices.length - 1];
    const f = v => v.toFixed(2).padStart(7);
    $('dhMatrix').textContent =
      `${f(T[0][0])} ${f(T[0][1])} ${f(T[0][2])}  ${f(T[0][3])}\n` +
      `${f(T[1][0])} ${f(T[1][1])} ${f(T[1][2])}  ${f(T[1][3])}\n` +
      `${f(T[2][0])} ${f(T[2][1])} ${f(T[2][2])}  ${f(T[2][3])}\n` +
      `${f(T[3][0])} ${f(T[3][1])} ${f(T[3][2])}  ${f(T[3][3])}`;
    const jac = dhJacobianMetrics(stateDH.dh, res);
    const jf = v => Math.abs(v) < 0.0005 ? '0.000' : v.toFixed(3);
    const rowLabels = ['Vx', 'Vy', 'Vz', 'ωx', 'ωy', 'ωz'];
    $('dhJacobian').innerHTML =
      `<table class="jacobian-table"><thead><tr><th>Motion</th>${
        Array.from({ length: n }, (_, i) => `<th>J${i + 1}</th>`).join('')
      }</tr></thead><tbody>${jac.J.map((row, r) =>
        `<tr><th>${rowLabels[r]}</th>${row.map(v => `<td>${jf(v)}</td>`).join('')}</tr>`
      ).join('')}</tbody></table>`;
    const labels = { stable: 'Stable', near: 'Near Singularity', singular: 'Singular' };
    $('dhSingularityBadge').textContent = labels[jac.level];
    $('dhSingularityStatusCard').className = `mini-card singularity-card ${jac.level}`;
    $('dhRank').textContent = `${jac.rank}/${jac.expectedRank}`;
    $('dhSigmaMin').textContent = jac.sigmaMin.toFixed(4);
    $('dhConditionNumber').textContent = Number.isFinite(jac.condition) ? jac.condition.toFixed(1) : '∞';
    $('dhSingularityNote').textContent = jac.level === 'stable'
      ? 'The normalized geometric Jacobian has full column rank.'
      : jac.level === 'near'
        ? 'The chain is close to losing an instantaneous motion direction.'
        : 'The chain has lost an instantaneous motion direction at this configuration.';
    if (ikResult && !ikResult.converged) setStatus('Approximate 3D IK solution', 'warning');
    else setStatus('Ready');
    return;
  }

  // 2D IK / FK
  drawGrid();
  const lv = links();
  let th = [], tgt = null, ok = true, trajectoryPath = null;
  const maxReach = lv.reduce((sum, value) => sum + value, 0);
  const fitScale = Math.min(1, (Math.min(canvas.width, canvas.height) / 2 - 42) / Math.max(maxReach, 1));
  state.viewScale = fitScale * state.viewZoom;
  drawReachZone(lv);
  const obstacle = drawObstacle();

  if (m === 'IK') {
    tgt = { x: numVal('xTarget'), y: numVal('yTarget') };
    const r = Math.hypot(tgt.x, tgt.y);
    const mx = lv.reduce((s,v)=>s+v,0);
    const mn = Math.max(0, Math.max(...lv) - (mx - Math.max(...lv)));
    ok = r <= mx && r >= mn;
    trajectoryPath = drawTrajectory(lv, tgt);
    th = solve2DForTarget(lv, tgt.x, tgt.y, state.lastThetas);
  } else {
    th = Array.from({ length: lv.length }, (_, i) => deg2rad(numVal(`theta${i + 1}In`)));
  }

  state.lastThetas = th.slice();
  const pts = fk(lv, th).joints;
  if (tgt) drawTarget(tgt.x, tgt.y, ok);
  const end = drawArm2D(pts, lv.length);
  const collision = collisionStatus(pts, obstacle, trajectoryPath);

  const err = tgt ? Math.hypot(end.x - tgt.x, end.y - tgt.y) : 0;
  $('endX').textContent = end.x.toFixed(1);
  $('endY').textContent = end.y.toFixed(1);
  $('errorMetric').textContent = err.toFixed(2);
  if ($('jointPositionError')) $('jointPositionError').textContent = err.toFixed(2);
  const mx = lv.reduce((s,v)=>s+v,0);
  const mn = Math.max(0, Math.max(...lv) - (mx - Math.max(...lv)));
  $('reachMetric').textContent = `${mn.toFixed(0)}-${mx.toFixed(0)}`;
  const radialUse = Math.hypot(end.x, end.y) / Math.max(mx, 1);
  $('workspaceMetric').textContent = `${clamp(radialUse * 100, 0, 100).toFixed(0)}%`;
  $('collisionMetric').textContent = collision.label;
  $('collisionMetricCard').className = `metric ${collision.level}`;
  updatePlanarMatrix(th, end);

  for (let i = 0; i < lv.length; i++) {
    const el = $(`thOut${i}`);
    if (el) el.textContent = `${rad2deg(th[i]||0).toFixed(2)} deg`;
  }

  const singularity = updateSingularity(lv, th);
  if (tgt && !ok) setStatus('Target outside reach', 'error');
  else if (collision.level === 'crossing') setStatus('Obstacle crossing', 'error');
  else if (collision.level === 'near') setStatus('Obstacle near link', 'warning');
  else if (tgt && err > 1) setStatus('Approximate solution', 'warning');
  else if (singularity.level === 'singular') setStatus('Singular configuration', 'error');
  else if (singularity.level === 'near') setStatus('Near singularity', 'warning');
  else setStatus('Ready');
}

// ══════════════════════════════════════
//  Report Export
// ══════════════════════════════════════
function fmt(v, digits = 3) {
  return Number.isFinite(v) ? Number(v).toFixed(digits) : 'Infinity';
}
function fmt2(v) {
  return fmt(Number(v), 2);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

function downloadFile(name, type, content) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function captureViewImage() {
  if ((mode() === 'DH' || mode() === 'IK3D') && renderer) {
    renderer.render(scene, camera);
    return renderer.domElement.toDataURL('image/png');
  }
  return canvas.toDataURL('image/png');
}

function current2DReport() {
  const m = mode();
  const lv = links();
  const target = m === 'IK' ? { x: numVal('xTarget'), y: numVal('yTarget') } : null;
  const anglesRad = m === 'IK'
    ? state.lastThetas.slice(0, lv.length)
    : Array.from({ length: lv.length }, (_, i) => deg2rad(numVal(`theta${i + 1}In`)));
  const fkResult = fk(lv, anglesRad);
  const end = fkResult.joints[fkResult.joints.length - 1];
  const jac = jacobianMetrics(lv, anglesRad);
  const maxReach = lv.reduce((sum, value) => sum + value, 0);
  const minReach = Math.max(0, Math.max(...lv) - (maxReach - Math.max(...lv)));
  const obstacle = $('showObstacle').checked
    ? { x: numVal('obstacleX'), y: numVal('obstacleY'), radius: clamp(numVal('obstacleR'), 10, 120) }
    : null;
  const collision = obstacle
    ? collisionStatus(fkResult.joints, { x: obstacle.x, y: obstacle.y, r: obstacle.radius })
    : { level: 'clear', label: 'Off', distance: Infinity };
  return {
    family: '2D planar arm',
    mode: m === 'IK' ? 'IK 2D' : 'FK 2D',
    solver: $('solverNote').textContent,
    linkLengths: lv,
    jointAnglesDeg: anglesRad.map(rad2deg),
    target,
    endEffector: { x: end.x, y: end.y },
    positionError: target ? Math.hypot(end.x - target.x, end.y - target.y) : 0,
    reachRange: { min: minReach, max: maxReach },
    workspaceUsePercent: clamp(Math.hypot(end.x, end.y) / Math.max(maxReach, 1) * 100, 0, 100),
    trajectory: m === 'IK' && $('showTrajectory').checked
      ? {
          start: { x: numVal('trajStartX'), y: numVal('trajStartY') },
          target,
          steps: clamp(numVal('trajStepsIn'), 8, 48),
        }
      : null,
    obstacle,
    collision: {
      level: collision.level,
      clearance: Number.isFinite(collision.distance) ? collision.distance : null,
    },
    jacobianCondition: {
      level: jac.level,
      sigmaMin: jac.sigmaMin,
      sigmaMax: jac.sigmaMax,
      condition: jac.condition,
    },
  };
}

function current3DReport() {
  const m = mode();
  const target = m === 'IK3D'
    ? { x: numVal('xTarget3D'), y: numVal('yTarget3D'), z: numVal('zTarget3D') }
    : null;
  const fkResult = fkDH(stateDH.dh);
  const jac = dhJacobianMetrics(stateDH.dh, fkResult);
  return {
    family: 'Standard DH spatial arm',
    mode: m === 'IK3D' ? 'IK 3D' : 'DH',
    solver: $('solverNote').textContent,
    jointCount: stateDH.jointCount,
    dhParameters: stateDH.dh.map((row, i) => ({ joint: i + 1, ...row })),
    target,
    endEffector: { x: fkResult.x, y: fkResult.y, z: fkResult.z },
    positionError: target ? Math.hypot(fkResult.x - target.x, fkResult.y - target.y, fkResult.z - target.z) : 0,
    transform: fkResult.matrices[fkResult.matrices.length - 1],
    jacobian: jac.J,
    jacobianCondition: {
      level: jac.level,
      rank: jac.rank,
      expectedRank: jac.expectedRank,
      sigmaMin: jac.sigmaMin,
      condition: jac.condition,
    },
  };
}

function buildReportData() {
  update();
  const data = mode() === 'DH' || mode() === 'IK3D' ? current3DReport() : current2DReport();
  return {
    title: 'Robot Arm Simulator Report',
    generatedAt: new Date().toISOString(),
    status: $('statusText').textContent,
    appVersion: 'v54-3d-trajectory-start-point',
    viewImage: captureViewImage(),
    ...data,
  };
}

function renderMatrixRows(matrix) {
  if (!matrix) return '';
  return matrix.map(row => `<tr>${row.map(v => `<td>${fmt(v, 4)}</td>`).join('')}</tr>`).join('');
}

function renderReportHtml(report) {
  const is3D = report.family.includes('spatial');
  const angles = report.jointAnglesDeg || report.dhParameters.map(row => row.theta);
  const target = report.target
    ? Object.entries(report.target).map(([k, v]) => `${k.toUpperCase()}: ${fmt(v, 2)}`).join(', ')
    : 'None';
  const end = Object.entries(report.endEffector).map(([k, v]) => `${k.toUpperCase()}: ${fmt(v, 2)}`).join(', ');
  const dhRows = report.dhParameters ? report.dhParameters.map(row =>
    `<tr><td>${row.joint}</td><td>${fmt(row.theta, 2)}</td><td>${fmt(row.d, 2)}</td><td>${fmt(row.a, 2)}</td><td>${fmt(row.alpha, 2)}</td></tr>`
  ).join('') : '';

  return `<!doctype html><html><head><meta charset="utf-8"><title>${escapeHtml(report.title)}</title>
    <style>
      body{font-family:Inter,Arial,sans-serif;margin:32px;color:#18202b;line-height:1.45}
      h1{margin:0 0 6px;font-size:28px} h2{margin:24px 0 10px;font-size:16px}
      .meta{color:#667080;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
      .box{border:1px solid #dbe3ec;border-radius:8px;padding:12px;background:#fbfdff}
      .label{display:block;color:#667080;font-size:12px;font-weight:700;text-transform:uppercase}
      table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}th,td{border:1px solid #dbe3ec;padding:7px;text-align:right}
      th{background:#eef3f8;color:#2563eb}.left{text-align:left}@media print{body{margin:18mm}.no-print{display:none}}
    </style></head><body>
      <button class="no-print" onclick="window.print()">Print / Save PDF</button>
      <h1>${escapeHtml(report.title)}</h1>
      <div class="meta">${escapeHtml(report.generatedAt)} · ${escapeHtml(report.appVersion)}</div>
      <div class="grid">
        <div class="box"><span class="label">Mode</span>${escapeHtml(report.mode)}</div>
        <div class="box"><span class="label">Solver</span>${escapeHtml(report.solver)}</div>
        <div class="box"><span class="label">Status</span>${escapeHtml(report.status)}</div>
        <div class="box"><span class="label">Target</span>${escapeHtml(target)}</div>
        <div class="box"><span class="label">End Effector</span>${escapeHtml(end)}</div>
        <div class="box"><span class="label">Position Error</span>${fmt(report.positionError, 3)}</div>
      </div>
      <h2>Simulator View</h2>
      <div class="box"><img src="${report.viewImage}" alt="Robot simulator view" style="width:100%;max-height:520px;object-fit:contain"></div>
      <h2>Joint Angles</h2>
      <table><thead><tr>${angles.map((_, i) => `<th>J${i + 1}</th>`).join('')}</tr></thead><tbody><tr>${angles.map(v => `<td>${fmt(v, 2)} deg</td>`).join('')}</tr></tbody></table>
      ${is3D ? `<h2>DH Parameters</h2><table><thead><tr><th>Joint</th><th>Theta</th><th>d</th><th>a</th><th>Alpha</th></tr></thead><tbody>${dhRows}</tbody></table>` : `<h2>Link Lengths</h2><table><tbody><tr>${report.linkLengths.map((v, i) => `<th>L${i + 1}</th>`).join('')}</tr><tr>${report.linkLengths.map(v => `<td>${fmt(v, 1)}</td>`).join('')}</tr></tbody></table>`}
      ${is3D ? `<h2>End-Effector Transform</h2><table><tbody>${renderMatrixRows(report.transform)}</tbody></table>` : ''}
      <h2>Jacobian Condition</h2>
      <table><tbody>${Object.entries(report.jacobianCondition).map(([k, v]) => `<tr><th class="left">${escapeHtml(k)}</th><td>${escapeHtml(Number.isFinite(v) ? fmt(v, 4) : v)}</td></tr>`).join('')}</tbody></table>
    </body></html>`;
}

function exportJsonReport() {
  const report = buildReportData();
  const stamp = report.generatedAt.replace(/[:.]/g, '-');
  downloadFile(`robot-kinematics-report-${stamp}.json`, 'application/json', JSON.stringify(report, null, 2));
}

function exportImageReport() {
  update();
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const a = document.createElement('a');
  a.href = captureViewImage();
  a.download = `robot-kinematics-view-${stamp}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function printReport() {
  const report = buildReportData();
  const win = window.open('', '_blank');
  if (!win) {
    const stamp = report.generatedAt.replace(/[:.]/g, '-');
    downloadFile(`robot-kinematics-report-${stamp}.html`, 'text/html', renderReportHtml(report));
    return;
  }
  win.document.write(renderReportHtml(report));
  win.document.close();
  const printWhenReady = () => {
    win.focus();
    setTimeout(() => win.print(), 120);
  };
  const reportImage = win.document.querySelector('img');
  if (reportImage && reportImage.decode) {
    reportImage.decode().then(printWhenReady).catch(printWhenReady);
  } else if (reportImage && !reportImage.complete) {
    reportImage.addEventListener('load', printWhenReady, { once: true });
    reportImage.addEventListener('error', printWhenReady, { once: true });
  } else {
    printWhenReady();
  }
}

// ══════════════════════════════════════
//  Events
// ══════════════════════════════════════
function canvasToWorld(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) * (canvas.width / r.width) - cx) / state.viewScale,
    y: (cy - (e.clientY - r.top) * (canvas.height / r.height)) / state.viewScale,
  };
}

function rememberCanvasPointer(e) {
  state.canvasPointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
}

function canvasPinchDistance() {
  const pts = [...state.canvasPointers.values()];
  if (pts.length < 2) return 0;
  return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
}

function endCanvasPointer(e) {
  state.canvasPointers.delete(e.pointerId);
  state.draggingTarget = false;
  if (state.canvasPointers.size < 2) state.pinchStartDistance = 0;
}

function init() {
  buildJointOutput(linkCount());

  [...document.getElementsByName('mode')].forEach(el => el.addEventListener('change', () => {
    const m = mode();
    if (m === 'DH' || m === 'IK3D') { buildDHTable(); buildDHSliders(); buildJointOutput(stateDH.jointCount); }
    else buildJointOutput(linkCount());
    update();
  }));

  [...document.getElementsByName('linkCount')].forEach(el => el.addEventListener('change', () => {
    if (mode() !== 'DH' && mode() !== 'IK3D') { buildJointOutput(linkCount()); update(); }
  }));

  ['elbow'].forEach(n => [...document.getElementsByName(n)].forEach(el => el.addEventListener('change', update)));
  [
    ...Array.from({ length: STD_JOINT_MAX }, (_, i) => `L${i + 1}`),
    'xTarget', 'yTarget', 'showReach', 'showWorkspaceFill',
    'showTrajectory', 'trajStartX', 'trajStartY', 'trajStepsRange', 'trajStepsIn',
    'showObstacle', 'obstacleX', 'obstacleY', 'obstacleR',
  ].forEach(id => $(id).addEventListener('input', event => {
    if (id === 'trajStepsRange') $('trajStepsIn').value = event.target.value;
    if (id === 'trajStepsIn') $('trajStepsRange').value = clamp(Number(event.target.value) || 8, 8, 48);
    update();
  }));
  ['xTarget3D', 'yTarget3D', 'zTarget3D', 'showTrajectory3D', 'trajStartX3D', 'trajStartY3D', 'trajStartZ3D', 'trajSteps3DRange', 'trajSteps3DIn'].forEach(id => {
    $(id).addEventListener('input', event => {
      if (id === 'trajSteps3DRange') $('trajSteps3DIn').value = event.target.value;
      if (id === 'trajSteps3DIn') $('trajSteps3DRange').value = clamp(Number(event.target.value) || 8, 8, 48);
      update();
    });
  });
  Array.from({ length: STD_JOINT_MAX }, (_, i) => `theta${i + 1}`).forEach(base => {
    $(`${base}Range`).addEventListener('input', e => { $(`${base}In`).value = e.target.value; update(); });
    $(`${base}In`).addEventListener('input', e => { $(`${base}Range`).value = e.target.value; update(); });
  });

  document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => {
    const n = b.dataset.preset;
    if (n === 'inspect') { document.querySelector('input[name="mode"][value="IK"]').checked = true; $('xTarget').value=150; $('yTarget').value=80; }
    if (n === 'reach') {
      document.querySelector('input[name="mode"][value="FK"]').checked = true;
      [10, 18, 6, -8, 12].forEach((value, i) => syncSlider(`theta${i + 1}`, value));
    }
    if (n === 'fold') {
      document.querySelector('input[name="mode"][value="FK"]').checked = true;
      [120, -135, 85, -70, 45].forEach((value, i) => syncSlider(`theta${i + 1}`, value));
    }
    buildJointOutput(linkCount()); update();
  }));

  $('resetBtn').addEventListener('click', () => {
    DEFAULT_LINKS.forEach((value, i) => { $(`L${i + 1}`).value = value; });
    $('xTarget').value=100; $('yTarget').value=30;
    DEFAULT_ANGLES.forEach((value, i) => syncSlider(`theta${i + 1}`, value));
    document.querySelector('input[name="mode"][value="IK"]').checked = true;
    document.querySelector('input[name="linkCount"][value="2"]').checked = true;
    document.querySelector('input[name="elbow"][value="down"]').checked = true;
    state.viewZoom = 1;
    state.lastThetas = [0,0,0,0,0]; buildJointOutput(2); update();
  });

  $('centerTargetBtn').addEventListener('click', () => { $('xTarget').value=0; $('yTarget').value=0; update(); });
  // Canvas drag for IK target
  canvas.addEventListener('pointerdown', e => {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    rememberCanvasPointer(e);
    if (state.canvasPointers.size === 2) {
      state.draggingTarget = false;
      state.pinchStartDistance = canvasPinchDistance();
      state.pinchStartZoom = state.viewZoom;
      return;
    }
    if (mode() !== 'IK') return;
    state.draggingTarget = true;
    const p = canvasToWorld(e); $('xTarget').value=p.x.toFixed(0); $('yTarget').value=p.y.toFixed(0); update();
  });
  canvas.addEventListener('pointermove', e => {
    e.preventDefault();
    if (!state.canvasPointers.has(e.pointerId)) return;
    rememberCanvasPointer(e);
    if (state.canvasPointers.size >= 2 && state.pinchStartDistance > 0) {
      state.draggingTarget = false;
      state.viewZoom = clamp(state.pinchStartZoom * (canvasPinchDistance() / state.pinchStartDistance), 0.55, 2.8);
      update();
      return;
    }
    if (!state.draggingTarget || mode() !== 'IK') return;
    const p = canvasToWorld(e); $('xTarget').value=p.x.toFixed(0); $('yTarget').value=p.y.toFixed(0); update();
  });
  canvas.addEventListener('pointerup', endCanvasPointer);
  canvas.addEventListener('pointercancel', endCanvasPointer);

  // DH controls
  $('jointMinus').addEventListener('click', () => {
    if (stateDH.jointCount > 2) {
      stateDH.jointCount--; stateDH.dh.pop();
      $('jointCountVal').textContent = stateDH.jointCount;
      buildDHTable(); buildDHSliders(); buildJointOutput(stateDH.jointCount);
      if (threeInit) buildArm3D();
      update();
    }
  });
  $('jointPlus').addEventListener('click', () => {
    if (stateDH.jointCount < 7) {
      stateDH.jointCount++;
      stateDH.dh.push({ ...DH_DEFAULTS[stateDH.jointCount - 1] });
      $('jointCountVal').textContent = stateDH.jointCount;
      buildDHTable(); buildDHSliders(); buildJointOutput(stateDH.jointCount);
      if (threeInit) buildArm3D();
      update();
    }
  });

  $('dhResetBtn').addEventListener('click', () => {
    stateDH.jointCount = 3;
    stateDH.dh = DH_DEFAULTS.slice(0,3).map(d => ({...d}));
    $('jointCountVal').textContent = '3';
    buildDHTable(); buildDHSliders(); buildJointOutput(3);
    if (threeInit) buildArm3D();
    update();
  });

  $('carryDhToIkBtn').addEventListener('click', () => {
    const res = fkDH(stateDH.dh);
    $('xTarget3D').value = res.x.toFixed(2);
    $('yTarget3D').value = res.y.toFixed(2);
    $('zTarget3D').value = res.z.toFixed(2);
    document.querySelector('input[name="mode"][value="IK3D"]').checked = true;
    buildDHTable(); buildDHSliders(); buildJointOutput(stateDH.jointCount);
    update();
  });

  document.querySelectorAll('[data-dh-preset]').forEach(b => b.addEventListener('click', () => {
    const n = b.dataset.dhPreset;
    if (n === 'home') stateDH.dh.forEach(d => d.theta = 0);
    if (n === 'reach') stateDH.dh.forEach((d,i) => d.theta = [30,-15,10,0,25,-10][i]||0);
    if (n === 'fold') stateDH.dh.forEach((d,i) => d.theta = [90,-120,80,-30,45,0][i]||0);
    syncDHTable(); buildDHSliders(); update();
  }));

  document.querySelectorAll('[data-export="png"]').forEach(btn => btn.addEventListener('click', exportImageReport));
  document.querySelectorAll('[data-export="json"]').forEach(btn => btn.addEventListener('click', exportJsonReport));
  document.querySelectorAll('[data-export="pdf"]').forEach(btn => btn.addEventListener('click', printReport));
  const downloadMenus = Array.from(document.querySelectorAll('.download-menu'));
  const closeInactiveDownloadMenus = event => {
    const target = event.target;
    const activeMenu = target instanceof Element ? target.closest('.download-menu') : null;
    downloadMenus.forEach(menu => {
      if (menu !== activeMenu) menu.open = false;
    });
  };
  document.addEventListener('pointerdown', closeInactiveDownloadMenus, true);
  document.addEventListener('click', closeInactiveDownloadMenus);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      downloadMenus.forEach(menu => { menu.open = false; });
    }
  });
  document.querySelectorAll('.download-options button').forEach(btn => btn.addEventListener('click', () => {
    btn.closest('details').open = false;
  }));

  update();
}

init();
