// Animation: state coverage of the actor under no chunking, uniform chunking
// and SEAR's random replanning. Mirrors Figure 2 (a)-(c) of the paper.
(function () {
  const { el } = window.SVGH;
  const root = document.getElementById("coverage");
  if (!root) return;

  const T = 12, R = 6;
  const X = (t) => 24 + t * 28;
  const Y = (r) => 20 + r * 22;
  const HIST = { base: 228, h: 56 };
  const TICK = 120;                 // ms per time step

  const MODES = [
    { key: "none", title: "No action chunking", color: "var(--nochunk)", verdict: "Actor queried in every state",
      sub: () => "The actor is queried in every state: full coverage, but no chunked exploration or fast value propagation." },
    { key: "uniform", title: "Uniform action chunking", color: "var(--chunk)", verdict: "Gaps in actor coverage",
      sub: (N) => `Each chunk of ${N} actions runs open-loop, so the actor is only queried in every ${N}${N === 2 ? "nd" : N === 3 ? "rd" : "th"} state.` },
    { key: "random", title: "SEAR: random replanning", color: "var(--sear)", verdict: "Uniform actor coverage",
      sub: (N) => `Only a random prefix k ∈ {1,…,${N}} of each chunk is executed before the actor replans.` },
  ];

  let N = 3, playing = true, col = -1, hold = 0, rounds = 0, timer = null;
  let batch = [];

  // ---- build panels -------------------------------------------------------------
  const panels = MODES.map((m) => {
    const div = document.createElement("div");
    div.className = "cov-panel";
    div.innerHTML = `<h4><i style="background:${m.color}"></i>${m.title}</h4><p class="sub"></p>`;
    const svg = el("svg", { viewBox: "0 0 360 236", role: "img", "aria-label": m.title }, div);
    const verdict = document.createElement("p");
    verdict.className = "verdict";
    verdict.textContent = m.verdict;
    div.appendChild(verdict);
    root.appendChild(div);

    const lines = [], nodes = [];
    for (let r = 0; r < R; r++) {
      const lr = [], nr = [];
      for (let t = 0; t < T - 1; t++) lr.push(el("line", { stroke: "#333", "stroke-width": 1.2 }, svg));
      for (let t = 0; t < T; t++) nr.push(el("circle", { r: 6.5, "stroke-width": 1.3, stroke: "#333", fill: "#fff" }, svg));
      lines.push(lr); nodes.push(nr);
    }

    el("text", { x: 4, y: 162, "font-size": 11.5, fill: "#6b7380", text: "actor queries per episode (training)" }, svg);
    el("line", { x1: 10, x2: 350, y1: HIST.base, y2: HIST.base, stroke: "#b8bec8" }, svg);
    el("line", { x1: 10, x2: 350, y1: HIST.base - HIST.h, y2: HIST.base - HIST.h, stroke: "#d5d9df", "stroke-dasharray": "3 3" }, svg);
    el("text", { x: 352, y: HIST.base - HIST.h + 3, "font-size": 9, fill: "#8a93a2", text: "1" }, svg);
    const bars = [];
    for (let t = 0; t < T; t++) bars.push(el("rect", { x: X(t) - 8, width: 16, y: HIST.base, height: 0, rx: 2, fill: m.color }, svg));

    return { m, div, svg, sub: div.querySelector(".sub"), lines, nodes, bars,
      counts: new Array(T).fill(0), den: new Array(T).fill(0) };
  });

  // ---- simulation -----------------------------------------------------------------
  const randint = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
  function queries(mode) {
    const q = new Array(T).fill(false);
    let t = 0;
    while (t < T) {
      q[t] = true;
      t += mode === "none" ? 1 : mode === "uniform" ? N : randint(1, N);
    }
    return q;
  }
  function newBatch() {
    batch = [];
    for (let r = 0; r < R; r++) {
      const ph = Math.random() * 6.28, f = 0.9 + Math.random() * 0.8;
      const off = [];
      for (let t = 0; t < T; t++) off.push(t === 0 ? 0 : 4.5 * Math.sin(ph + t * f) + (Math.random() - 0.5) * 2);
      batch.push({ off, q: MODES.map((m) => queries(m.key)) });
    }
    col = -1; hold = 0;
  }
  function accumulate(t) {
    panels.forEach((p, k) => {
      for (let r = 0; r < R; r++) { p.den[t]++; if (batch[r].q[k][t]) p.counts[t]++; }
    });
  }
  function fastForward(n) {
    for (let e = 0; e < n; e++) {
      panels.forEach((p) => {
        const q = queries(p.m.key);
        for (let t = 0; t < T; t++) { p.den[t]++; if (q[t]) p.counts[t]++; }
      });
    }
    render();
  }
  function reset() {
    panels.forEach((p) => { p.counts.fill(0); p.den.fill(0); });
    newBatch();
    rounds = 0;
    render();
  }

  // ---- rendering ----------------------------------------------------------------
  // queries at step t per episode started so far; columns the current
  // episodes have not reached yet lag behind, so the bars grow mid-episode
  const cov = (p, t) => (p.den[0] ? p.counts[t] / p.den[0] : 0);

  function render() {
    panels.forEach((p, k) => {
      p.sub.textContent = p.m.sub(N);
      // trajectories
      for (let r = 0; r < R; r++) {
        const b = batch[r];
        for (let t = 0; t < T; t++) {
          const n = p.nodes[r][t], seen = t <= col;
          n.setAttribute("cx", X(t));
          n.setAttribute("cy", Y(r) + b.off[t]);
          n.setAttribute("fill", seen && b.q[k][t] ? p.m.color : "#fff");
          n.setAttribute("opacity", seen ? 1 : 0.18);
          if (t < T - 1) {
            const l = p.lines[r][t];
            l.setAttribute("x1", X(t)); l.setAttribute("y1", Y(r) + b.off[t]);
            l.setAttribute("x2", X(t + 1)); l.setAttribute("y2", Y(r) + b.off[t + 1]);
            l.setAttribute("opacity", t + 1 <= col ? 0.9 : 0.15);
          }
        }
      }
      // coverage histogram
      for (let t = 0; t < T; t++) {
        const hh = cov(p, t) * HIST.h;
        p.bars[t].setAttribute("y", HIST.base - hh);
        p.bars[t].setAttribute("height", hh);
      }
    });
  }

  // one round = all rows have run a full episode; pause after N rounds
  function tick() {
    if (col < T - 1) {
      col++;
      accumulate(col);
      if (col === T - 1 && ++rounds >= N) setPlaying(false);
    } else if (++hold > 3) newBatch();
    render();
  }
  function loop() {
    clearInterval(timer);
    if (playing) timer = setInterval(tick, TICK);
  }

  // ---- controls -----------------------------------------------------------------
  const nIn = document.getElementById("cov-n"), nVal = document.getElementById("cov-n-val");
  const play = document.getElementById("cov-play");
  function setPlaying(on) {
    playing = on;
    play.textContent = playing ? "❚❚ Pause" : "▶ Play";
    loop();
  }
  nIn.oninput = () => { N = +nIn.value; nVal.textContent = N; reset(); setPlaying(true); };
  play.onclick = () => {
    if (!playing && rounds >= N) { rounds = 0; newBatch(); }
    setPlaying(!playing);
  };
  document.getElementById("cov-ff").onclick = () => fastForward(100);
  document.getElementById("cov-reset").onclick = () => { reset(); setPlaying(true); };

  nIn.value = N; nVal.textContent = N;
  newBatch();
  render();
  playing = false;
  const io = new IntersectionObserver((es) => {
    if (es[0].isIntersecting) { setPlaying(true); io.disconnect(); }
  }, { threshold: 0.3 });
  io.observe(root);
})();
