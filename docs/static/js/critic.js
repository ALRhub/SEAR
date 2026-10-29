// Animation: SEAR's causal transformer critic with multi-horizon targets.
(function () {
  const { el, mathText } = window.SVGH;
  const svg = document.getElementById("critic");
  if (!svg) return;

  const N = 5;
  const SPEED = 2;                          // playback speed multiplier
  const yc = (i) => 62 + i * 58;            // row centre; row 0 = state token
  const TOK_X = 170, TOK_W = 92, H = 36;
  const TF_X = 300, TF_W = 200;
  const Q_X = 540, Q_W = 150;
  const G_X = 725, G_W = 190;
  const SIG = { x: 990, y: (yc(1) + yc(N)) / 2, r: 26 };
  const BUF = { x: 75, y: 230 };

  const defs = el("defs", {}, svg);
  const mk = el("marker", { id: "arr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#6b7380" }, mk);

  // ---- replay buffer --------------------------------------------------------
  const buf = el("g", {}, svg);
  el("path", { d: "M15,150 L15,300 A60,16 0 0 0 135,300 L135,150", fill: "#fff", stroke: "#b8bec8", "stroke-width": 1.5 }, buf);
  el("ellipse", { cx: 75, cy: 150, rx: 60, ry: 16, fill: "#f2f4f7", stroke: "#b8bec8", "stroke-width": 1.5 }, buf);
  for (let y = 178; y <= 290; y += 14) el("line", { x1: 30, x2: 120, y1: y, y2: y, stroke: "#dde1e7", "stroke-width": 6, "stroke-linecap": "round" }, buf);
  el("rect", { class: "chunk-hl", x: 26, y: 213, width: 98, height: 30, rx: 6, fill: "none", stroke: "var(--sear)", "stroke-width": 2.5 }, buf);
  el("text", { x: 75, y: 330, "text-anchor": "middle", "font-size": 13, fill: "#6b7380", text: "Replay buffer" }, buf);

  // ---- transformer block ----------------------------------------------------
  const tf = el("g", { class: "tf" }, svg);
  el("rect", { class: "tf-body", x: TF_X, y: 30, width: TF_W, height: 352, rx: 14, fill: "var(--tf-fill)", stroke: "var(--tf-stroke)", "stroke-width": 1.5 }, tf);
  el("text", { x: TF_X + TF_W / 2, y: 20, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: "#b8691b", text: "Causal Transformer" }, tf);

  // attention paths: output row j sees inputs 0..j only
  const att = [];
  for (let j = 1; j <= N; j++) {
    for (let i = 0; i <= j; i++) {
      const p = el("path", {
        class: "att", pathLength: 1, fill: "none", stroke: "#d9822b", "stroke-width": 1.3,
        d: `M${TF_X},${yc(i)} C${TF_X + 100},${yc(i)} ${TF_X + 100},${yc(j)} ${TF_X + TF_W},${yc(j)}`,
        "data-tgt": j, "data-src": i,
        style: { "--d": `${((j - 1) * 0.55 + i * 0.07) / SPEED}s` },
      }, tf);
      att.push(p);
    }
  }

  // ---- tokens -----------------------------------------------------------------
  const toks = [];
  for (let i = 0; i <= N; i++) {
    const isS = i === 0;
    const g = el("g", {
      class: "tok", "data-src": i,
      style: { transform: `translate(${BUF.x - (TOK_X + TOK_W / 2)}px, ${BUF.y - yc(i)}px)`, "--d": `${(i * 0.12) / SPEED}s` },
    }, svg);
    el("rect", { x: TOK_X, y: yc(i) - H / 2, width: TOK_W, height: H, rx: 8, fill: isS ? "var(--state-fill)" : "var(--action-fill)", stroke: isS ? "var(--state-stroke)" : "var(--action-stroke)", "stroke-width": 1.5 }, g);
    const lbl = isS ? "s_t" : i === 1 ? "a_t" : `a_{t+${i - 1}}`;
    mathText(g, lbl, { x: TOK_X + TOK_W / 2, y: yc(i) + 5, "text-anchor": "middle", "font-weight": 600, fill: "#1b1f24" }, 15);
    el("line", { x1: TOK_X + TOK_W, x2: TF_X, y1: yc(i), y2: yc(i), stroke: isS ? "var(--state-stroke)" : "var(--action-stroke)", "stroke-width": 1.5 }, g);
    toks.push(g);
  }
  el("text", { x: TOK_X + TOK_W / 2, y: 20, "text-anchor": "middle", "font-size": 13, fill: "#6b7380", text: "chunk tokens" }, svg);

  // ---- column headers (in the empty state row) -------------------------------
  el("text", { class: "pop q", x: Q_X + Q_W / 2, y: 60, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#a8323b", text: "one Q-value per prefix" }, svg);
  const gh = el("g", { class: "pop g" }, svg);
  el("text", { x: G_X + G_W / 2, y: 54, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#4a5260", text: "multi-horizon targets" }, gh);
  el("rect", { x: G_X + 18, y: 66, width: 11, height: 11, rx: 2, fill: "var(--reward)" }, gh);
  el("text", { x: G_X + 34, y: 76, "font-size": 12, fill: "#6b7380", text: "reward" }, gh);
  el("rect", { x: G_X + 90, y: 65, width: 26, height: 13, rx: 6.5, fill: "#fff", stroke: "var(--action-stroke)" }, gh);
  el("text", { x: G_X + 122, y: 76, "font-size": 12, fill: "#6b7380", text: "target-critic bootstrap" }, gh);

  // ---- outputs, targets, errors ------------------------------------------------
  const rowEls = [];
  for (let j = 1; j <= N; j++) {
    const y = yc(j), d = (j - 1) * 0.18;
    const mh = j < N ? " mh-only" : "";
    el("line", { class: "draw q", pathLength: 1, x1: TF_X + TF_W, x2: Q_X - 3, y1: y, y2: y, stroke: "#d6505a", "stroke-width": 1.5, "marker-end": "url(#arr)", style: { "--d": `${d / SPEED}s` }, "data-row": j }, svg);
    const q = el("g", { class: "pop q", "data-row": j, style: { "--d": `${(d + 0.15) / SPEED}s` } }, svg);
    el("rect", { x: Q_X, y: y - H / 2, width: Q_W, height: H, rx: 8, fill: "var(--q-fill)", stroke: "var(--q-stroke)", "stroke-width": 1.5 }, q);
    mathText(q, `Q^{(${j})}(s_t, a_{t:t+${j}})`, { x: Q_X + Q_W / 2, y: y + 5, "text-anchor": "middle", fill: "#1b1f24" }, 14.5);

    const gw = el("g", { class: "mh-wrap" + mh, "data-row": j }, svg);
    const g = el("g", { class: "pop g", style: { "--d": `${((j - 1) * 0.6) / SPEED}s` } }, gw);
    el("rect", { x: G_X, y: y - H / 2, width: G_W, height: H, rx: 8, fill: "var(--g-fill)", stroke: "var(--g-stroke)", "stroke-width": 1.5 }, g);
    mathText(g, `G^{(${j})}`, { x: G_X + 10, y: y + 5, fill: "#1b1f24", "font-weight": 600 }, 14.5);
    for (let i = 0; i < j; i++) {
      el("rect", { class: "rw", x: G_X + 46 + i * 16, y: y - 6, width: 12, height: 12, rx: 2, fill: "var(--reward)", style: { "--d": `${((j - 1) * 0.6 + 0.2 + i * 0.07) / SPEED}s` } }, gw);
    }
    const bx = G_X + 46 + j * 16 + 2;
    const bs = el("g", { class: "rw", style: { "--d": `${((j - 1) * 0.6 + 0.3 + j * 0.07) / SPEED}s` } }, gw);
    el("rect", { x: bx, y: y - 10, width: 46, height: 20, rx: 10, fill: "#fff", stroke: "var(--action-stroke)" }, bs);
    mathText(bs, `γ^{${j}}Q`, { x: bx + 23, y: y + 5, "text-anchor": "middle", fill: "#1b1f24" }, 13);

    const e = el("g", { class: "mh-wrap" + mh, "data-row": j }, svg);
    el("line", { class: "draw e", pathLength: 1, x1: Q_X + Q_W + 2, x2: G_X - 2, y1: y, y2: y, stroke: "var(--low)", "stroke-width": 3, "stroke-dasharray": "1", style: { "--d": `${d / SPEED}s` } }, e);
    el("path", { class: "draw e", pathLength: 1, fill: "none", stroke: "#6b7380", "stroke-width": 1.4, "marker-end": "url(#arr)", d: `M${G_X + G_W},${y} C${G_X + G_W + 30},${y} ${SIG.x - 50},${SIG.y} ${SIG.x - SIG.r - 2},${SIG.y}`, style: { "--d": `${(d + 0.3) / SPEED}s` } }, e);
    rowEls.push(q, gw, e);
  }

  const sig = el("g", { class: "pop e", style: { "--d": `${0.9 / SPEED}s` } }, svg);
  el("circle", { cx: SIG.x, cy: SIG.y, r: SIG.r, fill: "#fff", stroke: "#6b7380", "stroke-width": 1.5 }, sig);
  el("text", { x: SIG.x, y: SIG.y + 8, "text-anchor": "middle", "font-size": 24, fill: "#1b1f24", text: "Σ" }, sig);
  el("text", { x: SIG.x, y: SIG.y + 46, "text-anchor": "middle", "font-size": 12, fill: "#6b7380", text: "Bellman" }, sig);
  el("text", { x: SIG.x, y: SIG.y + 60, "text-anchor": "middle", "font-size": 12, fill: "#6b7380", text: "errors" }, sig);
  const count = el("text", { x: SIG.x, y: SIG.y + 82, "text-anchor": "middle", "font-size": 12.5, "font-weight": 700, fill: "var(--sear)" }, sig);

  // ---- update loop -------------------------------------------------------------
  const ud = `M${SIG.x},${SIG.y + 92} L${SIG.x},430 L${TF_X + TF_W / 2},430 L${TF_X + TF_W / 2},${382 + 8}`;
  el("path", { class: "draw u", pathLength: 1, d: ud, fill: "none", stroke: "#6b7380", "stroke-width": 1.6, "marker-end": "url(#arr)" }, svg);
  const flow = el("path", { class: "flow", d: ud, fill: "none", stroke: "var(--sear)", "stroke-width": 2.2, "stroke-dasharray": "6 10", opacity: 0 }, svg);
  el("text", { class: "pop u", x: (SIG.x + TF_X) / 2 + 60, y: 448, "text-anchor": "middle", "font-size": 13, fill: "#4a5260", text: "gradient update of the critic parameters φ" }, svg);

  // ---- hover: explain components; for a Q row, show which inputs it sees ------
  const focusable = svg.querySelectorAll("[data-src], [data-row]");
  function hitArea(x, y, w, h, html, onEnter, onLeave) {
    const r = el("rect", { class: "hit", x, y, width: w, height: h, fill: "transparent" }, svg);
    r.addEventListener("mouseenter", () => { hoverHtml = html(); if (onEnter) onEnter(); refreshCaption(); });
    r.addEventListener("mouseleave", () => { hoverHtml = null; if (onLeave) onLeave(); refreshCaption(); });
  }
  hitArea(10, 130, 130, 210, () => captions()[0]);
  hitArea(TOK_X - 5, 36, TOK_W + 10, yc(N) + 24 - 36, () => captions()[1]);
  hitArea(TF_X, 30, TF_W, 352, () => captions()[2]);
  for (let j = 1; j <= N; j++) {
    hitArea(Q_X - 6, yc(j) - 26, G_X + G_W - Q_X + 12, 52, () => hoverCaption(j), () => focus(j), () => focus(0));
  }
  hitArea(SIG.x - 36, SIG.y - 36, 72, 130, () => captions()[5]);
  hitArea(TF_X + TF_W / 2 - 12, 416, SIG.x - TF_X - TF_W / 2 + 24, 40, () => UPDATE);

  function focus(j) {
    focusable.forEach((n) => {
      n.classList.remove("dim", "hot");
      if (!j) return;
      if (n.dataset.tgt !== undefined) n.classList.toggle(+n.dataset.tgt === j ? "hot" : "dim", true);
      else if (n.dataset.src !== undefined) { if (+n.dataset.src > j) n.classList.add("dim"); }
      else if (n.dataset.row !== undefined && +n.dataset.row !== j) n.classList.add("dim");
    });
  }

  // ---- stage machine ----------------------------------------------------------
  const durations = [1800, 2000, 3600, 2200, 3800, 2600, 3400].map((d) => d / SPEED);
  const LAST = durations.length - 1;
  const captions = () => [
    "<b>Sample.</b> A training point is drawn from the replay buffer: a state $s_t$ and the $N$ actions $a_{t:t+N}$ that were executed after it (here $N=5$), together with the observed rewards. Our implementation only samples chunks that lie within a single episode, so a chunk never crosses an episode boundary.",
    "<b>Tokenize.</b> The state and each action of the chunk become one token of the critic.",
    "<b>Causal attention.</b> The token of action $a_{t+k-1}$ only attends to $s_t$ and earlier actions, so the $k$-th output depends on exactly the prefix $a_{t:t+k}$ and nothing that comes after it.",
    "<b>One Q-value per prefix.</b> A single forward pass predicts $Q^{(1)}(s_t,a_{t:t+1}),\\,\\dots,\\,Q^{(N)}(s_t,a_{t:t+N})$.",
    mh
      ? "<b>Multi-horizon targets.</b> Each prefix gets its own target $G^{(k)}=\\sum_{i<k}\\gamma^i r_{t+i}+\\gamma^k\\hat Q(s_{t+k},a'),\\ a'\\sim\\pi(\\cdot\\mid s_{t+k})$: the observed $k$-step return plus a bootstrapped value. Because the critic is conditioned on the executed actions, no importance weights are needed."
      : "<b>Single target (ablation).</b> Only the full-chunk target $G^{(N)}$ is used, as is common in offline action-chunking RL. The shorter prefixes get no training signal.",
    mh
      ? "<b>Bellman errors.</b> All $N$ squared errors $(Q^{(k)}-G^{(k)})^2$ are averaged, which gives $N$ regression targets per sample instead of one. The number of targets now grows with the chunk size, just like the action dimension."
      : "<b>Bellman error.</b> A single regression target for the whole <span class='nowrap'>$N\\cdot\\dim(a)$-dimensional</span> action chunk.",
    UPDATE,
  ];
  const UPDATE = "<b>Update.</b> Gradients from every prefix flow into the shared transformer weights.";
  const HINT = "<span style='color:var(--ink-soft)'>Hover over any part of the diagram (replay buffer, tokens, transformer, Q-values and targets, Bellman errors, update) to see what it does.</span>";
  const hoverCaption = (k) =>
    `<b>Prefix k = ${k}.</b> $Q^{(${k})}$ sees $s_t$ and $a_{t:t+${k}}$ only. Its target $G^{(${k})}$ sums ${k} observed reward${k > 1 ? "s" : ""} and bootstraps at $s_{t+${k}}$ with the target critic and a fresh action chunk from the policy.`;

  let stage = 0, timer = null, mh = true, playing = false, hoverHtml = null;
  const capEl = document.getElementById("critic-caption");
  const playBtn = document.getElementById("critic-play");
  const mhRadios = document.querySelectorAll("input[name=critic-target]");
  const mhText = document.getElementById("critic-mh-text");

  function renderMath(node, html) {
    node.innerHTML = html;
    if (window.renderMathInElement) {
      renderMathInElement(node, { delimiters: [{ left: "$", right: "$", display: false }], throwOnError: false });
    }
  }
  // hovered component > hint once finished > caption of the current step
  function refreshCaption() {
    renderMath(capEl, hoverHtml || (stage === LAST ? HINT : captions()[stage]));
  }

  function go(s) {
    stage = s;
    for (let i = 0; i < durations.length; i++) svg.classList.toggle("st" + i, i <= s);
    flow.setAttribute("opacity", s >= LAST ? 1 : 0);
    refreshCaption();
  }
  function updateButton() {
    playBtn.textContent = playing ? "❚❚ Pause" : stage === LAST ? "↻ Replay" : "▶ Play";
  }
  // advance through the steps; playback stops on the final one
  function schedule() {
    clearTimeout(timer);
    if (playing && stage === LAST) playing = false;
    updateButton();
    if (!playing) return;
    timer = setTimeout(() => { go(stage + 1); schedule(); }, durations[stage]);
  }
  function pause() { playing = false; schedule(); }

  playBtn.onclick = () => {
    if (playing) return pause();
    playing = true;
    if (stage !== LAST) return schedule();
    // replay: clear the diagram, then build it up again from the start
    clearTimeout(timer);
    for (let i = 0; i < durations.length; i++) svg.classList.remove("st" + i);
    flow.setAttribute("opacity", 0);
    updateButton();
    timer = setTimeout(() => { go(0); schedule(); }, 300);
  };
  document.getElementById("critic-prev").onclick = () => { pause(); go(Math.max(0, stage - 1)); updateButton(); };
  document.getElementById("critic-next").onclick = () => { pause(); go(Math.min(LAST, stage + 1)); updateButton(); };

  const mhNotes = {
    on: "<b>Multi-horizon critic targets (SEAR).</b> Every prefix of the chunk gets its own Bellman target, so each replay sample gives $N$ regression targets instead of one. The number of targets grows with the chunk size, just like the action dimension. This richer signal makes the critic much easier to fit: SEAR-10 reaches <b>over 80%</b> IQM success on Metaworld-Hard.",
    off: "<b>Naive N-step target.</b> This is how action-chunking critics are commonly trained in offline-to-online RL. The critic has to evaluate an <span class='nowrap'>$N\\cdot\\dim(a)$-dimensional</span> action (40 dimensions for chunks of 10 on Metaworld's 4-D actions) but gets only one target per sample. In online RL, this jump in dimensionality cancels out the benefits of action chunking: IQM success on Metaworld-Hard reaches only <b>about 30%</b>, compared with <b>over 80%</b> for SEAR and <b>about 60%</b> for non-chunking baselines.",
  };
  function setMH() {
    mh = document.querySelector("input[name=critic-target]:checked").value === "mh";
    svg.classList.toggle("nomh", !mh);
    count.textContent = mh ? `${N} targets` : "1 target";
    renderMath(mhText, mh ? mhNotes.on : mhNotes.off);
    refreshCaption();
  }
  mhRadios.forEach((r) => (r.onchange = setMH));

  go(0);
  setMH();
  // start once the diagram scrolls into view
  const io = new IntersectionObserver((es) => {
    if (es[0].isIntersecting) { playing = true; schedule(); io.disconnect(); }
  }, { threshold: 0.4 });
  io.observe(svg);
})();
