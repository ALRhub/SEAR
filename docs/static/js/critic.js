// Animation: SEAR's causal transformer critic with multi-horizon targets.
// Two layouts share the same elements and classes: a landscape one (rows of
// tokens flowing left to right) and a portrait one for phones (columns of
// tokens flowing top to bottom).
(function () {
  const { el, mathText } = window.SVGH;
  const svg = document.getElementById("critic");
  if (!svg) return;

  const N = 5;
  const SPEED = 2;                          // playback speed multiplier
  const portraitMQ = window.matchMedia("(max-width: 640px)");
  const touch = window.matchMedia("(hover: none)").matches;
  let flow, count, focusable = [];

  // ---- geometry -------------------------------------------------------------------
  function landscape() {
    const yc = (i) => 62 + i * 58;          // row centre; row 0 = state token
    const TOK_X = 170, TOK_W = 92, H = 36, TF_X = 300, TF_W = 200;
    const Q_X = 540, Q_W = 150, G_X = 725, G_W = 190;
    const SIG = { x: 990, y: (yc(1) + yc(N)) / 2, r: 26 };
    return {
      viewBox: "0 0 1040 460", BUF: { x: 75, y: 230 }, SIG,
      buffer(g) {
        el("path", { d: "M15,150 L15,300 A60,16 0 0 0 135,300 L135,150", fill: "#fff", stroke: "#b8bec8", "stroke-width": 1.5 }, g);
        el("ellipse", { cx: 75, cy: 150, rx: 60, ry: 16, fill: "#f2f4f7", stroke: "#b8bec8", "stroke-width": 1.5 }, g);
        for (let y = 178; y <= 290; y += 14) el("line", { x1: 30, x2: 120, y1: y, y2: y, stroke: "#dde1e7", "stroke-width": 6, "stroke-linecap": "round" }, g);
        el("rect", { class: "chunk-hl", x: 26, y: 213, width: 98, height: 30, rx: 6, fill: "none", stroke: "var(--sear)", "stroke-width": 2.5 }, g);
        el("text", { x: 75, y: 330, "text-anchor": "middle", "font-size": 13, fill: "#6b7380", text: "Replay buffer" }, g);
      },
      tf: { x: TF_X, y: 30, w: TF_W, h: 352, lx: TF_X + TF_W / 2, ly: 20, fs: 14 },
      tokLabel: { x: TOK_X + TOK_W / 2, y: 20 },
      tok: (i) => ({ x: TOK_X, y: yc(i) - H / 2, w: TOK_W, h: H, cx: TOK_X + TOK_W / 2, cy: yc(i), fs: 15 }),
      tokLink: (i) => [TOK_X + TOK_W, yc(i), TF_X, yc(i)],
      att: (i, j) => `M${TF_X},${yc(i)} C${TF_X + 100},${yc(i)} ${TF_X + 100},${yc(j)} ${TF_X + TF_W},${yc(j)}`,
      qLink: (j) => [TF_X + TF_W, yc(j), Q_X - 3, yc(j)],
      q: (j) => ({ x: Q_X, y: yc(j) - H / 2, w: Q_W, h: H, cx: Q_X + Q_W / 2, ty: yc(j) + 5, fs: 14.5, label: `Q^{(${j})}(s_t, a_{t:t+${j}})` }),
      g: (j) => ({ x: G_X, y: yc(j) - H / 2, w: G_W, h: H, lx: G_X + 10, ly: yc(j) + 5, anchor: "start" }),
      reward: (j, i) => ({ x: G_X + 46 + i * 16, y: yc(j) - 6 }),
      pill: (j) => ({ x: G_X + 46 + j * 16 + 2, y: yc(j) - 10, w: 46, h: 20, fs: 13 }),
      errLink: (j) => [Q_X + Q_W + 2, yc(j), G_X - 2, yc(j)],
      toSig: (j) => `M${G_X + G_W},${yc(j)} C${G_X + G_W + 30},${yc(j)} ${SIG.x - 50},${SIG.y} ${SIG.x - SIG.r - 2},${SIG.y}`,
      sigText: [{ x: SIG.x, y: SIG.y + 46, a: "middle", t: "Bellman" }, { x: SIG.x, y: SIG.y + 60, a: "middle", t: "errors" }],
      count: { x: SIG.x, y: SIG.y + 82, a: "middle" },
      update: `M${SIG.x},${SIG.y + 92} L${SIG.x},430 L${TF_X + TF_W / 2},430 L${TF_X + TF_W / 2},${382 + 8}`,
      updLabel: { x: (SIG.x + TF_X) / 2 + 60, y: 448, fs: 13 },
      headers(gq, gg) {
        el("text", { x: Q_X + Q_W / 2, y: 60, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#a8323b", text: "one Q-value per prefix" }, gq);
        el("text", { x: G_X + G_W / 2, y: 54, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: "#4a5260", text: "multi-horizon targets" }, gg);
        el("rect", { x: G_X + 18, y: 66, width: 11, height: 11, rx: 2, fill: "var(--reward)" }, gg);
        el("text", { x: G_X + 34, y: 76, "font-size": 12, fill: "#6b7380", text: "reward" }, gg);
        el("rect", { x: G_X + 90, y: 65, width: 26, height: 13, rx: 6.5, fill: "#fff", stroke: "var(--action-stroke)" }, gg);
        el("text", { x: G_X + 122, y: 76, "font-size": 12, fill: "#6b7380", text: "target-critic bootstrap" }, gg);
      },
      hits: {
        buffer: [10, 130, 130, 210], tokens: [TOK_X - 5, 36, TOK_W + 10, yc(N) + 24 - 36], tf: [TF_X, 30, TF_W, 352],
        row: (j) => [Q_X - 6, yc(j) - 26, G_X + G_W - Q_X + 12, 52],
        sig: [SIG.x - 36, SIG.y - 36, 72, 130], update: [TF_X + TF_W / 2 - 12, 416, SIG.x - TF_X - TF_W / 2 + 24, 40],
      },
    };
  }

  function portrait() {
    const cx = (i) => 40 + i * 56;          // column centre; column 0 = state token
    const W = 50, TOK_Y = 103, TOK_H = 30, TF = { x: 12, y: 160, w: 336, h: 112 };
    const Q_Y = 286, Q_H = 34, G_Y = 342, G_H = 122, R_Y = 372;
    const SIG = { x: 180, y: 518, r: 22 };
    return {
      viewBox: "0 0 360 590", BUF: { x: 180, y: 50 }, SIG,
      buffer(g) {
        el("path", { d: "M130,20 L130,76 A50,12 0 0 0 230,76 L230,20", fill: "#fff", stroke: "#b8bec8", "stroke-width": 1.5 }, g);
        el("ellipse", { cx: 180, cy: 20, rx: 50, ry: 12, fill: "#f2f4f7", stroke: "#b8bec8", "stroke-width": 1.5 }, g);
        for (let y = 40; y <= 74; y += 9) el("line", { x1: 142, x2: 218, y1: y, y2: y, stroke: "#dde1e7", "stroke-width": 5, "stroke-linecap": "round" }, g);
        el("rect", { class: "chunk-hl", x: 137, y: 50, width: 86, height: 16, rx: 5, fill: "none", stroke: "var(--sear)", "stroke-width": 2.2 }, g);
        el("text", { x: 240, y: 55, "font-size": 12.5, fill: "#6b7380", text: "Replay buffer" }, g);
      },
      tf: { x: TF.x, y: TF.y, w: TF.w, h: TF.h, lx: 180, ly: 153, fs: 13 },
      tokLabel: null,
      tok: (i) => ({ x: cx(i) - W / 2, y: TOK_Y, w: W, h: TOK_H, cx: cx(i), cy: TOK_Y + TOK_H / 2, fs: 13 }),
      tokLink: (i) => [cx(i), TOK_Y + TOK_H, cx(i), TF.y],
      att: (i, j) => `M${cx(i)},${TF.y} C${cx(i)},${TF.y + TF.h / 2} ${cx(j)},${TF.y + TF.h / 2} ${cx(j)},${TF.y + TF.h}`,
      qLink: (j) => [cx(j), TF.y + TF.h, cx(j), Q_Y - 3],
      q: (j) => ({ x: cx(j) - W / 2, y: Q_Y, w: W, h: Q_H, cx: cx(j), ty: Q_Y + 22, fs: 13, label: `Q^{(${j})}` }),
      g: (j) => ({ x: cx(j) - W / 2, y: G_Y, w: W, h: G_H, lx: cx(j), ly: G_Y + 20, anchor: "middle" }),
      reward: (j, i) => ({ x: cx(j) - 6, y: R_Y + i * 15 }),
      pill: (j) => ({ x: cx(j) - 22, y: R_Y + j * 15 + 1, w: 44, h: 18, fs: 11 }),
      errLink: (j) => [cx(j), Q_Y + Q_H + 2, cx(j), G_Y - 2],
      toSig: (j) => `M${cx(j)},${G_Y + G_H} C${cx(j)},${G_Y + G_H + 30} ${SIG.x},${SIG.y - 50} ${SIG.x},${SIG.y - SIG.r - 2}`,
      sigText: [{ x: SIG.x + 32, y: SIG.y - 2, a: "start", t: "Bellman errors" }],
      count: { x: SIG.x + 32, y: SIG.y + 14, a: "start" },
      update: `M${SIG.x - SIG.r},${SIG.y} L5,${SIG.y} L5,${TF.y + TF.h / 2} L${TF.x - 1},${TF.y + TF.h / 2}`,
      updLabel: { x: 180, y: 570, fs: 12 },
      headers(gq, gg) {
        el("text", { x: cx(0), y: Q_Y + 15, "text-anchor": "middle", "font-size": 10.5, "font-weight": 600, fill: "#a8323b", text: "Q-value" }, gq);
        el("text", { x: cx(0), y: Q_Y + 28, "text-anchor": "middle", "font-size": 10.5, "font-weight": 600, fill: "#a8323b", text: "per prefix" }, gq);
        el("text", { x: cx(0), y: G_Y + 20, "text-anchor": "middle", "font-size": 10.5, "font-weight": 600, fill: "#4a5260", text: "targets" }, gg);
        el("rect", { x: cx(0) - 6, y: R_Y, width: 12, height: 12, rx: 2, fill: "var(--reward)" }, gg);
        el("text", { x: cx(0), y: R_Y + 25, "text-anchor": "middle", "font-size": 9.5, fill: "#6b7380", text: "reward" }, gg);
        el("rect", { x: cx(0) - 13, y: R_Y + 36, width: 26, height: 12, rx: 6, fill: "#fff", stroke: "var(--action-stroke)" }, gg);
        el("text", { x: cx(0), y: R_Y + 61, "text-anchor": "middle", "font-size": 9.5, fill: "#6b7380", text: "bootstrap" }, gg);
      },
      hits: {
        buffer: [120, 4, 200, 86], tokens: [10, TOK_Y - 3, 340, TOK_H + 6], tf: [TF.x, TF.y, TF.w, TF.h],
        row: (j) => [cx(j) - 28, Q_Y - 4, 56, G_Y + G_H - Q_Y + 8],
        sig: [SIG.x - 30, SIG.y - 28, 150, 56], update: [0, TF.y, 12, SIG.y - TF.y + 10],
      },
    };
  }

  // ---- build the diagram for the current layout ----------------------------------
  function build() {
    const L = portraitMQ.matches ? portrait() : landscape();
    const { SIG, BUF } = L;
    svg.innerHTML = "";
    svg.setAttribute("viewBox", L.viewBox);
    svg.closest(".hscroll").classList.toggle("portrait", portraitMQ.matches);

    const defs = el("defs", {}, svg);
    const mk = el("marker", { id: "arr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#6b7380" }, mk);

    L.buffer(el("g", {}, svg));

    // transformer block with causal attention: output j sees inputs 0..j only
    const tf = el("g", { class: "tf" }, svg);
    el("rect", { class: "tf-body", x: L.tf.x, y: L.tf.y, width: L.tf.w, height: L.tf.h, rx: 14, fill: "var(--tf-fill)", stroke: "var(--tf-stroke)", "stroke-width": 1.5 }, tf);
    el("text", { x: L.tf.lx, y: L.tf.ly, "text-anchor": "middle", "font-size": L.tf.fs, "font-weight": 600, fill: "#b8691b", stroke: "#f6f7f9", "stroke-width": 5, "paint-order": "stroke", text: "Causal Transformer" }, tf);
    for (let j = 1; j <= N; j++) {
      for (let i = 0; i <= j; i++) {
        el("path", {
          class: "att", pathLength: 1, fill: "none", stroke: "#d9822b", "stroke-width": 1.3, d: L.att(i, j),
          "data-tgt": j, "data-src": i, style: { "--d": `${((j - 1) * 0.55 + i * 0.07) / SPEED}s` },
        }, tf);
      }
    }

    // tokens fly in from the replay buffer
    for (let i = 0; i <= N; i++) {
      const isS = i === 0, t = L.tok(i);
      const g = el("g", {
        class: "tok", "data-src": i,
        style: { transform: `translate(${BUF.x - t.cx}px, ${BUF.y - t.cy}px)`, "--d": `${(i * 0.12) / SPEED}s` },
      }, svg);
      el("rect", { x: t.x, y: t.y, width: t.w, height: t.h, rx: 8, fill: isS ? "var(--state-fill)" : "var(--action-fill)", stroke: isS ? "var(--state-stroke)" : "var(--action-stroke)", "stroke-width": 1.5 }, g);
      const lbl = isS ? "s_t" : i === 1 ? "a_t" : `a_{t+${i - 1}}`;
      mathText(g, lbl, { x: t.cx, y: t.cy + 5, "text-anchor": "middle", "font-weight": 600, fill: "#1b1f24" }, t.fs);
      const [x1, y1, x2, y2] = L.tokLink(i);
      el("line", { x1, y1, x2, y2, stroke: isS ? "var(--state-stroke)" : "var(--action-stroke)", "stroke-width": 1.5 }, g);
    }
    if (L.tokLabel) el("text", { x: L.tokLabel.x, y: L.tokLabel.y, "text-anchor": "middle", "font-size": 13, fill: "#6b7380", text: "chunk tokens" }, svg);

    L.headers(el("g", { class: "pop q" }, svg), el("g", { class: "pop g" }, svg));

    // outputs, targets, errors
    for (let j = 1; j <= N; j++) {
      const d = (j - 1) * 0.18, mh = j < N ? " mh-only" : "";
      const [qx1, qy1, qx2, qy2] = L.qLink(j);
      el("line", { class: "draw q", pathLength: 1, x1: qx1, y1: qy1, x2: qx2, y2: qy2, stroke: "#d6505a", "stroke-width": 1.5, "marker-end": "url(#arr)", style: { "--d": `${d / SPEED}s` }, "data-row": j }, svg);
      const qb = L.q(j);
      const q = el("g", { class: "pop q", "data-row": j, style: { "--d": `${(d + 0.15) / SPEED}s` } }, svg);
      el("rect", { x: qb.x, y: qb.y, width: qb.w, height: qb.h, rx: 8, fill: "var(--q-fill)", stroke: "var(--q-stroke)", "stroke-width": 1.5 }, q);
      mathText(q, qb.label, { x: qb.cx, y: qb.ty, "text-anchor": "middle", fill: "#1b1f24" }, qb.fs);

      const gb = L.g(j);
      const gw = el("g", { class: "mh-wrap" + mh, "data-row": j }, svg);
      const g = el("g", { class: "pop g", style: { "--d": `${((j - 1) * 0.6) / SPEED}s` } }, gw);
      el("rect", { x: gb.x, y: gb.y, width: gb.w, height: gb.h, rx: 8, fill: "var(--g-fill)", stroke: "var(--g-stroke)", "stroke-width": 1.5 }, g);
      mathText(g, `G^{(${j})}`, { x: gb.lx, y: gb.ly, "text-anchor": gb.anchor, fill: "#1b1f24", "font-weight": 600 }, 14.5);
      for (let i = 0; i < j; i++) {
        const r = L.reward(j, i);
        el("rect", { class: "rw", x: r.x, y: r.y, width: 12, height: 12, rx: 2, fill: "var(--reward)", style: { "--d": `${((j - 1) * 0.6 + 0.2 + i * 0.07) / SPEED}s` } }, gw);
      }
      const pb = L.pill(j);
      const bs = el("g", { class: "rw", style: { "--d": `${((j - 1) * 0.6 + 0.3 + j * 0.07) / SPEED}s` } }, gw);
      el("rect", { x: pb.x, y: pb.y, width: pb.w, height: pb.h, rx: pb.h / 2, fill: "#fff", stroke: "var(--action-stroke)" }, bs);
      mathText(bs, `γ^{${j}}Q`, { x: pb.x + pb.w / 2, y: pb.y + pb.h / 2 + 4.5, "text-anchor": "middle", fill: "#1b1f24" }, pb.fs);

      const e = el("g", { class: "mh-wrap" + mh, "data-row": j }, svg);
      const [ex1, ey1, ex2, ey2] = L.errLink(j);
      el("line", { class: "draw e", pathLength: 1, x1: ex1, y1: ey1, x2: ex2, y2: ey2, stroke: "var(--low)", "stroke-width": 3, "stroke-dasharray": "1", style: { "--d": `${d / SPEED}s` } }, e);
      el("path", { class: "draw e", pathLength: 1, fill: "none", stroke: "#6b7380", "stroke-width": 1.4, "marker-end": "url(#arr)", d: L.toSig(j), style: { "--d": `${(d + 0.3) / SPEED}s` } }, e);
    }

    const sig = el("g", { class: "pop e", style: { "--d": `${0.9 / SPEED}s` } }, svg);
    el("circle", { cx: SIG.x, cy: SIG.y, r: SIG.r, fill: "#fff", stroke: "#6b7380", "stroke-width": 1.5 }, sig);
    el("text", { x: SIG.x, y: SIG.y + 8, "text-anchor": "middle", "font-size": SIG.r - 2, fill: "#1b1f24", text: "Σ" }, sig);
    L.sigText.forEach((t) => el("text", { x: t.x, y: t.y, "text-anchor": t.a, "font-size": 12, fill: "#6b7380", text: t.t }, sig));
    count = el("text", { x: L.count.x, y: L.count.y, "text-anchor": L.count.a, "font-size": 12.5, "font-weight": 700, fill: "var(--sear)" }, sig);

    // update loop back into the transformer
    el("path", { class: "draw u", pathLength: 1, d: L.update, fill: "none", stroke: "#6b7380", "stroke-width": 1.6, "marker-end": "url(#arr)" }, svg);
    flow = el("path", { class: "flow", d: L.update, fill: "none", stroke: "var(--sear)", "stroke-width": 2.2, "stroke-dasharray": "6 10", opacity: 0 }, svg);
    el("text", { class: "pop u", x: L.updLabel.x, y: L.updLabel.y, "text-anchor": "middle", "font-size": L.updLabel.fs, fill: "#4a5260", text: "gradient update of the critic parameters φ" }, svg);

    // hover (or tap): explain components; for a Q row, show which inputs it sees
    focusable = svg.querySelectorAll("[data-src], [data-row]");
    const H = L.hits;
    hitArea(H.buffer, () => captions()[0]);
    hitArea(H.tokens, () => captions()[1]);
    hitArea(H.tf, () => captions()[2]);
    for (let j = 1; j <= N; j++) hitArea(H.row(j), () => hoverCaption(j), j);
    hitArea(H.sig, () => captions()[5]);
    hitArea(H.update, () => UPDATE);
  }

  function hitArea([x, y, w, h], html, row) {
    const r = el("rect", { class: "hit", x, y, width: w, height: h, fill: "transparent" }, svg);
    const enter = () => { hoverHtml = html(); focus(row || 0); refreshCaption(); };
    const leave = () => { hoverHtml = null; focus(0); refreshCaption(); };
    if (touch) {
      r.addEventListener("click", (ev) => { ev.stopPropagation(); enter(); });
    } else {
      r.addEventListener("mouseenter", enter);
      r.addEventListener("mouseleave", leave);
    }
  }
  // on touch screens, tapping outside a component clears the explanation
  if (touch) {
    document.addEventListener("click", (ev) => {
      if (hoverHtml && !ev.target.closest(".hit")) { hoverHtml = null; focus(0); refreshCaption(); }
    });
  }

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
  const HINT = "<span style='color:var(--ink-soft)'>" + (touch ? "Tap" : "Hover over") + " any part of the diagram (replay buffer, tokens, transformer, Q-values and targets, Bellman errors, update) to see what it does.</span>";
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
  function updateCount() { count.textContent = mh ? `${N} targets` : "1 target"; }
  function setMH() {
    mh = document.querySelector("input[name=critic-target]:checked").value === "mh";
    svg.classList.toggle("nomh", !mh);
    updateCount();
    renderMath(mhText, mh ? mhNotes.on : mhNotes.off);
    refreshCaption();
  }
  mhRadios.forEach((r) => (r.onchange = setMH));

  build();
  go(0);
  setMH();
  // switch layouts when the screen crosses the phone breakpoint
  portraitMQ.addEventListener("change", () => { build(); updateCount(); go(stage); });
  // start once the diagram scrolls into view
  const io = new IntersectionObserver((es) => {
    if (es[0].isIntersecting) { playing = true; schedule(); io.disconnect(); }
  }, { threshold: 0.4 });
  io.observe(svg);
})();
