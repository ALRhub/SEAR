// Small SVG helpers shared by the animations.
window.SVGH = (function () {
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) {
      if (k === "text") e.textContent = attrs[k];
      else if (k === "style") Object.assign(e.style, attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }

  // Render a tiny TeX-like string ("Q^{(2)}(s_t, a_{t:t+2})") as <text> with
  // sub/superscript <tspan>s. Uses dy offsets so it works in every browser.
  function mathText(parent, str, attrs, fs) {
    fs = fs || 14;
    const t = el("text", Object.assign({ "font-size": fs }, attrs), parent);
    const re = /([_^])(\{([^}]*)\}|(.))/g;
    let last = 0, shift = 0, m;
    const push = (txt, kind) => {
      if (!txt) return;
      const target = kind === "_" ? fs * 0.3 : kind === "^" ? -fs * 0.4 : 0;
      const sp = el("tspan", { text: txt }, t);
      if (target !== shift) sp.setAttribute("dy", (target - shift).toFixed(1));
      if (kind) sp.setAttribute("font-size", (fs * 0.7).toFixed(1));
      shift = target;
    };
    while ((m = re.exec(str))) {
      push(str.slice(last, m.index), null);
      push(m[3] !== undefined ? m[3] : m[4], m[1]);
      last = re.lastIndex;
    }
    push(str.slice(last), null);
    return t;
  }

  return { NS, el, mathText };
})();
