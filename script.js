/* Local-only interactions: theme, progress, scrollspy, figure-mount, demo, heatmap. No network. */
(function(){
  "use strict";
  var root = document.documentElement;

  /* ---- theme: persist, respect OS, no flash ---- */
  function currentTheme(){
    try{ var s = localStorage.getItem("recirc-theme"); if(s) return s; }catch(e){}
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light";
  }
  function applyTheme(t){
    root.setAttribute("data-theme", t);
    var b = document.getElementById("theme-toggle");
    if(b) b.textContent = (t === "dark") ? "☀ light" : "☾ dark";
    try{ localStorage.setItem("recirc-theme", t); }catch(e){}
  }
  applyTheme(currentTheme());
  document.addEventListener("click", function(e){
    if(e.target && e.target.id === "theme-toggle") applyTheme(root.getAttribute("data-theme")==="dark"?"light":"dark");
    var top = e.target && e.target.closest && e.target.closest("#to-top");
    if(top) window.scrollTo({top:0, behavior:(window.matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth")});
  });

  /* ---- reading progress + back-to-top ---- */
  var bar = document.getElementById("progress"), toTop = document.getElementById("to-top");
  function onScroll(){
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var p = max > 0 ? (h.scrollTop / max) : 0;
    if(bar) bar.style.width = (p*100).toFixed(2) + "%";
    if(toTop) toTop.classList.toggle("show", h.scrollTop > 800);
  }
  document.addEventListener("scroll", onScroll, {passive:true}); onScroll();

  /* ---- scrollspy with robust bottom detection ---- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".toc a[href^='#'], .mobile-toc a[href^='#']"));
  var secs = links.map(function(a){ return document.querySelector(a.getAttribute("href")); }).filter(Boolean);
  var uniqueSecs = Array.from(new Set(secs));
  if("IntersectionObserver" in window && uniqueSecs.length){
    var active = null;
    var obs = new IntersectionObserver(function(entries){
      entries.forEach(function(en){
        if(en.isIntersecting){
          active = "#" + en.target.id;
          links.forEach(function(a){ a.classList.toggle("active", a.getAttribute("href")===active); });
        }
      });
    }, {rootMargin:"-20% 0px -65% 0px"});
    uniqueSecs.forEach(function(s){ obs.observe(s); });
    /* ensure bottom-of-page activates appendix */
    window.addEventListener("scroll", function(){
      if((window.innerHeight + window.scrollY) >= (document.documentElement.scrollHeight - 60)){
        links.forEach(function(a){ a.classList.toggle("active", a.getAttribute("href")==="#appendix"); });
      }
    }, {passive:true});
  }

  /* ---- figure injection from local figures.html (Chinmay reference pattern) ---- */
  var figureFileCache = new Map();
  async function injectFigures(rootNode) {
    var mounts = Array.prototype.slice.call((rootNode || document).querySelectorAll(".fig-mount[data-fig]"));
    if (!mounts.length) return;
    var inlineTpl = document.getElementById("fig-inline");
    var inline = inlineTpl ? inlineTpl.content : null;
    var sources = mounts.map(function(m){ return m.getAttribute("data-src") || "figures.html"; });
    var uniqueSources = Array.from(new Set(sources));
    await Promise.all(uniqueSources.map(async function(src) {
      if (figureFileCache.has(src)) return;
      try {
        var resp = await fetch(src);
        if (resp.ok) {
          var html = await resp.text();
          var doc = new DOMParser().parseFromString(html, "text/html");
          figureFileCache.set(src, doc);
        }
      } catch (e) {
        console.warn("Local figure fetch failed for " + src, e);
      }
    }));
    mounts.forEach(function(mount) {
      var figId = mount.getAttribute("data-fig");
      var src = mount.getAttribute("data-src") || "figures.html";
      var node = inline ? inline.querySelector("#" + figId) : null;
      if (!node) {
        var doc = figureFileCache.get(src);
        node = doc ? doc.getElementById(figId) : null;
      }
      if (node) {
        mount.replaceWith(document.importNode(node, true));
      }
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function(){ injectFigures(); });
  } else {
    injectFigures();
  }

  /* ---- coherent heatmap (alpha = 0.10 slice, n=100, baseline 0.25) ---- */
  var heat = document.getElementById("heatmap");
  if(heat){
    var rows = [
      {s:"s16", cells:[{d:"d7",acc:.33},{d:"d9",acc:.24}]},
      {s:"s18", cells:[{d:"d7",acc:.38},{d:"d9",acc:.33}]},
      {s:"s20", cells:[{d:"d7",acc:.33},{d:"d9",acc:.30}]}
    ];
    var base = 0.25, html = '<div class="h corner"></div><div class="h colh">dest 7</div><div class="h colh">dest 9</div>';
    rows.forEach(function(r){
      html += '<div class="h rowh mono">src '+r.s.slice(1)+'</div>';
      r.cells.forEach(function(c){
        var d = c.acc - base;
        var t = Math.max(0, Math.min(1, (d + 0.02) / 0.15));
        var cold = (d < 0);
        var bg = cold ? "background:var(--card);" : "background:var(--accent-soft);background:color-mix(in srgb, var(--accent) "+Math.round(8+t*26)+"% , var(--card));";
        var border = cold ? "border:1.5px dashed var(--bad);" : "";
        html += '<div class="h cell" style="'+bg+border+'" title="'+r.s+'→'+c.d+': '+c.acc.toFixed(2)+' (Δ '+(d>=0?"+":"")+d.toFixed(2)+' vs 0.25)">'
          + '<b class="mono">'+c.acc.toFixed(2)+'</b><small>Δ '+(d>=0?"+":"")+d.toFixed(2)+'</small></div>';
      });
    });
    heat.innerHTML = html;
  }

  /* ---- interactive recirculation demo (SVG, matching design tokens) ---- */
  var svg = document.getElementById("demo-svg");
  var alpha = document.getElementById("alpha"), alphaVal = document.getElementById("alpha-val"),
      playBtn = document.getElementById("play"), stepLbl = document.getElementById("step-lbl"),
      mixLbl = document.getElementById("mix-lbl"), modeSel = document.getElementById("sched");
  if(svg && alpha){
    var NS = "http://www.w3.org/2000/svg";
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var t = 0, playing = false, timer = null;
    var css = function(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); };
    function draw(){
      while(svg.firstChild) svg.removeChild(svg.firstChild);
      var W=640,H=300;
      svg.setAttribute("viewBox","0 0 "+W+" "+H);
      var ink = css("--ink")||"#333", soft = css("--ink-soft")||"#777",
          deep = css("--deep")||"#B3401F", shal = css("--shallow")||"#2F43C4",
          line = css("--line")||"#ddd", card = css("--card")||"#fff",
          accent = css("--accent")||"#2F43C4";
      function stack(x,label,active){
        var g = document.createElementNS(NS,"g");
        var r = document.createElementNS(NS,"rect");
        r.setAttribute("x",x); r.setAttribute("y",30); r.setAttribute("width",120); r.setAttribute("height",210);
        r.setAttribute("rx",10); r.setAttribute("fill", active ? "none" : "var(--paper-2)");
        r.setAttribute("stroke",line); r.setAttribute("stroke-width", active ? "1.5" : "1");
        if(!active) r.setAttribute("stroke-dasharray","3 3");
        g.appendChild(r);
        [["shallow · d (layer 9)",205,shal],["deep · s (layer 18)",70,deep]].forEach(function(L){
          var ln = document.createElementNS(NS,"line");
          ln.setAttribute("x1",x); ln.setAttribute("x2",x+120); ln.setAttribute("y1",L[1]); ln.setAttribute("y2",L[1]);
          ln.setAttribute("stroke",L[2]); ln.setAttribute("stroke-width",1.2); ln.setAttribute("stroke-dasharray","3 4"); ln.setAttribute("opacity",".8");
          g.appendChild(ln);
          var tx = document.createElementNS(NS,"text");
          tx.setAttribute("x",x+126); tx.setAttribute("y",L[1]+4); tx.setAttribute("font-size","9.5");
          tx.setAttribute("fill",soft); tx.setAttribute("font-family","var(--sans)"); tx.textContent = L[0];
          g.appendChild(tx);
        });
        var lb = document.createElementNS(NS,"text");
        lb.setAttribute("x",x+60); lb.setAttribute("y",262); lb.setAttribute("text-anchor","middle");
        lb.setAttribute("font-size","11.5"); lb.setAttribute("font-weight","600"); lb.setAttribute("fill",ink); lb.setAttribute("font-family","var(--sans)");
        lb.textContent = label; g.appendChild(lb);
        svg.appendChild(g);
      }
      var a = parseFloat(alpha.value);
      var mode = modeSel ? modeSel.value : "cross";
      var x0=60, x1=260, x2=460;
      stack(x0, mode==="cross" ? "step t (warm-up)" : "step t+1 · pass 1", true);
      stack(x1, mode==="cross" ? "step t+1" : "step t+1 · pass 2", true);
      stack(x2, mode==="cross" ? "step t+2" : "step t+2 …", false);
      function dot(x,y,c,r){
        var e=document.createElementNS(NS,"circle");
        e.setAttribute("cx",x); e.setAttribute("cy",y); e.setAttribute("r",r||7);
        e.setAttribute("fill",c); e.setAttribute("stroke",card); e.setAttribute("stroke-width",2);
        svg.appendChild(e); return e;
      }
      function arc(x0,y0,x1,y1,c){
        var p=document.createElementNS(NS,"path");
        var mx=(x0+x1)/2;
        p.setAttribute("d","M"+x0+","+y0+" C"+mx+","+(y0-46)+" "+mx+","+(y1-46)+" "+x1+","+y1);
        p.setAttribute("fill","none"); p.setAttribute("stroke",c); p.setAttribute("stroke-width",2.2);
        svg.appendChild(p);
        var ah=document.createElementNS(NS,"path");
        ah.setAttribute("d","M"+x1+","+y1+" l-9,-3 l3.5,6 z");
        ah.setAttribute("fill",c); svg.appendChild(ah);
      }
      var ph = t % 3;
      if(mode==="cross"){
        dot(x0+60,70,deep); dot(x1+60,205,shal);
        arc(x0+60,70,x1+60,205,deep);
        if(ph>=1){ dot(x1+60,70,deep); dot(x2+60,205,shal); arc(x1+60,70,x2+60,205,deep); }
      }else{
        dot(x0+60,70,deep); dot(x1+60,205,shal);
        arc(x0+60,70,x1+60,205,deep);
        var star=document.createElementNS(NS,"text");
        star.setAttribute("x",x1+60); star.setAttribute("y",120); star.setAttribute("text-anchor","middle");
        star.setAttribute("font-size","12"); star.setAttribute("font-weight","700"); star.setAttribute("fill",accent); star.textContent="★ logits from rerun";
        svg.appendChild(star);
      }
      /* moving pulse */
      var px = x0+60 + (ph/2)*(x1-x0);
      if(mode==="cross" && ph<2){
        var pu=document.createElementNS(NS,"circle");
        pu.setAttribute("cx",px); pu.setAttribute("cy",120); pu.setAttribute("r",4.5);
        pu.setAttribute("fill",deep); pu.setAttribute("opacity",".85"); svg.appendChild(pu);
        if(!reduce){ pu.innerHTML='<animate attributeName="opacity" values=".85;.2;.85" dur="1.2s" repeatCount="indefinite"/>'; }
      }
      if(stepLbl) stepLbl.textContent = "t = " + t;
      if(alphaVal) alphaVal.textContent = "α = " + a.toFixed(2);
      var beta = (document.getElementById("beta") && document.getElementById("beta").value === "convex")
        ? (1 - a).toFixed(2) : "1.00";
      if(mixLbl) mixLbl.textContent = "d′ = " + a.toFixed(2) + "·f(s) + " + beta + "·d";
    }
    function tick(){ t++; draw(); }
    if(playBtn) playBtn.addEventListener("click", function(){
      if(reduce){
        tick();
        return;
      }
      if(playing){ clearInterval(timer); playing=false; playBtn.textContent="▶ Play steps"; return; }
      playing=true; playBtn.textContent="⏸ Pause";
      timer=setInterval(tick, 1100);
    });
    ["input","change"].forEach(function(ev){
      alpha.addEventListener(ev, draw);
      if(modeSel) modeSel.addEventListener(ev, draw);
      var b=document.getElementById("beta"); if(b) b.addEventListener(ev, draw);
    });
    draw();
  }
})();
