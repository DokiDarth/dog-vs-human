"""A tiny local website for the dog-vs-human model. No extra installs needed:
it uses Python's built-in web server.

Usage:
    python app.py [--port 8000]
then open http://localhost:8000 in your browser.
"""
import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image

from predict import load_model
from train import eval_tf, pick_device

MAX_UPLOAD = 15 * 1024 * 1024  # 15 MB

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dog or Human?</title>
<style>
  :root { --bg:#f6f5f2; --card:#fff; --text:#1d1d1f; --muted:#6e6e73; --line:#dcdad5;
          --dog:#c2410c; --human:#1d4ed8; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161616; --card:#212121; --text:#f2f2f2; --muted:#a1a1a6; --line:#3a3a3a;
            --dog:#fb923c; --human:#60a5fa; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  main { max-width:520px; margin:48px auto; padding:0 16px; }
  h1 { font-size:28px; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 24px; }
  #drop { display:block; border:2px dashed var(--line); border-radius:14px; background:var(--card);
          padding:28px; text-align:center; cursor:pointer; transition:border-color .15s; }
  #drop.over { border-color:var(--human); }
  #drop img { max-width:100%; max-height:320px; border-radius:10px; display:none; margin:0 auto 12px; }
  #hint { color:var(--muted); }
  input[type=file] { display:none; }
  #result { margin-top:20px; background:var(--card); border:1px solid var(--line);
            border-radius:14px; padding:20px; display:none; }
  #label { font-size:24px; font-weight:650; margin:0 0 12px; }
  .row { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:14px; }
  .row span:first-child { width:52px; }
  .bar { flex:1; height:10px; background:var(--line); border-radius:5px; overflow:hidden; }
  .fill { height:100%; width:0; transition:width .4s; }
  .row span:last-child { width:52px; text-align:right; font-variant-numeric:tabular-nums; }
  #note { color:var(--muted); font-size:14px; margin:12px 0 0; }
</style>
</head>
<body>
<main>
  <h1>Dog or Human?</h1>
  <p class="sub">Drop a photo below and the model will guess what it is.</p>
  <label id="drop">
    <img id="preview" alt="Your photo">
    <span id="hint">Click to choose a photo, or drag one here</span>
    <input id="file" type="file" accept="image/*">
  </label>
  <section id="result">
    <p id="label"></p>
    <div id="bars"></div>
    <p id="note"></p>
  </section>
</main>
<script>
const drop = document.getElementById("drop"), file = document.getElementById("file");
const preview = document.getElementById("preview"), hint = document.getElementById("hint");
const result = document.getElementById("result"), label = document.getElementById("label");
const bars = document.getElementById("bars"), note = document.getElementById("note");

file.addEventListener("change", () => file.files[0] && classify(file.files[0]));
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) classify(e.dataTransfer.files[0]);
});

async function classify(img) {
  preview.src = URL.createObjectURL(img);
  preview.style.display = "block";
  hint.textContent = "Thinking...";
  result.style.display = "none";
  try {
    const res = await fetch("/predict", { method: "POST", body: img });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    show(data);
    hint.textContent = "Click or drop another photo";
  } catch (err) {
    hint.textContent = "Error: " + err.message;
  }
}

function show(data) {
  const pct = p => (p * 100).toFixed(1) + "%";
  label.textContent = (data.label === "dog" ? "🐶 Dog" : "🧑 Human") + " (" + pct(data.confidence) + " sure)";
  label.style.color = `var(--${data.label})`;
  bars.innerHTML = Object.entries(data.probabilities).map(([name, p]) =>
    `<div class="row"><span>${name}</span><div class="bar"><div class="fill" style="background:var(--${name})" data-w="${p * 100}"></div></div><span>${pct(p)}</span></div>`
  ).join("");
  requestAnimationFrame(() => bars.querySelectorAll(".fill").forEach(f => f.style.width = f.dataset.w + "%"));
  note.textContent = data.confidence < 0.9
    ? "Not very sure. The model only knows dogs and humans, so it may be something else."
    : "";
  result.style.display = "block";
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            return self.send_json({"error": "not found"}, 404)
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/predict":
            return self.send_json({"error": "not found"}, 404)
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= MAX_UPLOAD:
            return self.send_json({"error": "send an image up to 15 MB"}, 400)
        try:
            img = Image.open(io.BytesIO(self.rfile.read(size))).convert("RGB")
        except Exception:
            return self.send_json({"error": "that file isn't an image I can read"}, 400)

        x = eval_tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(MODEL(x), dim=1)[0].tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        self.send_json({
            "label": CLASSES[best],
            "confidence": probs[best],
            "probabilities": dict(zip(CLASSES, probs)),
        })

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    DEVICE = pick_device()
    MODEL, CLASSES = load_model(DEVICE)  # load once, reuse for every request

    print(f"Dog-vs-human app running at http://localhost:{args.port}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
