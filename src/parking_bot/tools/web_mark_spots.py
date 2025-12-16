import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Parking spots marker</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; }
    .row { display: flex; gap: 16px; }
    canvas { border: 1px solid #ccc; max-width: 100%; height: auto; }
    .panel { min-width: 320px; }
    button { padding: 8px 12px; margin-right: 8px; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
    ul { padding-left: 18px; }
  </style>
</head>
<body>
  <h2>Parking spots marker</h2>
  <p>
    - Click: add point<br/>
    - <code>n</code>: finish polygon (asks for id)<br/>
    - <code>u</code>: undo last point<br/>
    - <code>Esc</code>: clear current polygon
  </p>

  <div class="row">
    <div>
      <canvas id="c"></canvas>
    </div>
    <div class="panel">
      <div style="margin-bottom: 12px;">
        <button id="save">Save</button>
        <button id="clear">Clear all</button>
      </div>
      <div>
        <div><b>Spots:</b> <span id="count">0</span></div>
        <ul id="list"></ul>
      </div>
      <div id="status" style="margin-top: 12px;"></div>
    </div>
  </div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const listEl = document.getElementById('list');
const countEl = document.getElementById('count');

let img = new Image();
let current = []; // [[x,y],...]
let spots = [];   // [{id, polygon:[[x,y],...]}, ...]

function setStatus(msg) { statusEl.textContent = msg; }

function redraw() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0);

  // existing spots
  for (const s of spots) {
    drawPoly(s.polygon, 'rgba(0,180,0,0.9)', 2);
    const [cx, cy] = centroid(s.polygon);
    drawText(s.id, cx, cy, 'rgba(0,180,0,0.95)');
  }

  // current
  if (current.length) {
    drawPoly(current, 'rgba(220,0,0,0.9)', 2, false);
    for (const [x,y] of current) {
      ctx.beginPath();
      ctx.arc(x,y,3,0,Math.PI*2);
      ctx.fillStyle = 'rgba(220,0,0,0.9)';
      ctx.fill();
    }
  }
}

function drawText(text, x, y, color) {
  ctx.font = '16px sans-serif';
  ctx.fillStyle = color;
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.lineWidth = 3;
  ctx.strokeText(text, x+4, y-4);
  ctx.fillText(text, x+4, y-4);
}

function drawPoly(points, color, width, closed=true) {
  if (!points || points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i=1;i<points.length;i++) ctx.lineTo(points[i][0], points[i][1]);
  if (closed) ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function centroid(points) {
  let sx=0, sy=0;
  for (const p of points) { sx += p[0]; sy += p[1]; }
  return [sx/points.length, sy/points.length];
}

function refreshList() {
  listEl.innerHTML = '';
  for (const s of spots) {
    const li = document.createElement('li');
    li.textContent = `${s.id} (${s.polygon.length} pts)`;
    listEl.appendChild(li);
  }
  countEl.textContent = String(spots.length);
}

canvas.addEventListener('click', (ev) => {
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((ev.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.round((ev.clientY - rect.top) * (canvas.height / rect.height));
  current.push([x,y]);
  redraw();
});

document.addEventListener('keydown', (ev) => {
  if (ev.key === 'u') {
    current.pop();
    redraw();
  }
  if (ev.key === 'Escape') {
    current = [];
    redraw();
  }
  if (ev.key === 'n') {
    if (current.length < 3) {
      setStatus('Need >=3 points for polygon');
      return;
    }
    const sid = prompt('Spot id (e.g. A1):', `spot_${spots.length+1}`) || `spot_${spots.length+1}`;
    spots.push({id: sid, polygon: current});
    current = [];
    refreshList();
    redraw();
    setStatus('Added spot ' + sid);
  }
});

document.getElementById('clear').addEventListener('click', () => {
  if (!confirm('Clear all spots?')) return;
  spots = [];
  current = [];
  refreshList();
  redraw();
});

document.getElementById('save').addEventListener('click', async () => {
  const payload = {
    spots,
    // image_size will be filled by server
  };
  const r = await fetch('/save', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(payload),
  });
  const t = await r.text();
  setStatus(t);
});

async function init() {
  const r = await fetch('/meta');
  const meta = await r.json();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    redraw();
  };
  img.src = '/image';
  setStatus(`Image: ${meta.width}x${meta.height}`);
}

init();
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Mark parking spots in browser (no OpenCV GUI)")
    p.add_argument("--image", required=True, help="Path to frame image")
    p.add_argument("--out", default="data/spots.json", help="Output JSON file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = Image.open(img_path).size

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
                return
            if path == "/meta":
                self._send(200, "application/json", json.dumps({"width": w, "height": h}).encode("utf-8"))
                return
            if path == "/image":
                data = img_path.read_bytes()
                ctype = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                self._send(200, ctype, data)
                return
            self._send(404, "text/plain; charset=utf-8", b"Not found")

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/save":
                self._send(404, "text/plain; charset=utf-8", b"Not found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            spots = data.get("spots", [])

            payload = {"image_size": [w, h], "spots": spots}
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._send(200, "text/plain; charset=utf-8", f"Saved: {out_path}".encode("utf-8"))

        def log_message(self, format, *args):
            return

    server = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Open: {url}")
    print("Press Ctrl+C to stop")

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
