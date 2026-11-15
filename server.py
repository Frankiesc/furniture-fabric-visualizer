# /// script
# requires-python = ">=3.11"
# dependencies = ["mflux"]
# ///
"""Fabric Fitting Room: a local web app that re-renders a furniture photo in a
new fabric using FLUX.2 klein running on this Mac (via mflux / MLX).

Start it with:  bash run.sh      (or: uv run server.py)
Then open:      http://127.0.0.1:8765
"""
import base64
import gc
import inspect
import io
import json
import os
import random
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
RENDERS = ROOT / "renders"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8765"))
QUANTIZE = int(os.environ.get("QUANTIZE", "8"))  # 8-bit weights; 0 = full precision
MAX_BODY = 80 * 1024 * 1024

# name shown in the page -> ModelConfig factory method in mflux
MODELS = {
    "flux2-klein-9b": "flux2_klein_9b",
    "flux2-klein-4b": "flux2_klein_4b",
}

BASE_PROMPT = (
    "Reupholster the piece of furniture in image 1 with the fabric shown in image 2. "
    "Cover every upholstered surface (seat, back, arms and cushions) with that exact "
    "fabric: the same color, pattern, weave and sheen, at a realistic scale, following "
    "the existing seams, folds and contours. Keep the furniture's shape and proportions, "
    "its legs and any wood or metal parts, the room, the camera angle and the lighting "
    "exactly as they are in image 1. Photorealistic result."
)

_render_lock = threading.Lock()
_state = {"model_name": None, "model": None, "phase": "idle", "progress": None, "download": None}


class StepProgress:
    """mflux callback that reports how far the current render has got."""

    def call_before_loop(self, seed, prompt, latents, config, **kwargs):
        _state["progress"] = {"step": 0, "steps": config.num_inference_steps}

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **kwargs):
        import mlx.core as mx

        # MLX is lazy: finish this step's work before counting it as done.
        mx.eval(latents)
        progress = _state["progress"]
        _state["progress"] = {"step": progress["step"] + 1, "steps": progress["steps"]}


def _track_downloads():
    """Report model download progress to the page. mflux downloads weights with
    huggingface_hub's snapshot_download, which takes a tqdm class for its progress."""
    from huggingface_hub.utils import tqdm
    from mflux.models.common.resolution import path_resolution

    if getattr(path_resolution.snapshot_download, "tracked", False):
        return

    class Progress(tqdm):
        # snapshot_download keeps one byte counter for the whole model, and its total
        # grows as each file starts. Count here too: a bar that isn't shown (no
        # terminal) does not keep its own count.
        def __init__(self, *args, **kwargs):
            self.tracked = kwargs.get("name") == "huggingface_hub.snapshot_download"
            self.done = kwargs.get("initial") or 0
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if self.tracked:
                self.done += n or 0
                _state["download"] = {"done": self.done, "total": self.total or 0}
            return super().update(n)

    original = path_resolution.snapshot_download

    def snapshot_download(*args, **kwargs):
        if kwargs.get("local_files_only"):
            return original(*args, **kwargs)
        _state["phase"] = "downloading"
        try:
            return original(*args, tqdm_class=Progress, **kwargs)
        finally:
            _state["phase"] = "loading"
            _state["download"] = None

    snapshot_download.tracked = True
    path_resolution.snapshot_download = snapshot_download


def _accepted(func, kwargs):
    """Drop keyword arguments the installed mflux version does not take."""
    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def load_model(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(MODELS)}.")
    if _state["model_name"] == name:
        return _state["model"]

    _state["phase"] = "loading"
    _state["model"] = None
    _state["model_name"] = None
    gc.collect()

    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2KleinEdit

    _track_downloads()
    config = getattr(ModelConfig, MODELS[name])()
    kwargs = {"model_config": config}
    if QUANTIZE:
        kwargs["quantize"] = QUANTIZE
    print(f"[model] loading {name} (quantize={QUANTIZE or 'off'}). "
          "The first run downloads the weights; progress shows below.", flush=True)
    model = Flux2KleinEdit(**_accepted(Flux2KleinEdit.__init__, kwargs))
    model.callbacks.register(StepProgress())
    _state["model"] = model
    _state["model_name"] = name
    print(f"[model] {name} ready", flush=True)
    return model


def decode_image(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Image data is missing or not a data URL.")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    return img.convert("RGB")


def fit_size(width, height, long_side):
    """Scale to the requested long side, keeping the aspect ratio, in multiples of 16."""
    scale = long_side / max(width, height)
    w = max(256, int(round(width * scale / 16)) * 16)
    h = max(256, int(round(height * scale / 16)) * 16)
    return w, h


def render(payload):
    model_name = payload.get("model", "flux2-klein-9b")
    steps = max(1, min(int(payload.get("steps", 4)), 12))
    long_side = max(512, min(int(payload.get("size", 1024)), 1536))
    seed = payload.get("seed")
    seed = int(seed) if seed not in (None, "") else random.randint(0, 2**31 - 1)
    notes = str(payload.get("notes", "")).strip()[:500]
    label = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(payload.get("label", "fabric")))[:40]

    furniture = decode_image(payload.get("furniture"))
    fabric = decode_image(payload.get("fabric"))
    width, height = fit_size(*furniture.size, long_side)
    furniture = furniture.resize((width, height), Image.LANCZOS)
    fabric.thumbnail((1024, 1024), Image.LANCZOS)

    prompt = BASE_PROMPT + (f" Additional instructions: {notes}" if notes else "")

    with _render_lock:
        started = time.time()
        try:
            model = load_model(model_name)
            _state["phase"] = "rendering"
            _state["progress"] = None
            with tempfile.TemporaryDirectory() as tmp:
                furniture_path = os.path.join(tmp, "furniture.png")
                fabric_path = os.path.join(tmp, "fabric.png")
                furniture.save(furniture_path)
                fabric.save(fabric_path)
                kwargs = {
                    "seed": seed,
                    "prompt": prompt,
                    "image_paths": [furniture_path, fabric_path],
                    "num_inference_steps": steps,
                    "width": width,
                    "height": height,
                    "guidance": 1.0,
                }
                result = model.generate_image(**_accepted(model.generate_image, kwargs))

                RENDERS.mkdir(exist_ok=True)
                out_path = RENDERS / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}-{seed}.png"
                pil = getattr(result, "image", result)
                if isinstance(pil, Image.Image):
                    pil.save(out_path)
                else:
                    result.save(str(out_path))
        finally:
            _state["phase"] = "idle"
            _state["progress"] = None

    data = base64.b64encode(out_path.read_bytes()).decode()
    return {
        "image": "data:image/png;base64," + data,
        "seed": seed,
        "seconds": round(time.time() - started, 1),
        "file": out_path.name,
        "width": width,
        "height": height,
    }


def explain(error):
    text = f"{type(error).__name__}: {error}"
    low = text.lower()
    if any(k in low for k in ("gated", "401", "403", "access to model", "restricted")):
        text += (
            "\n\nThis model needs a Hugging Face login. Accept the terms on the model's "
            "Hugging Face page, then run 'uvx --from huggingface_hub hf auth login' in "
            "Terminal and try again. FLUX.2 klein 4B needs no login."
        )
    return text


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(200, {
                "phase": _state["phase"],
                "progress": _state["progress"],
                "download": _state["download"],
                "loaded": _state["model_name"],
                "models": list(MODELS),
                "quantize": QUANTIZE,
            })
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/render":
            return self._json(404, {"error": "Not found"})
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json(415, {"error": "Send JSON."})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self._json(413, {"error": "The images are too large to send."})
        try:
            payload = json.loads(self.rfile.read(length))
            self._json(200, render(payload))
        except Exception as error:  # report every failure to the page
            import traceback
            traceback.print_exc()
            self._json(500, {"error": explain(error)})

    def log_message(self, fmt, *args):
        if "/api/status" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Fabric Fitting Room is running at {url}  (Ctrl+C to stop)", flush=True)
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
