# Fabric Fitting Room

**See a piece of furniture in a new fabric before you reupholster it.**

Give it a photo of a sofa, chair or ottoman and one or more fabric photos, and it
renders the piece in each fabric. The image model (FLUX.2 klein, run through
[mflux](https://github.com/filipstrand/mflux)) runs entirely on your Mac.
Nothing is uploaded.

- **Batch renders:** one render per fabric sample, queued and shown side by side.
- **Close-up view:** click a render to see it larger, then zoom in and pan to
  check the weave, pattern scale and seams.
- **Progress as it works:** each render shows its steps, and the first run shows
  the model download.
- **Saved automatically:** every render is also written to the `renders` folder.

## Requirements

You need an Apple Silicon Mac (M1 or newer) on macOS 14 (Sonoma) or newer. Intel
Macs can't run it, because MLX only runs on Apple Silicon. There are two models
to choose from in the app:

| | klein 4B (lighter) | klein 9B (default, best quality) |
|---|---|---|
| **Memory: minimum** | 16 GB | 32 GB |
| **Memory: recommended** | 24 GB or more | 48 GB or more |
| **Peak memory while rendering** | about 11–13 GB (estimated) | 24–27 GB (measured) |
| **Download / disk space** | 16 GB | 35 GB |
| **Render time at 1024 px** | roughly half of 9B (estimated) | about 2 min on an M4 Pro |
| **Hugging Face login** | not needed | needed (gated model) |
| **Licence** | Apache 2.0 | FLUX non-commercial |

- The 9B figures were measured on an M4 Pro with 48 GB, at 1024 px, 4 steps and
  8-bit weights (4-bit peaks at 24 GB).
- The 4B figures are estimated by scaling the 9B results by model size.
- On a Mac at the minimum, close other apps while rendering, or start the app
  with `QUANTIZE=4` (see [Settings](#settings)) to save memory. macOS falls back
  to swap when memory runs out, which makes renders much slower.
- Render time grows with output size and detail passes, and with fewer GPU cores.

## Setup

1. **Install uv.** It fetches Python and mflux for you.

   ```sh
   brew install uv
   ```

2. **For klein 9B only, log in to Hugging Face.**
   1. Sign in at [huggingface.co](https://huggingface.co).
   2. Accept the terms on the
      [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
      model page.
   3. Log in from Terminal:

      ```sh
      uvx --from huggingface_hub hf auth login
      ```

   To skip this, pick **klein 4B (lighter)** in the app.

## Run

```sh
cd ~/Documents/furniture-fabric-visualizer
bash run.sh
```

The app opens at <http://127.0.0.1:8765>. Press **Ctrl+C** in Terminal to stop it.

The first render downloads the model, and the render card shows how much is left.
After that the model loads from disk. It stays in memory between renders while
the app is running.

## Using it

1. **Furniture:** one clear, well-lit photo.
2. **Fabrics:** flat, evenly lit close-ups. Each one produces its own render.
3. **Render**, then use the results:
   - **Click a render** to see it larger. Pinch or Cmd-scroll to zoom in, and
     scroll or drag to move around. The **−**, **100%** and **+** buttons and
     Cmd `=` / `-` / `0` work too.
   - **Try again** re-renders with a different random variation.
   - **Save** downloads the image. Every render is also saved in the `renders`
     folder.

## Settings

Set these when starting the app:

| Setting | Effect |
|---|---|
| `QUANTIZE=4 bash run.sh` | Uses less memory. The default is `8`; `0` is full precision. |
| `PORT=9000 bash run.sh` | Changes the port. |

The instruction sent to the model is `BASE_PROMPT` at the top of `server.py`.

## Troubleshooting

- **"This model needs a Hugging Face login":** accept the terms on the model page
  and run the login command in [Setup](#setup), or switch to klein 4B.
- **"The app's engine is not running":** start it with `bash run.sh` and keep
  Terminal open.
- **Renders are very slow:** the Mac is probably short of memory. Use klein 4B or
  `QUANTIZE=4`, or close other apps.
