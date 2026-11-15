# Fabric Fitting Room

A local web app: give it a photo of a piece of furniture and one or more fabric
photos, and it renders the furniture in each fabric. The image model
(FLUX.2 klein, run through mflux) runs on this Mac. Nothing is uploaded.

## One-time setup

1. Install uv (it fetches Python and mflux for you):

       brew install uv

2. For the default model, klein 9B (best quality, non-commercial licence, about
   32 GB download):
   - Sign in at huggingface.co and accept the terms on the
     `black-forest-labs/FLUX.2-klein-9B` model page.
   - Log in from Terminal:

         uvx --from huggingface_hub hf auth login

   To skip this, pick "klein 4B (lighter)" in the app. It needs no login and is
   about a 15 GB download.

## Run

    cd ~/Documents/furniture-fabric-visualizer
    bash run.sh

The app opens at http://127.0.0.1:8765. Press Ctrl+C in Terminal to stop it.

The first render downloads the model, which takes a while; progress shows in
Terminal. After that the model loads from disk, and it stays in memory between
renders while the app is running.

## Using it

- Furniture: one clear, well-lit photo.
- Fabrics: flat, evenly lit close-ups. Each one produces its own render.
- "Try again" re-renders with a different random variation.
- Every render is also saved in the `renders` folder.

## Settings

- `QUANTIZE=4 bash run.sh` uses less memory (default is 8; 0 is full precision).
- `PORT=9000 bash run.sh` changes the port.
- The instruction sent to the model is `BASE_PROMPT` at the top of `server.py`.
