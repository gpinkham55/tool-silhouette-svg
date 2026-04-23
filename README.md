# Tool Silhouette → CNC SVG

Convert a top-down backlit photo of tools laid on a 17" × 12" grid into a scaled SVG of object outlines, ready for CNC foam cutting.

**Two flavors in this repo:**

1. **Web app (`docs/`)** — runs entirely in the browser via OpenCV.js. Host for free on GitHub Pages. Photos never leave the user's device.
2. **Python app (`app.py` + `segmenter.py`)** — Streamlit desktop/local version with the same pipeline.

## Web app

### Try it

Once deployed: `https://<your-username>.github.io/tool-silhouette-svg/`

### Deploy to GitHub Pages

1. Push this repo to GitHub.
2. Repo → **Settings** → **Pages**.
3. **Source:** Deploy from branch. **Branch:** `main`. **Folder:** `/docs`. Save.
4. Wait ~1 min. Your URL appears at the top of the Pages panel.

No build step. The `/docs` folder contains plain HTML/CSS/JS plus an empty `.nojekyll` file so GitHub serves files as-is.

### Run locally

```bash
cd docs
python -m http.server 5173
# open http://localhost:5173
```

## Python app (alternative)

### Install

```bash
git clone https://github.com/<you>/tool-silhouette-svg.git
cd tool-silhouette-svg
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Workflow (both versions)

1. Upload a top-down photo of tools on the backlit grid.
2. Tune sliders:
   - **Blur radius** — smooths noise before threshold.
   - **Threshold offset** — shifts Otsu-chosen threshold up/down.
   - **Morph open** — kills thin grid lines and ruler ink. Raise until only tools remain.
   - **Morph close** — fills holes inside tool silhouettes.
   - **Min area (px²)** — drops tiny speckle contours.
   - **Edge margin (px)** — drops contours touching the warped frame border (ruler markings, shadows).
   - **Backlit** — invert if objects are brighter than the background.
   - **Auto-orient** — swaps warp dimensions when the detected grid quad is portrait.
3. Toggle **Show original photo overlay** to compare mask vs. photo.
4. Click **Download SVG**. The file's `width`/`height` are in millimeters so CAM software imports at true size.

## Pipeline

```
photo → gray → blur → Otsu-threshold → largest 4-gon = grid border
     → perspective-warp to known mm dimensions
     → re-threshold → morph open (strip grid lines) → morph close (fill holes)
     → external contours → filter by area + edge margin
     → scale pixel coords to mm → emit <svg>
```

## Project layout

```
docs/              Web app (GitHub Pages)
  index.html
  app.js           OpenCV.js pipeline
  styles.css
  .nojekyll

app.py             Streamlit UI (Python alt)
segmenter.py       OpenCV pipeline
svg_writer.py      SVG emitter
requirements.txt
```

## Grid size

Defaults to 17" × 12" (431.8 × 304.8 mm). Change via the sidebar inputs for any grid.

## License

MIT
