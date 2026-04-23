# Tool Silhouette → CNC SVG

Convert a top-down backlit photo of tools laid on an 18" × 12" grid into a scaled SVG of object outlines, ready for CNC foam cutting.

## Features

- **Streamlit UI** — sliders for live tuning, no rebuild needed.
- **Auto grid-border detection** — finds the 4 corners of the backlit panel and perspective-warps to true 18" × 12".
- **Otsu threshold** — robust segmentation for backlit subjects.
- **Raw pixel contours** — every contour point preserved in the SVG (no simplification).
- **Real-world scaled SVG** — output `viewBox` is in millimeters so CAM software imports at true size.
- **Original-image overlay toggle** — checkbox flips between photo-overlay preview and clean mask.

## Install

```bash
git clone https://github.com/<you>/tool-silhouette-svg.git
cd tool-silhouette-svg
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Workflow

1. Upload top-down photo of tools on the backlit grid.
2. Tune sliders in the sidebar:
   - **Blur radius** — smooths noise before threshold (odd values).
   - **Threshold offset** — shift Otsu-chosen threshold up/down.
   - **Morph close** — fills small holes inside objects.
   - **Min area (px²)** — drops tiny speckle contours.
   - **Edge margin (px)** — drops contours touching the warped frame border.
   - **Backlit** — invert if objects are brighter than background.
3. Click **Download SVG**. The file has a `width`/`height` in mm so CNC CAM imports at real size.

## Project layout

```
app.py          Streamlit UI
segmenter.py    OpenCV pipeline (border detect, warp, threshold, contours)
svg_writer.py   Scale contours to mm and emit SVG
requirements.txt
```

## Grid size

Defaults to 18" × 12" (457.2 × 304.8 mm). Change in the sidebar for other grids.

## License

MIT
