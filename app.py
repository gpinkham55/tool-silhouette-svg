"""Streamlit UI for tool-silhouette SVG generation."""
from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from segmenter import SegmentParams, segment
from svg_writer import contours_to_svg

st.set_page_config(page_title="Tool Silhouette SVG", layout="wide")
st.title("Tool Silhouette → CNC SVG")
st.caption("Top-down backlit photo on 18\" × 12\" grid → scaled SVG outlines for foam cutting.")

with st.sidebar:
    st.header("Grid")
    w_in = st.number_input("Grid width (in)", value=18.0, step=0.5)
    h_in = st.number_input("Grid height (in)", value=12.0, step=0.5)
    px_per_mm = st.slider("Warp resolution (px/mm)", 2.0, 8.0, 4.0, 0.5)

    st.header("Segmentation")
    blur_radius = st.slider("Blur radius", 1, 31, 5, 2)
    threshold_offset = st.slider("Threshold offset (± Otsu)", -60, 60, 0, 1)
    morph_open = st.slider("Morph open (kills grid lines)", 0, 25, 7, 1)
    morph_close = st.slider("Morph close (fills holes)", 0, 25, 9, 1)
    invert = st.checkbox("Backlit (objects darker)", value=True)
    auto_orient = st.checkbox("Auto-orient warp (portrait/landscape)", value=True)

    st.header("Filtering")
    min_area_px = st.slider("Min area (px²)", 50, 20000, 500, 50)
    edge_margin_px = st.slider("Edge margin (px)", 0, 100, 10, 1)

    st.header("Preview")
    show_original = st.checkbox("Show original photo overlay", value=True)

uploaded = st.file_uploader("Upload top-down photo", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded is None:
    st.info("Upload a backlit photo of tools on the grid to begin.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

params = SegmentParams(
    grid_w_mm=w_in * 25.4,
    grid_h_mm=h_in * 25.4,
    px_per_mm=px_per_mm,
    blur_radius=blur_radius,
    min_area_px=min_area_px,
    edge_margin_px=edge_margin_px,
    invert=invert,
    threshold_offset=threshold_offset,
    morph_open=morph_open,
    morph_close=morph_close,
    auto_orient=auto_orient,
)

result = segment(bgr, params)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Warped + detected contours")
    overlay = result.warped_bgr.copy() if show_original else np.full_like(result.warped_bgr, 255)
    cv2.drawContours(overlay, result.contours, -1, (0, 0, 255), 2)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.caption(
        f"{len(result.contours)} object(s) kept. "
        f"{'Border auto-detected.' if result.border_quad is not None else 'Border NOT detected — full frame used.'}"
    )

with col2:
    st.subheader("Binary mask")
    st.image(result.binary, use_column_width=True, clamp=True)

svg_text = contours_to_svg(
    result.contours,
    result.warped_bgr.shape[1],
    result.warped_bgr.shape[0],
    params.grid_w_mm,
    params.grid_h_mm,
)

st.download_button(
    "Download SVG",
    data=svg_text.encode("utf-8"),
    file_name="tool_outlines.svg",
    mime="image/svg+xml",
    disabled=len(result.contours) == 0,
)

with st.expander("SVG preview (raw text)"):
    st.code(svg_text[:2000] + ("\n..." if len(svg_text) > 2000 else ""), language="xml")
