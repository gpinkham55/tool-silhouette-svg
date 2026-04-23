"""Write contours as raw pixel polygons scaled to real-world mm for CNC."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

MM_PER_INCH = 25.4


def contours_to_svg(
    contours: list[np.ndarray],
    warp_w_px: int,
    warp_h_px: int,
    grid_w_mm: float,
    grid_h_mm: float,
) -> str:
    """Build SVG string. Border rect = grid outline. Polygons = raw pixel contours scaled to mm."""
    sx = grid_w_mm / warp_w_px
    sy = grid_h_mm / warp_h_px

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{grid_w_mm}mm",
            "height": f"{grid_h_mm}mm",
            "viewBox": f"0 0 {grid_w_mm} {grid_h_mm}",
        },
    )

    ET.SubElement(
        svg,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": f"{grid_w_mm}",
            "height": f"{grid_h_mm}",
            "fill": "none",
            "stroke": "black",
            "stroke-width": "0.5",
        },
    )

    g = ET.SubElement(svg, "g", {"fill": "none", "stroke": "black", "stroke-width": "0.3"})
    for c in contours:
        pts = c.reshape(-1, 2)
        coords = " ".join(f"{x * sx:.3f},{y * sy:.3f}" for x, y in pts)
        ET.SubElement(g, "polygon", {"points": coords})

    ET.indent(svg, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(svg, encoding="unicode")
