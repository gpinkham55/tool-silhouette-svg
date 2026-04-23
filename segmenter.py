"""Segmentation pipeline: backlit photo -> grid-warped binary -> object contours."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentParams:
    grid_w_mm: float = 431.8   # 17"
    grid_h_mm: float = 304.8   # 12"
    px_per_mm: float = 4.0     # warp resolution (1828 x 1219 at 4 px/mm)
    blur_radius: int = 5       # gaussian kernel (odd)
    min_area_px: int = 500     # drop tiny blobs
    edge_margin_px: int = 10   # drop blobs touching border
    invert: bool = True        # True = backlit (objects darker than background)
    threshold_offset: int = 0  # added to Otsu threshold
    morph_open: int = 7        # kernel size for opening (kills grid lines + speckle). 0 = off.
    morph_close: int = 9       # kernel size for closing (fills object holes). 0 = off.
    auto_orient: bool = True   # swap warp dims if detected grid quad is portrait


@dataclass
class SegmentResult:
    warped_bgr: np.ndarray          # perspective-corrected photo
    binary: np.ndarray              # thresholded mask
    contours: list[np.ndarray]      # kept external contours (pixel coords in warped frame)
    border_quad: np.ndarray | None  # 4x2 src corners of grid in original image (or None)


def _odd(n: int) -> int:
    return max(1, n | 1)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_grid_border(gray: np.ndarray) -> np.ndarray | None:
    """Find the outer grid as a 4-corner quad. Returns ordered 4x2 or None."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Backlit border -> bright rectangle. Try Otsu both ways and pick best.
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.2 * img_area:   # border should dominate frame
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None
    return _order_corners(best)


def warp_to_grid(bgr: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h))


def segment(bgr: np.ndarray, p: SegmentParams) -> SegmentResult:
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = detect_grid_border(gray_full)

    out_w = int(round(p.grid_w_mm * p.px_per_mm))
    out_h = int(round(p.grid_h_mm * p.px_per_mm))

    if quad is not None:
        if p.auto_orient:
            # Measure quad aspect. If portrait but target is landscape (or vice versa), swap.
            w_top = np.linalg.norm(quad[1] - quad[0])
            h_left = np.linalg.norm(quad[3] - quad[0])
            quad_portrait = h_left > w_top
            target_portrait = out_h > out_w
            if quad_portrait != target_portrait:
                out_w, out_h = out_h, out_w
        warped = warp_to_grid(bgr, quad, out_w, out_h)
    else:
        warped = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    k = _odd(p.blur_radius)
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_val = int(np.clip(otsu_val + p.threshold_offset, 0, 255))

    mode = cv2.THRESH_BINARY_INV if p.invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blur, thr_val, 255, mode)

    # Open first: kills thin grid lines and ruler ink (thinner than kernel).
    if p.morph_open and p.morph_open > 0:
        ks = _odd(p.morph_open)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Close second: fills holes inside surviving objects.
    if p.morph_close and p.morph_close > 0:
        ks = _odd(p.morph_close)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    kept: list[np.ndarray] = []
    m = p.edge_margin_px
    h, w = binary.shape
    for c in contours:
        if cv2.contourArea(c) < p.min_area_px:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= m or y <= m or (x + cw) >= (w - m) or (y + ch) >= (h - m):
            continue
        kept.append(c)

    return SegmentResult(warped_bgr=warped, binary=binary, contours=kept, border_quad=quad)
