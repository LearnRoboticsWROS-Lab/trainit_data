"""Instance mask -> one polygon per instance, in pure numpy + integer Python (INTERNAL, M4.5).

The polygon policy of the YOLO segmentation export (``POLICY``, recorded in its sidecar). For each instance id of a
16-bit compact-id mask:

1. ``mask == id``, cropped to its extent;
2. its 8-connected components (a run-length union-find); the LARGEST is kept (most pixels; a tie goes to the one
   whose first pixel comes first in raster order). The other components are dropped and COUNTED by the caller;
3. the OUTER boundary of the kept component is traced (Moore-neighbour tracing, clockwise on screen, from its
   top-most then left-most pixel, the stopping rule "back at the start pixel and about to step to the second
   one"). Holes are ignored: a YOLO polygon has no holes, so a ring exports as its filled outline;
4. the closed boundary is simplified by Douglas-Peucker with tolerance ``epsilon_px``: anchored at the start pixel
   and the boundary pixel farthest from it (first one on a tie), each half simplified on its own. Every distance
   test is EXACT (integer cross products against the tolerance as a ``Fraction``), so the result never depends on
   floating-point rounding, the interpreter or the platform;
5. repeated consecutive vertices are merged. Fewer than 3 vertices, or a zero area, is a DEGENERATE instance (e.g. a
   one-pixel-wide line): it has no polygon.

Vertices are PIXEL CENTRES of boundary pixels, in the corner-origin continuous image frame of this repository's
boxes (``labels.py``: the top-left pixel CORNER is (0, 0)): the centre of pixel column ``c``, row ``r`` is
``(c + 0.5, r + 0.5)``. A rasterizer that fills the pixels whose integer index lies inside the polygon after
truncating the de-normalized vertices to integers (Ultralytics' ``polygon2mask``: ``astype(int32)`` +
``cv2.fillPoly``) gets back exactly the traced boundary pixels, even after the 6-decimal normalization (the half
pixel absorbs the rounding).
"""
from bisect import bisect_left
from fractions import Fraction

import numpy as np

POLICY = {'name': 'largest_component_outer_boundary', 'version': 1, 'connectivity': 8, 'holes': 'ignored',
          'vertices': 'pixel_centre', 'simplification': 'douglas_peucker'}

# The 8 neighbour offsets (d_row, d_col), CLOCKWISE on screen (rows grow downwards) starting East.
_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}
_WEST = 4


def _runs(mask):
    """The horizontal runs of a 2-D bool array, in raster order: ``[(row, start, end_exclusive)]``."""
    height = mask.shape[0]
    padded = np.zeros((height, mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    rows, starts = np.nonzero(step == 1)
    _, ends = np.nonzero(step == -1)              # row-major order: the i-th end closes the i-th start
    return list(zip(rows.tolist(), starts.tolist(), ends.tolist()))


def largest_component(mask):
    """``(component, n_components, dropped_pixels)``: the largest 8-connected component of a 2-D bool array (a new
    bool array of the same shape), how many components there are, and how many pixels the others hold.
    ``component`` is None for an empty mask."""
    runs = _runs(mask)
    if not runs:
        return None, 0, 0
    parent = list(range(len(runs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_row = {}
    for index, (row, start, end) in enumerate(runs):
        by_row.setdefault(row, []).append(index)
    for row, indices in by_row.items():
        above = by_row.get(row - 1)
        if not above:
            continue
        above_ends = [runs[j][2] for j in above]
        for i in indices:
            _, start, end = runs[i]
            # 8-connected to a run [s, e) of the row above when s <= end and start <= e (diagonals included)
            k = bisect_left(above_ends, start)
            while k < len(above) and runs[above[k]][1] <= end:
                a, b = find(i), find(above[k])
                if a != b:
                    parent[max(a, b)] = min(a, b)             # the root is the component's first run (raster order)
                k += 1
    sizes = {}
    for index, (_, start, end) in enumerate(runs):
        root = find(index)
        sizes[root] = sizes.get(root, 0) + end - start
    keep = min(sizes, key=lambda root: (-sizes[root], root))  # most pixels, then first in raster order
    component = np.zeros(mask.shape, dtype=bool)
    for index, (row, start, end) in enumerate(runs):
        if find(index) == keep:
            component[row, start:end] = True
    total = sum(sizes.values())
    return component, len(sizes), total - sizes[keep]


def trace_outer_boundary(component):
    """The outer boundary of ONE 8-connected component (a 2-D bool array) as ``[(col, row)]`` boundary pixels,
    clockwise on screen from the top-most then left-most pixel. A pixel where the boundary pinches appears more
    than once. Holes are not traced."""
    rows, cols = np.nonzero(component)
    if rows.size == 0:
        return []
    padded = np.zeros((component.shape[0] + 2, component.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = component
    grid = padded.tolist()                                    # plain lists: fast scalar access
    start = (int(rows[0]) + 1, int(cols[0]) + 1)              # raster order: top-most, then left-most

    def step(pixel, came_from):
        """Scan the neighbours of ``pixel`` clockwise from the background neighbour ``came_from``: the first
        foreground one and the background neighbour just before it (None if ``pixel`` is isolated)."""
        r, c = pixel
        first = _DIR_INDEX[(came_from[0] - r, came_from[1] - c)]
        previous = came_from
        for k in range(1, 8):
            dr, dc = _DIRS[(first + k) % 8]
            nxt = (r + dr, c + dc)
            if grid[nxt[0]][nxt[1]]:
                return nxt, previous
            previous = nxt
        return None, None

    west = (start[0], start[1] - 1)
    second, back = step(start, west)
    if second is None:                                        # a single pixel
        return [(start[1] - 1, start[0] - 1)]
    boundary = [start]
    pixel, came_from = second, back
    limit = 4 * int(rows.size) + 8                            # a boundary visits a pixel at most 4 times
    while True:
        nxt, back = step(pixel, came_from)
        if pixel == start and nxt == second:
            break
        boundary.append(pixel)
        if len(boundary) > limit:                             # never: the tracing rule terminates
            raise RuntimeError('boundary tracing did not close')
        pixel, came_from = nxt, back
    return [(c - 1, r - 1) for r, c in boundary]


def _dp_chain(points, eps2: Fraction) -> list:
    """Douglas-Peucker of an OPEN chain of integer points (iterative; exact integer arithmetic): the kept indices."""
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = points[i]
        bx, by = points[j]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        best, best_at = -1, -1
        for t in range(i + 1, j):
            px, py = points[t]
            if length2 == 0:
                d = (px - ax) ** 2 + (py - ay) ** 2
            else:
                cross = dx * (py - ay) - dy * (px - ax)
                d = cross * cross                              # = distance^2 * length2
            if d > best:
                best, best_at = d, t
        if best > (eps2 if length2 == 0 else eps2 * length2):
            keep[best_at] = True
            stack.append((i, best_at))
            stack.append((best_at, j))
    return [p for p, k in zip(points, keep) if k]


def simplify_closed(points, epsilon_px: float) -> list:
    """Douglas-Peucker of a CLOSED polygon of integer points (see the module docstring), consecutive duplicates
    merged. Returns the vertices (possibly fewer than 3)."""
    if len(points) < 3:
        return list(points)
    eps2 = Fraction(epsilon_px) ** 2
    x0, y0 = points[0]
    far = max(range(len(points)), key=lambda i: ((points[i][0] - x0) ** 2 + (points[i][1] - y0) ** 2, -i))
    if far == 0:
        return [points[0]]
    first = _dp_chain(points[:far + 1], eps2)
    second = _dp_chain(points[far:] + [points[0]], eps2)
    out = first[:-1] + second[:-1]
    merged = [p for i, p in enumerate(out) if p != out[i - 1]] if len(out) > 1 else out
    return merged


def doubled_area(points) -> int:
    """Twice the signed shoelace area of integer vertices (0 for a degenerate polygon)."""
    n = len(points)
    return sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))


def instance_polygon(mask, epsilon_px: float):
    """The policy on ONE instance's bool mask (full image size). Returns ``(vertices, info)``: ``vertices`` is
    ``[(col, row)]`` boundary pixel indices (the vertex is that pixel's centre) or None if the instance is
    empty or degenerate; ``info`` = ``{components, pixels, dropped_pixels}`` where ``pixels`` counts the kept
    component."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None, {'components': 0, 'pixels': 0, 'dropped_pixels': 0}
    cols = np.flatnonzero(mask.any(axis=0))
    r0, c0 = int(rows[0]), int(cols[0])
    crop = mask[r0:int(rows[-1]) + 1, c0:int(cols[-1]) + 1]
    component, count, dropped = largest_component(crop)
    pixels = int(np.count_nonzero(component))
    info = {'components': count, 'pixels': pixels, 'dropped_pixels': dropped}
    vertices = simplify_closed(trace_outer_boundary(component), epsilon_px)
    if len(vertices) < 3 or doubled_area(vertices) == 0:
        return None, info
    return [(c + c0, r + r0) for c, r in vertices], info


__all__ = ['POLICY', 'largest_component', 'trace_outer_boundary', 'simplify_closed', 'doubled_area',
           'instance_polygon']
