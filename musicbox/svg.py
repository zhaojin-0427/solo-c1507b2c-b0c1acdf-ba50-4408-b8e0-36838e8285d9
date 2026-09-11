"""Printable SVG rendering of the unrolled cylinder.

The drawing uses millimetre units throughout: the root element carries
physical ``width``/``height`` in mm and a 1:1 viewBox, so printing at 100%
scale yields a true-size drilling template. Horizontal axis = circumference
(angle), vertical axis = cylinder axis (reed positions). Rendering is a pure
function of its inputs, hence deterministic.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from .engine import TxNote, circumference, pitch_name
from .models import ArrangementRequest, Metrics, PinOut, SolutionSpec

MARGIN_L = 24.0  # reed labels
MARGIN_R = 6.0
MARGIN_T = 14.0  # title block
MARGIN_B = 10.0  # angle ruler


def _fmt(v: float) -> str:
    s = f"{v:.3f}"
    return "0" if s in ("-0", "-0.000") else s


def render_unrolled_svg(
    *,
    content_hash: str,
    req: ArrangementRequest,
    sol: SolutionSpec,
    pins: list[PinOut],
    deleted_marks: list[TxNote],
    metrics: Metrics,
) -> str:
    circ = circumference(req)
    length = req.cylinder.effective_length_mm
    cons = req.constraints
    width = MARGIN_L + circ + MARGIN_R
    height = MARGIN_T + length + MARGIN_B

    def X(x_mm: float) -> float:
        return MARGIN_L + x_mm

    def Y(y_mm: float) -> float:
        return MARGIN_T + y_mm

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width)}mm" '
        f'height="{_fmt(height)}mm" viewBox="0 0 {_fmt(width)} {_fmt(height)}" '
        'font-family="monospace">'
    )
    out.append(f'<rect x="0" y="0" width="{_fmt(width)}" height="{_fmt(height)}" fill="#ffffff"/>')

    grid = f"{sol.quantize_grid_beats:g}" if sol.quantize_grid_beats else "off"
    title = (
        f"music-box cylinder | hash {content_hash[:12]} | "
        f"transpose {sol.transpose_semitones:+d} st | tempo x{sol.tempo_factor:g} | "
        f"grid {grid} | deleted {metrics.deleted_count} | "
        f"min clearance {metrics.min_clearance_mm} mm"
    )
    out.append(
        f'<text x="{_fmt(MARGIN_L)}" y="8" font-size="3.4" fill="#111">{escape(title)}</text>'
    )

    # seam forbidden zone (the seam unrolls to both edges of the rectangle)
    half_seam = cons.seam_zone_mm / 2.0
    if half_seam > 0:
        out.append(
            f'<rect x="{_fmt(X(0))}" y="{_fmt(Y(0))}" width="{_fmt(half_seam)}" '
            f'height="{_fmt(length)}" fill="#f8d7da"/>'
        )
        out.append(
            f'<rect x="{_fmt(X(circ - half_seam))}" y="{_fmt(Y(0))}" '
            f'width="{_fmt(half_seam)}" height="{_fmt(length)}" fill="#f8d7da"/>'
        )

    # cylinder outline
    out.append(
        f'<rect x="{_fmt(X(0))}" y="{_fmt(Y(0))}" width="{_fmt(circ)}" '
        f'height="{_fmt(length)}" fill="none" stroke="#333" stroke-width="0.3"/>'
    )

    # angle ruler along the top edge, one tick every 30 degrees
    for deg in range(0, 361, 30):
        x = X(circ * deg / 360.0)
        out.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(Y(0))}" x2="{_fmt(x)}" '
            f'y2="{_fmt(Y(-2.2))}" stroke="#888" stroke-width="0.15"/>'
        )
        out.append(
            f'<text x="{_fmt(x)}" y="{_fmt(Y(-3.2))}" font-size="2.2" fill="#555" '
            f'text-anchor="middle">{deg}</text>'
        )

    # reed lines with pitch labels
    for reed in sorted(req.comb, key=lambda r: r.axial_mm):
        y = Y(reed.axial_mm)
        label = f"{pitch_name(reed.pitch)} ({reed.pitch})"
        out.append(
            f'<line x1="{_fmt(X(0))}" y1="{_fmt(y)}" x2="{_fmt(X(circ))}" y2="{_fmt(y)}" '
            f'stroke="#ccc" stroke-width="0.15"/>'
        )
        out.append(
            f'<text x="1" y="{_fmt(y + 0.8)}" font-size="2.4" fill="#333">{escape(label)}</text>'
        )

    # deleted notes: red crosses at their (transformed) would-be positions
    r = cons.pin_diameter_mm / 2.0
    for m in deleted_marks:
        x, y = X(m.x_mm), Y(m.axial_mm)  # type: ignore[arg-type]
        out.append(
            f'<line x1="{_fmt(x - r)}" y1="{_fmt(y - r)}" x2="{_fmt(x + r)}" '
            f'y2="{_fmt(y + r)}" stroke="#dc3545" stroke-width="0.25"/>'
        )
        out.append(
            f'<line x1="{_fmt(x - r)}" y1="{_fmt(y + r)}" x2="{_fmt(x + r)}" '
            f'y2="{_fmt(y - r)}" stroke="#dc3545" stroke-width="0.25"/>'
        )

    # pins: blue = locked note, green = normal
    for p in pins:
        x, y = X(p.x_mm), Y(p.axial_mm)
        fill = "#0b5ed7" if p.locked else "#198754"
        tip = escape(
            f"{p.note_id} {p.pitch_name} beat {p.beat} angle {p.angle_deg} deg "
            f"axial {p.axial_mm} mm"
        )
        out.append(
            f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(r)}" fill="{fill}" '
            f'fill-opacity="0.8" stroke="#222" stroke-width="0.1">'
            f"<title>{tip}</title></circle>"
        )

    out.append("</svg>")
    return "\n".join(out)
