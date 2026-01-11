"""Shared visualization utilities for report generation."""

from typing import Any, Literal


MAX_CLUSTER_COLORS = 8


def generate_pca_scatter_plot(
    pca_data: list[dict[str, Any]],
    mode: Literal["cluster", "anomaly"],
    title: str = "PCA Projection (2D)",
    explained_variance: tuple[float, float] | None = None,
) -> str:
    """
    Generate PCA scatter plot SVG with axis labels and ticks.

    Args:
        pca_data: List of points with 'x', 'y', and mode-specific label
        mode: Visualization mode - 'cluster' for cluster data, 'anomaly' for anomaly detection
        title: Plot title/caption
        explained_variance: Tuple of (PC1 %, PC2 %) explained variance ratios

    Returns:
        HTML string with SVG chart and legend
    """
    if not pca_data:
        return "<p>No visualization data available.</p>"

    # Extract and normalize coordinates
    try:
        xs = [p["x"] for p in pca_data]
        ys = [p["y"] for p in pca_data]
    except KeyError as e:
        return f"<p>Invalid data structure: missing key {e}</p>"

    if not xs or not ys:
        return ""

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1

    # Chart dimensions with padding
    padding = 10
    width = 100 - 2 * padding
    height = 100 - 2 * padding

    # Generate points and legend based on mode
    if mode == "cluster":
        points_svg, legend_html = _generate_cluster_points(
            pca_data, min_x, min_y, range_x, range_y, padding, width, height
        )
    else:  # anomaly
        points_svg, legend_html = _generate_anomaly_points(
            pca_data, min_x, min_y, range_x, range_y, padding, width, height
        )

    # Generate axis tick labels with actual data range
    x_ticks = ""
    y_ticks = ""
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]

    for pos in tick_positions:
        # X-axis ticks
        x_val = min_x + (pos * range_x)
        x_coord = padding + (pos * width)
        x_ticks += f'<line x1="{x_coord}" y1="{100-padding}" x2="{x_coord}" y2="{100-padding+2}" stroke="#666" stroke-width="0.5"/>'
        x_ticks += f'<text x="{x_coord}" y="{100-padding+5}" text-anchor="middle" font-size="3" fill="#666">{x_val:.2f}</text>'

        # Y-axis ticks
        y_val = min_y + (pos * range_y)
        y_coord = 100 - (padding + (pos * height))
        y_ticks += f'<line x1="{padding-2}" y1="{y_coord}" x2="{padding}" y2="{y_coord}" stroke="#666" stroke-width="0.5"/>'
        y_ticks += f'<text x="{padding-4}" y="{y_coord+1}" text-anchor="end" font-size="3" fill="#666">{y_val:.2f}</text>'

    # Add axis labels with explained variance
    pc1_label = f"PC1 ({explained_variance[0]:.1f}%)" if explained_variance else "PC1"
    pc2_label = f"PC2 ({explained_variance[1]:.1f}%)" if explained_variance else "PC2"

    axis_labels = f"""
        <text x="50" y="98" text-anchor="middle" font-size="4" fill="#333" font-weight="600">{pc1_label}</text>
        <text x="2" y="50" text-anchor="middle" font-size="4" fill="#333" font-weight="600" transform="rotate(-90, 2, 50)">{pc2_label}</text>
    """

    return f"""
    <div class="chart-container">
        <svg class="chart-svg" viewBox="-5 -5 110 110" preserveAspectRatio="xMidYMid meet">
            <rect x="0" y="0" width="100" height="100" fill="#fafafa" rx="4" />
            <!-- Axes -->
            <line x1="{padding}" y1="{100-padding}" x2="{100-padding}" y2="{100-padding}" class="chart-axis" />
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{100-padding}" class="chart-axis" />
            {x_ticks}
            {y_ticks}
            {axis_labels}
            {points_svg}
        </svg>
        <div style="text-align:center; font-size:0.8em; color:#888; margin-top:5px;">{title}</div>
    </div>
    {legend_html}
    """


def _generate_cluster_points(
    pca_data: list[dict],
    min_x: float,
    min_y: float,
    range_x: float,
    range_y: float,
    padding: float,
    width: float,
    height: float,
) -> tuple[str, str]:
    """Generate SVG points and legend for cluster visualization."""
    points_svg = ""
    clusters = set()

    for p in pca_data:
        cx = padding + ((p["x"] - min_x) / range_x) * width
        cy = 100 - (padding + ((p["y"] - min_y) / range_y) * height)  # Flip Y
        c_id = p["cluster"]
        clusters.add(c_id)
        points_svg += (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
            f'class="scatter-pt cluster-{c_id % MAX_CLUSTER_COLORS}" />'
        )

    legend_html = '<div class="legend">'
    for c_id in sorted(clusters):
        legend_html += (
            f'<div class="legend-item">'
            f'<div class="legend-dot cluster-{c_id % MAX_CLUSTER_COLORS}"></div>Cluster {c_id}'
            f"</div>"
        )
    legend_html += "</div>"

    return points_svg, legend_html


def _generate_anomaly_points(
    pca_data: list[dict],
    min_x: float,
    min_y: float,
    range_x: float,
    range_y: float,
    padding: float,
    width: float,
    height: float,
) -> tuple[str, str]:
    """Generate SVG points and legend for anomaly visualization."""
    # Draw normal points first, then anomalies on top
    normal_pts = [p for p in pca_data if p["label"] == 1]
    anomaly_pts = [p for p in pca_data if p["label"] == -1]

    points_svg = ""
    for p in normal_pts + anomaly_pts:
        cx = padding + ((p["x"] - min_x) / range_x) * width
        cy = 100 - (padding + ((p["y"] - min_y) / range_y) * height)
        cls = "pt-normal" if p["label"] == 1 else "pt-anomaly"
        points_svg += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" class="scatter-pt {cls}" />'

    legend_html = """
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background: #1a73e8; opacity:0.3;"></div>Normal
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #d93025;"></div>Anomaly
        </div>
    </div>
    """

    return points_svg, legend_html
