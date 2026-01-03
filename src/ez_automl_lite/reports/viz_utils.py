"""Shared visualization utilities for report generation."""

from typing import Any, Literal


def generate_pca_scatter_plot(
    pca_data: list[dict[str, Any]],
    mode: Literal["cluster", "anomaly"],
    title: str = "PCA Projection (2D)",
) -> str:
    """
    Generate PCA scatter plot SVG.

    Args:
        pca_data: List of points with 'x', 'y', and mode-specific label
        mode: Visualization mode - 'cluster' for cluster data, 'anomaly' for anomaly detection
        title: Plot title/caption

    Returns:
        HTML string with SVG chart and legend
    """
    if not pca_data:
        return "<p>No visualization data available.</p>"

    # Extract and normalize coordinates
    xs = [p["x"] for p in pca_data]
    ys = [p["y"] for p in pca_data]

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

    return f"""
    <div class="chart-container">
        <svg class="chart-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <rect x="0" y="0" width="100" height="100" fill="#fafafa" rx="4" />
            <!-- Axes -->
            <line x1="{padding}" y1="{100-padding}" x2="{100-padding}" y2="{100-padding}" class="chart-axis" />
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{100-padding}" class="chart-axis" />
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
            f'class="scatter-pt cluster-{c_id % 8}" />'
        )

    legend_html = '<div class="legend">'
    for c_id in sorted(clusters):
        legend_html += (
            f'<div class="legend-item">'
            f'<div class="legend-dot cluster-{c_id % 8}"></div>Cluster {c_id}'
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
            <div class="legend-dot" style="background: #1a73e8; opacity:0.5;"></div>Normal
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #d93025;"></div>Anomaly
        </div>
    </div>
    """

    return points_svg, legend_html
