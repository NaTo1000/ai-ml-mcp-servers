"""
Time-Series MCP Server
Load series, rolling stats, naive forecast, anomaly detection, resampling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "timeseries",
    "Time-series utilities: load, rolling statistics, naive forecasts, anomaly detection, resampling.",
)

_series: Dict[str, Any] = {}


@mcp.tool()
def load_series(values: List[float], name: str = "default", timestamps: Optional[List[str]] = None) -> str:
    """Load a numeric time series (optionally with ISO timestamps)."""
    import numpy as np
    arr = np.asarray(values, dtype=float)
    _series[name] = {"values": arr, "timestamps": timestamps}
    return safe_json({"status": "loaded", "name": name, "length": len(arr), "mean": float(arr.mean()), "std": float(arr.std())})


@mcp.tool()
def rolling_stats(name: str = "default", window: int = 7) -> str:
    """Compute rolling mean and std."""
    import numpy as np
    if name not in _series:
        return safe_json({"error": f"Series '{name}' not loaded"})
    v = _series[name]["values"]
    if len(v) < window:
        return safe_json({"error": "Series shorter than window"})
    means, stds = [], []
    for i in range(window - 1, len(v)):
        w = v[i - window + 1 : i + 1]
        means.append(float(np.mean(w)))
        stds.append(float(np.std(w)))
    return safe_json({"rolling_mean": means, "rolling_std": stds, "window": window})


@mcp.tool()
def forecast_naive(name: str = "default", horizon: int = 5, method: str = "last") -> str:
    """Naive forecast: last | mean | drift."""
    import numpy as np
    if name not in _series:
        return safe_json({"error": f"Series '{name}' not loaded"})
    v = _series[name]["values"]
    if method == "mean":
        pred = [float(np.mean(v))] * horizon
    elif method == "drift":
        slope = (v[-1] - v[0]) / max(len(v) - 1, 1)
        pred = [float(v[-1] + slope * (i + 1)) for i in range(horizon)]
    else:  # last
        pred = [float(v[-1])] * horizon
    return safe_json({"forecast": pred, "method": method, "horizon": horizon})


@mcp.tool()
def detect_anomalies(name: str = "default", z_thresh: float = 3.0) -> str:
    """Flag points whose |z-score| exceeds threshold."""
    import numpy as np
    if name not in _series:
        return safe_json({"error": f"Series '{name}' not loaded"})
    v = _series[name]["values"]
    mu, sigma = float(np.mean(v)), float(np.std(v)) + 1e-9
    z = (v - mu) / sigma
    idxs = np.where(np.abs(z) > z_thresh)[0].tolist()
    return safe_json({
        "anomaly_indices": idxs,
        "anomaly_values": [float(v[i]) for i in idxs],
        "z_scores": [float(z[i]) for i in idxs],
        "threshold": z_thresh,
    })


@mcp.tool()
def resample(name: str = "default", factor: int = 2, method: str = "mean") -> str:
    """Downsample by factor using mean or median."""
    import numpy as np
    if name not in _series:
        return safe_json({"error": f"Series '{name}' not loaded"})
    v = _series[name]["values"]
    n = len(v) // factor
    if method == "median":
        out = [float(np.median(v[i * factor : (i + 1) * factor])) for i in range(n)]
    else:
        out = [float(np.mean(v[i * factor : (i + 1) * factor])) for i in range(n)]
    return safe_json({"resampled": out, "factor": factor, "method": method, "new_length": len(out)})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
