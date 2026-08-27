"""
Tabular / Classic ML MCP Server
Load CSVs, train sklearn models, predict, feature importance, cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "tabular-ml",
    "Classic machine learning on tabular data (CSV). "
    "Train classifiers/regressors, predict, inspect feature importance, cross-validate.",
)

_data: Dict[str, Any] = {}
_models: Dict[str, Any] = {}


@mcp.tool()
def load_csv(path: str, name: str = "default") -> str:
    """Load a CSV file into memory under the given name."""
    import pandas as pd
    df = pd.read_csv(path)
    _data[name] = df
    return safe_json({
        "status": "loaded",
        "name": name,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    })


@mcp.tool()
def describe_data(name: str = "default") -> str:
    """Return basic statistics and missing-value summary for a loaded dataset."""
    if name not in _data:
        return safe_json({"error": f"Dataset '{name}' not loaded"})
    df = _data[name]
    desc = df.describe(include="all").fillna("").to_dict()
    missing = df.isnull().sum().to_dict()
    return safe_json({"describe": desc, "missing": missing, "shape": list(df.shape)})


@mcp.tool()
def train_classifier(
    target: str,
    features: Optional[List[str]] = None,
    name: str = "default",
    model_name: str = "clf",
    algorithm: str = "random_forest",
) -> str:
    """Train a classifier. algorithm: random_forest | logistic | gradient_boosting."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    if name not in _data:
        return safe_json({"error": f"Dataset '{name}' not loaded"})
    df = _data[name]
    feats = features or [c for c in df.columns if c != target]
    X = df[feats].select_dtypes(include="number").fillna(0)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    algos = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic": LogisticRegression(max_iter=1000),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    model = algos.get(algorithm, algos["random_forest"])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    _models[model_name] = {"model": model, "features": list(X.columns), "task": "classification"}
    return safe_json({
        "status": "trained",
        "model_name": model_name,
        "algorithm": algorithm,
        "accuracy": acc,
        "n_features": len(X.columns),
        "report": classification_report(y_test, preds, output_dict=True),
    })


@mcp.tool()
def train_regressor(
    target: str,
    features: Optional[List[str]] = None,
    name: str = "default",
    model_name: str = "reg",
    algorithm: str = "random_forest",
) -> str:
    """Train a regressor. algorithm: random_forest | linear | gradient_boosting."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    if name not in _data:
        return safe_json({"error": f"Dataset '{name}' not loaded"})
    df = _data[name]
    feats = features or [c for c in df.columns if c != target]
    X = df[feats].select_dtypes(include="number").fillna(0)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    algos = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "linear": LinearRegression(),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }
    model = algos.get(algorithm, algos["random_forest"])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    _models[model_name] = {"model": model, "features": list(X.columns), "task": "regression"}
    return safe_json({
        "status": "trained",
        "model_name": model_name,
        "algorithm": algorithm,
        "rmse": rmse,
        "r2": r2,
        "n_features": len(X.columns),
    })


@mcp.tool()
def predict(model_name: str, records: List[Dict[str, float]]) -> str:
    """Run prediction on a list of feature dictionaries."""
    if model_name not in _models:
        return safe_json({"error": f"Model '{model_name}' not found"})
    entry = _models[model_name]
    import pandas as pd
    X = pd.DataFrame(records)[entry["features"]].fillna(0)
    preds = entry["model"].predict(X)
    return safe_json({"predictions": preds.tolist(), "model_name": model_name})


@mcp.tool()
def feature_importance(model_name: str) -> str:
    """Return feature importances if the model supports them."""
    if model_name not in _models:
        return safe_json({"error": f"Model '{model_name}' not found"})
    entry = _models[model_name]
    model = entry["model"]
    if not hasattr(model, "feature_importances_"):
        return safe_json({"error": "Model has no feature_importances_"})
    imp = dict(zip(entry["features"], model.feature_importances_.tolist()))
    sorted_imp = dict(sorted(imp.items(), key=lambda x: -x[1]))
    return safe_json({"feature_importance": sorted_imp})


@mcp.tool()
def cross_validate(
    target: str,
    features: Optional[List[str]] = None,
    name: str = "default",
    cv: int = 5,
    task: str = "classification",
) -> str:
    """Run k-fold cross-validation (accuracy for classification, r2 for regression)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import numpy as np

    if name not in _data:
        return safe_json({"error": f"Dataset '{name}' not loaded"})
    df = _data[name]
    feats = features or [c for c in df.columns if c != target]
    X = df[feats].select_dtypes(include="number").fillna(0)
    y = df[target]
    model = RandomForestClassifier(n_estimators=50, random_state=42) if task == "classification" else RandomForestRegressor(n_estimators=50, random_state=42)
    scoring = "accuracy" if task == "classification" else "r2"
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return safe_json({
        "scores": scores.tolist(),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scoring": scoring,
        "cv": cv,
    })


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
