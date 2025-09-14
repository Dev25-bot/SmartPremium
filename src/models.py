
"""
Model definitions and helper functions.
"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # optional dependency; check requirements

def get_model_dict():
    """
    Return a dict of candidate models with short names.
    """
    models = {
        "linear": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1)
    return models
