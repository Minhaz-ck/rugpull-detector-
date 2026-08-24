import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["liquidity_usd","fdv","market_cap","price_change_24h",
            "buys_24h","sells_24h","volume_24h","holder_count","is_verified","pair_age_days"]

def build_cleaning_pipeline():
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", num_pipe, FEATURES)]), FEATURES
