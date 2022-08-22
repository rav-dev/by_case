from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from regression_model.config.core import config

price_pipe = Pipeline(
    [
        # feature drop
        ("drop_features", DropFeatures(features_to_drop=config.model_config.ref_var)),
        # one hot encoding
        (
            "categorical_encoder",
            OneHotEncoder(
                # encoding_method="ordered",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # feature scaling
        ("scaler", RobustScaler()),
        # random forest regrssion training
        (
            "random forest regressor",
            RandomForestRegressor(
                max_depth=config.model_config.max_depth,
                min_samples_split=config.model_config.min_samples_split,
                n_estimators=config.model_config.n_estimators,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
