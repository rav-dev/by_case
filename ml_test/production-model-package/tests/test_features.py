# from regression_model import __version__ as _version
from regression_model import pipeline
from regression_model.config.core import config
from regression_model.processing.data_manager import feature_droper, load_pipeline


def test_check_feats(sample_input_data):
    assert sample_input_data.shape[0] == 1738
    assert sample_input_data.shape[1] == 17


def test_uname_presence(sample_input_data):
    res = sample_input_data.columns.str.match("Unnamed")
    assert False in res


# def test_feat_drop(sample_input_data):
#     data = feature_droper(sample_input_data)
#     features_to_drop = config.model_config.ref_var
#     res  =  all(item in features_to_drop  for item in data.columns)
#     assert res == False


def test_pipeline_drops_unnecessary_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    drop_cols = config.model_config.drop_cols
    res_X_train = all(item in drop_cols for item in X_train.columns)
    assert res_X_train == False
    pipeline.price_pipe.fit(X_train, y_train)
    transformed_inputs = pipeline.price_pipe[:-1].transform(X_train)
    res_X_train = all(item in drop_cols for item in X_train.columns)
    assert res_X_train == False
