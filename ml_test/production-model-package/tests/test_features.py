from regression_model.config.core import config
#from regression_model.processing.features import TemporalVariableTransformer


def test_check_feats(sample_input_data):
    assert sample_input_data.shape[0] == 1738
    assert sample_input_data.shape[1] == 18

def test_feat_drop(sample_input_data):
    drop_features = config.model_config.ref_var
    assert drop_features not in 
