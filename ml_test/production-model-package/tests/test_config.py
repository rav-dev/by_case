#this module is for the conf.yml testing 
from pathlib import Path

import pytest
from pydantic import ValidationError

from regression_model.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml,
)
#correct yml structure 
TEST_CONFIG_TEXT = """
package_name: regression_model
data_file: hour.csv
test_data_file : hour_fraction.csv
ref_var:
  - instant
drop_cols:
  - instant
pipeline_name: regression_model
pipeline_save_file: regression_model_output_v
target: cnt
test_size: 0.2
features:
  - instant
numericals_robust_scale:
  - temp
categorical_vars:
  - season
random_state: 0
max_depth: 20
min_samples_split: 4
n_estimators: 20
"""
#incorrect yml structure
INVALID_TEST_CONFIG_TEXT = """
package_name: regression_model
data_file: hour.csv
test_data_file : hour_fraction.csv
ref_var:
  - instant
drop_cols:
  - instant
pipeline_name: regression_model
pipeline_save_file: regression_model_output_v
target: cnt
test_size: 0.2
features:
  - instant
numericals_robust_scale:
  - temp
categorical_vars:
  - season
random_state: 0
max_depth: 20
min_samples_split: 4
n_estimators: 20
"""


def test_fetch_config_structure(tmpdir):
    # tests weather the yml structure is correct or not
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)
    config = create_and_validate_config(parsed_config=parsed_config)
    assert config.model_config
    assert config.app_config


# def test_config_validation_raises_error_for_invalid_config(tmpdir):
#    # Given
#    # We make use of the pytest built-in tmpdir fixture
#    configs_dir = Path(tmpdir)
#    config_1 = configs_dir / "sample_config.yml"
#
#    # invalid config attempts to set a prohibited loss
#    # function which we validate against an allowed set of
#    # loss function parameters.
#    config_1.write_text(INVALID_TEST_CONFIG_TEXT)
#    parsed_config = fetch_config_from_yaml(cfg_path=config_1)
#
#    # When
#    with pytest.raises(ValidationError) as excinfo:
#        create_and_validate_config(parsed_config=parsed_config)

#    # Then
#    assert "not in the allowed " in str(excinfo.value)


def test_missing_config_field_raises_validation_error(tmpdir):
    # checks whether there is any missing entity in the config.yml
    #this is test is maid to fail for demonstration
    configs_dir = Path(tmpdir)
    config_1 = configs_dir / "sample_config.yml"
    # TEST_CONFIG_TEXT = """package_name: regression_model"""
    config_1.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(excinfo.value)
    assert "pipeline_name" in str(excinfo.value)
