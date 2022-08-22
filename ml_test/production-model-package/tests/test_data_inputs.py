#testing module to check the input data 
from regression_model import pipeline
from regression_model.config.core import config
from regression_model.processing.validation import validate_inputs


def test_validate_inputs(sample_input_data):
    # checking the expected shape of the data for scoring 
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)
    assert not errors
    assert len(sample_input_data) == 1738
    assert len(validated_inputs) == 1738

def test_pipeline_predict_takes_validated_input(pipeline_inputs, sample_input_data):
    # check whether the data which is input to the model is valid or not
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline.price_pipe.fit(X_train, y_train)
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)
    predictions = pipeline.price_pipe.predict(
        validated_inputs[config.model_config.features]
    )
    assert predictions is not None
    assert errors is None

def test_validate_inputs_identifies_errors(sample_input_data):
    # check whether the features have correct datatypes or not
    test_inputs = sample_input_data.copy()
    test_inputs.at[1, "holiday"] = 1  # we dont expect a string
    validated_inputs, errors = validate_inputs(input_data=test_inputs)
    assert errors
    assert errors[1] == {"holiday": ["didnt expected string"]}