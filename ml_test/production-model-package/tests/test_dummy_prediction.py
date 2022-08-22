#dummy to test to check if the vanila prediction of the model 
import numpy as np
from regression_model.predict import make_prediction
import math

def test_make_prediction(sample_input_data):
    ''''check the predictions of the model. 
        we expect the model to predict around 300 value with tolerance for +- 50 
    '''
    expected_first_prediction_value = 300 
    expected_no_predictions = 1738
    result = make_prediction(input_data=sample_input_data)
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=50)
