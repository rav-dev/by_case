#this module is to test the model prediction performance 
from regression_model.predict import make_prediction
from regression_model.config.core import config
from regression_model.predict import make_prediction as alt_make_prediction

def test_prediction_quality_against_benchmark(raw_training_data, sample_input_data):
    '''checks the model prediction values for the desired benchmarks'''
    input_df = raw_training_data.drop(config.model_config.target, axis=1)
    output_df = raw_training_data[config.model_config.target]
    # Generate rough benchmarks (you would tweak depending on your model)
    benchmark_flexibility = 500
    #lower benchmark
    benchmark_lower_boundary = (
        output_df.iloc[0] - benchmark_flexibility
    ) 
    #upper benchmark
    benchmark_upper_boundary = (
        output_df.iloc[0] + benchmark_flexibility
    )
    subject = make_prediction(input_data=input_df[0:1])
    assert subject is not None
    prediction = subject.get("predictions")[0]
    assert isinstance(prediction, float)
    assert prediction > benchmark_lower_boundary
    assert prediction < benchmark_upper_boundary



    '''caveats'''
    #use different aspect of test data to check boundary cases here we are just using the training data for demonstration purpose
    #use tight benchmarks 
    #test model performance against another model but only one model is asked to be demonstrated in the business case