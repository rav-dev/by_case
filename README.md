# by_case - code in by_case/ml_test/
This is the by_case repo created by Ravish Aggarwal

- In this case I showed the transition from the research phase/POC phase in data science lifecycle to a MVP phase
- The research phase is jupyter notebook where I performed 
    - exploratory data analysis
    - Data preprocessing 
    - Prepared data for training 
    - Trained various regression model (having performance measurement index as R2 score, also calculated MAD)
    - Selected the best model based on the R2 score which is Random Forest Regressor
- Then transformed the code from the research phase to MVP phase (production)
    - write test cases - Unit tests 
        - For preprocessing and features used in the training 
        - For config.yml used 
        - For input data used in model training 
        - For model quality benchmark testing 
- used good practices from pydantic, pytest framework and tox for testing purpose 
-  the test cases are limited to the unit tests as this is the scope of the business case so no integration tests, differential tests or shadow mode tests are considered for this case study. But if interest arises can be demonstrated
- some test cases are purposefully designed to fail for demonstration purpose
