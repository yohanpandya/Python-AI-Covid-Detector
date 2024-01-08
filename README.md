In this project, I use PyTorch to create a regression model that can predict how many deaths there will be for a Wisconsin census tract, given the number of people who have tested positive, broken down by age. The train.csv and test.csv files are based on data from https://dhsgis.wi.gov/, downloaded in Spring of 2023 (it appears the dataset is no longer available). p2.py is a JupyterLab file that was converted from .ipynb to .py. It is broken up into 10 cells, each one outputting an important step in the process of training and testing a ML model.

This project requires you to have matplotlib and torch installed (which can both be installed with pip install ___). In this project, I leverage pandas dataframes and pyTorch tensors to output useful information about my predictions.

Cell 1 - Memory Consumption Calculation:

This cell imports necessary libraries, reads data from CSV files into Pandas DataFrames (train_df and test_df), and converts them into PyTorch tensors (trainX, trainY, testX, and testY).
It calculates the approximate memory consumption of the trainX tensor by iterating through its elements and summing up the memory used by each element.

Cell 2 - Difference Calculation:

This cell calculates the maximum absolute difference between the original trainX tensor and a new tensor trainX_float16 created by converting trainX to a lower precision data type (torch.float16). The goal is to determine the impact on precision when using a lower precision data type.

Cell 3 - CUDA Availability Check:

This cell checks whether a CUDA-compatible GPU is available for GPU acceleration by calling torch.cuda.is_available() and prints the result as a Boolean value (True/False).

Cell 4 - Predicted Number of Deaths Calculation:

In this cell, a set of coefficients (coef) and the first row of the testX tensor (first_row) are used to calculate the predicted number of deaths for the first census tract. The result is printed as firstRowResult.

Cell 5 - Average Predicted Deaths Calculation:

This cell calculates the average number of predicted deaths over the entire testX dataset by multiplying testX by the coef tensor and computing the mean.

Cell 6 - Optimization Initialization:

This cell defines a quadratic function f(x) and initializes an optimization variable x with a value of 0.0 for later optimization.

Cell 7 - Optimization:

This cell performs an optimization to find the value of x that minimizes the quadratic function f(x). It uses stochastic gradient descent (SGD) with a learning rate of 0.1.

Cell 8 - Mean-Squared Error Calculation with Zero Coefficients:

In this cell, the mean-squared error (MSE) is calculated when making predictions using a vector of zero coefficients (coef). The MSE is computed by comparing the predictions with the actual trainY values.

Cell 9 - Batch Gradient Descent Optimization:

This cell demonstrates batch gradient descent optimization to update the coefficients (coef) for a linear regression model. It uses a dataset loader (dl) to process the training data in batches and updates the coefficients iteratively.

Cell 10 - Mean-Squared Error Calculation on Test Data:

This cell calculates the mean-squared error (MSE) on the test data (testX and testY) using the optimized coefficients (coef) obtained in the previous cell.
The code cells include various computations, optimizations, and evaluations related to machine learning and numerical optimization. They use PyTorch for tensor operations and optimization routines, making it a comprehensive example of data processing, modeling, and evaluation tasks.
