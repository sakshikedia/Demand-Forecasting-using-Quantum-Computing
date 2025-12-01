# README: Demand Forecasting Using Quantum Computing

## Project Overview

This project explores the application of quantum computing techniques to the domain of demand forecasting, particularly focusing on stock price prediction. By integrating classical data preprocessing methods with quantum machine learning models, the project aims to harness the potential computational advantages of quantum algorithms to improve forecasting accuracy on financial time-series data.

The core of the project revolves around developing a hybrid classical-quantum pipeline that fetches historical stock data, preprocesses it for quantum-ready input, designs and trains a quantum neural network-based regressor using Qiskit and related quantum machine learning libraries, and compares the results against classical benchmarks.

***

## Modules and Detailed Descriptions

### 1. Data Acquisition and Preprocessing

The data module is responsible for acquiring historical stock market data for a specific ticker symbol using the Yahoo Finance API (`yfinance` library). In this project, data for the stock ticker 'F' (Ford Motor Company) from January 1, 2000, to the present date is retrieved.

Key steps in this module include:
- Downloading daily stock price data encompassing Open, High, Low, Close prices, and Volume.
- Calculating percentage change for the closing price to use as a primary target variable for forecasting.
- Dropping missing entries and organizing the dataset for further analysis.
- Selecting relevant features such as Open, High, Low, Close, and Volume.
- Normalizing the selected features and target variable using MinMaxScaler to scale the data between 0 and 1, which is a critical requirement for quantum circuits that often expect normalized input due to amplitude encoding or other quantum data encoding schemes.

This preprocessing step ensures the financial data is in an optimal format for quantum circuit input and improves the stability of model training.

### 2. Quantum Circuit Design

This module revolves around constructing a parameterized quantum circuit that acts as the feature map and ansatz for the quantum neural network regressor.

Core components include:
- Defining the number of qubits based on the number of features (in this case, five qubits corresponding to Open, High, Low, Close, and Volume).
- Creating parameterized rotation gates for each qubit using rotation angles that are trainable parameters.
- Structuring the circuit with layers of rotations (Ry gates) and entangling gates (CNOT or others), forming a variational quantum circuit capable of representing complex nonlinear functions suitable for regression tasks.
- Displaying and verifying the constructed circuit visually using Qiskit's built-in drawing tools.

The parameterized quantum circuit serves as the model architecture in the quantum machine learning framework, where the quantum computer or simulator evaluates the circuit for different input parameters during training and inference.

### 3. Quantum Neural Network (QNN) Model

Integrating the quantum circuit into a machine learning pipeline is managed here through Qiskit's machine learning module:

- Using the `EstimatorQNN` interface to wrap the quantum circuit as a neural network with trainable parameters.
- Choosing a variational quantum regressor (VQR) as the algorithm that uses the QNN for regression.
- Specifying the loss function as the L2 loss (mean squared error) and using an optimizer such as COBYLA, which is a classical optimization algorithm well-suited for the noisy optimization problems in quantum machine learning.
- Training involves feeding the normalized feature vectors into the quantum circuit, evaluating the predicted outputs, and adjusting parameters to minimize the loss.

This model effectively acts as a quantum-enhanced regressor, capable of capturing complex data patterns leveraging quantum state representations and transformations.

### 4. Model Training and Prediction

This module covers the training phase of the quantum neural network with the prepared dataset and performing predictions.

Key details:
- The model is trained on the normalized features and targets, iterating through numerous optimization steps until convergence or reaching a maximum number of iterations.
- Once trained, the model predicts the normalized stock prices for the entire dataset.
- The predicted normalized outputs are inverse-transformed back to the original scale using the target scaler, allowing direct comparison with real stock price values.

Training quantum models can be computationally expensive; thus, it’s typically done using quantum simulators provided by Aer or other quantum backends unless access to actual quantum hardware is available.

### 5. Visualization and Evaluation

Visualization is crucial for assessing the model's performance visually and quantitatively:

- Plotting the actual vs. predicted closing prices over the time range being studied.
- The plot uses different colors and line styles to differentiate actual market trends and model predictions.
- Setting axis labels, title, grid, and legend for clarity and presentation quality.

For quantitative evaluation, the project computes:
- Root Mean Squared Error (RMSE) between actual closing prices and predictions, offering an intuitive metric for forecasting accuracy.

The RMSE value allows evaluating how closely the quantum model approximates real stock market behavior, providing insight into its practical efficacy.

### 6. Integration with Classical Learning (Implicit)

While the project focuses on quantum approaches, the data normalization and preprocessing rely on classical methodologies (e.g., sklearn’s MinMaxScaler). Furthermore, classical optimization algorithms play a role in tuning quantum model parameters. This hybrid design illustrates how quantum computing synergizes with classical computation to address complex forecasting problems.

***

## Technologies and Libraries Used

- **Python**: Primary programming language for scripting, data handling, and orchestration of quantum workflows.
- **yfinance**: API to fetch historical market data from Yahoo Finance.
- **NumPy and Pandas**: Core libraries for numerical operations and data frame management.
- **Matplotlib**: Used for visualization of time series and model predictions.
- **Qiskit Framework**: The main quantum computing library used for designing quantum circuits, quantum simulators (Aer), and integrating quantum machine learning components.
- **Qiskit Machine Learning**: Specialized sublibrary to apply quantum neural networks and variational algorithms for regression tasks.
- **scikit-learn**: Provides preprocessing utilities and evaluation metrics such as RMSE.
- **COBYLA Optimizer**: Classical optimizer adapted to the quantum context for efficient parameter updates.

***

## System Workflow Summary

1. **Fetch Historical Data** from finance APIs.
2. **Preprocess Data** by cleaning, selecting features, and normalizing.
3. **Construct Parameterized Quantum Circuits** as learning models.
4. **Wrap Circuits in Quantum Neural Networks** and define regression objectives.
5. **Train Quantum Model** using hybrid classical-quantum optimization.
6. **Predict Stock Prices** and reverse-scaling predicted values.
7. **Visualize Results** and compute error metrics for evaluation.

***

## Potential Extensions

- Extending to multiple stock tickers and multi-variate forecasting.
- Experimenting with different quantum encoding schemes like amplitude encoding or basis encoding.
- Comparing with classical deep learning models such as LSTM or CNN for timeseries.
- Implementation on real quantum hardware for empirical assessment.
- Incorporating longer timeframes and intraday data for higher granularity.

***

This project is a pioneering effort showcasing how quantum computation can be applied to classical financial forecasting problems by creating a modular and extensible framework. It demonstrates essential quantum circuit modeling techniques alongside modern classical preprocessing and evaluation methods to provide meaningful forecasting insights.

