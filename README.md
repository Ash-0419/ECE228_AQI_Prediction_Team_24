# AQI Prediction Optimization Project

This project implements and compares four different machine learning models optimized using **Grey Wolf Optimizer (GWO)** and **Particle Swarm Optimization (PSO)** for Air Quality Index (AQI) prediction.

## Models Implemented
1. **XGBoost**: Gradient Boosting Regressor.
2. **Prophet**: Time-series forecasting by Meta.
3. **LSTM**: Long Short-Term Memory Neural Network.
4. **ARIMA-BiLSTM**: Hybrid model combining statistical ARIMA with deep learning BiLSTM.

## Optimizers
- **GWO (Grey Wolf Optimizer)**: Swarm-based metaheuristic inspired by grey wolves.
- **PSO (Particle Swarm Optimization)**: Population-based stochastic optimization.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd AQI_Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train all models from scratch and generate the comparison reports, run each model in Jupyter Notebook or Google Colab.

## Results
Final evaluation metrics and relavent comparison graphs of Predicted vs. Actual AQI are visible upon running each model's relavent file.
