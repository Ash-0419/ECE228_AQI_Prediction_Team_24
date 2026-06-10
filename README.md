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

To train all models from scratch and generate the comparison report:
```bash
python master_trainer.py
```

The script will:
- Load and preprocess the `AirQuality_Cleaned.csv` data.
- Run hyperparameter optimization for each model using GWO and PSO.
- Train the final models.
- Save metrics (RMSE, MAE, R2, Time) to `final_results_report.csv`.

## Results
Final evaluation metrics are saved automatically to `final_results_report.csv` after execution.
