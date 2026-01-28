# Climate Data Analysis and Regression Modeling

A Python-based analysis of historical temperature data using regression models to explore climate trends and evaluate model performance. Completed as part of MIT 6.100B (Introduction to Computational Thinking and Data Science).

## Overview

This project analyzes temperature data from 21 U.S. cities spanning 1961-2015 to investigate climate trends using polynomial regression models. The analysis explores concepts including:

- Linear and polynomial regression fitting
- Model evaluation using R² and RMSE metrics
- Overfitting vs. generalization
- Moving averages for noise reduction
- Statistical significance (SE/slope ratio)

## Key Findings

### Temperature Trends
- **Single-day analysis** (Jan 10th): High noise, low R² (~0.05)
- **Yearly average**: Reduced noise, improved R² (~0.19)
- **Multi-city average**: Further noise reduction, R² (~0.75)
- **5-year moving average**: Smoothest trend, R² (~0.92)

All analyses show a positive slope with SE/slope < 0.5, supporting the claim of rising temperatures over time.

### Model Comparison (Degrees 1, 2, 20)

| Model | Training R² | Test RMSE |
|-------|-------------|-----------|
| Degree 1 | 0.92 | 0.08 |
| Degree 2 | 0.94 | 0.21 |
| Degree 20 | 0.97 | 1.49 |

**Key insight**: Higher-degree models achieve better training R² but worse test RMSE due to overfitting. The degree-1 (linear) model generalizes best to unseen data.

### Extreme Temperature Analysis
Contrary to expectations, the standard deviation of temperatures across cities shows a *decreasing* trend over time, suggesting temperature variation is not increasing by this metric.

## Project Structure

```
├── ps5.py              # Main implementation
├── ps5_test.py         # Unit tests
├── data.csv            # Temperature dataset
└── ps5_writeup.pdf     # Analysis and discussion
```

## Functions Implemented

- `generate_models()` - Fits polynomial regression models to data
- `evaluate_models_on_training()` - Plots data with best-fit curves, computes R² and SE/slope
- `moving_average()` - Computes moving average with variable window size
- `rmse()` - Calculates root mean squared error
- `r_squared()` - Calculates coefficient of determination
- `gen_cities_avg()` - Computes average temperature across multiple cities
- `gen_std_devs()` - Computes standard deviation of temperatures across cities
- `evaluate_models_on_testing()` - Evaluates trained models on test data

## Concepts Demonstrated

- **Noise reduction**: Averaging across time (moving average) and space (multiple cities) reveals underlying trends
- **Overfitting**: Complex models (degree 20) fit training data well but fail on new data
- **Generalization**: Simple models (degree 1) capture the true trend and predict better
- **Statistical significance**: SE/slope ratio indicates confidence in trend direction

## Requirements

- Python 3.x
- NumPy
- Matplotlib (pylab)

## Usage

```python
python ps5.py
```

## Acknowledgments

Dataset and problem set provided by MIT 6.100B course staff.
