# Gradient Descent Playground

An educational web app that visualizes gradient descent and various optimizers on simple 1D functions.

## Features

- **Three 1D Functions:**
  - Quadratic: `f(x) = (x - 2)² + 1`
  - Non-convex double well: `f(x) = x⁴ - 3x² + 2`
  - Absolute value with linear term: `f(x) = |x| + 0.1x`

- **Three Optimizers:**
  - Vanilla Gradient Descent
  - Momentum (β = 0.9)
  - Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8)

- **Interactive Visualization:**
  - Real-time plotting of function landscape and optimization trajectory
  - Results table showing iteration, x_t, and f(x_t) values
  - Summary statistics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run main.py
```

The app will open in your browser. Use the sidebar to:
- Select a function
- Choose an optimizer
- Set the starting point x₀
- Adjust the learning rate α
- Set the number of steps

Click "Run Optimization" to visualize the optimization process!

## Requirements

- Python 3.8+
- streamlit
- numpy
- matplotlib
- pandas

