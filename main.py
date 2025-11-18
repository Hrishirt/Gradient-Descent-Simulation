"""
Gradient Descent Playground
A Streamlit app for visualizing gradient descent and optimizers on 1D functions.
"""

import numpy as np
import streamlit as st
from typing import Callable, Tuple, Dict, Any, Optional
import sympy as sp
from sympy import lambdify, Symbol
import time
import plotly.graph_objects as go


def f_quad(x: float) -> float:
    """Quadratic function: f(x) = (x - 2)^2 + 1"""
    return (x - 2) ** 2 + 1

def grad_quad(x: float) -> float:
    """Gradient of quadratic: f'(x) = 2(x - 2)"""
    return 2 * (x - 2)

def f_nonconvex(x: float) -> float:
    """Non-convex double well: f(x) = x^4 - 3x^2 + 2"""
    return x ** 4 - 3 * x ** 2 + 2

def grad_nonconvex(x: float) -> float:
    """Gradient of non-convex: f'(x) = 4x^3 - 6x"""
    return 4 * x ** 3 - 6 * x

def f_abs_lin(x: float) -> float:
    """Absolute value with linear term: f(x) = |x| + 0.1x"""
    return np.abs(x) + 0.1 * x

def grad_abs_lin(x: float) -> float:
    """Gradient of abs+linear: f'(x) = sign(x) + 0.1, sign(0) = 0"""
    if x > 0:
        return 1.0 + 0.1
    elif x < 0:
        return -1.0 + 0.1
    else:
        return 0.0 + 0.1

FUNCTIONS_1D: Dict[str, Dict[str, Any]] = {
    "Quadratic (x - 2)² + 1": {
        "func": f_quad,
        "grad": grad_quad,
        "plot_range": 6,
        "default_start": -4.0,
    },
    "Non-convex (x⁴ - 3x² + 2)": {
        "func": f_nonconvex,
        "grad": grad_nonconvex,
        "plot_range": 6,
        "default_start": -4.0,
    },
    "Abs + linear (|x| + 0.1x)": {
        "func": f_abs_lin,
        "grad": grad_abs_lin,
        "plot_range": 5,
        "default_start": -4.0,
    },
}


# ============================================================================
# 2D function definitions
# ============================================================================

def f_rosenbrock(point: np.ndarray) -> float:
    """Rosenbrock banana function."""
    x, y = point
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def grad_rosenbrock(point: np.ndarray) -> np.ndarray:
    x, y = point
    df_dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    df_dy = 200 * (y - x ** 2)
    return np.array([df_dx, df_dy], dtype=float)


def f_saddle(point: np.ndarray) -> float:
    """Simple saddle surface."""
    x, y = point
    return x ** 2 - y ** 2


def grad_saddle(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([2 * x, -2 * y], dtype=float)


def f_bowl(point: np.ndarray) -> float:
    """Offset quadratic bowl."""
    x, y = point
    return (x - 1) ** 2 + (y + 1) ** 2 + 1


def grad_bowl(point: np.ndarray) -> np.ndarray:
    x, y = point
    return np.array([2 * (x - 1), 2 * (y + 1)], dtype=float)


def f_ripple(point: np.ndarray) -> float:
    """Sinusoidal ripple surface."""
    x, y = point
    return np.sin(x) * np.cos(y) + 0.1 * (x ** 2 + y ** 2)


def grad_ripple(point: np.ndarray) -> np.ndarray:
    x, y = point
    df_dx = np.cos(x) * np.cos(y) + 0.2 * x
    df_dy = -np.sin(x) * np.sin(y) + 0.2 * y
    return np.array([df_dx, df_dy], dtype=float)


FUNCTIONS_2D: Dict[str, Dict[str, Any]] = {
    "Rosenbrock banana": {
        "func": f_rosenbrock,
        "grad": grad_rosenbrock,
        "plot_range": 2,
        "default_start": (-1.5, 1.5),
    },
    "Saddle (x² - y²)": {
        "func": f_saddle,
        "grad": grad_saddle,
        "plot_range": 4,
        "default_start": (2.0, 2.0),
    },
    "Bowl ((x-1)² + (y+1)²)": {
        "func": f_bowl,
        "grad": grad_bowl,
        "plot_range": 4,
        "default_start": (3.0, -1.0),
    },
    "Ripple (sin x cos y + 0.1(x²+y²))": {
        "func": f_ripple,
        "grad": grad_ripple,
        "plot_range": 5,
        "default_start": (2.5, 2.5),
    },
}

# ============================================================================
# Optimizer Implementations
# ============================================================================

def gd_step(x: float, grad_fn: Callable[[float], float], lr: float, state: Optional[Any]) -> Tuple[float, Optional[Any]]:
    """
    Vanilla Gradient Descent step.
    x_{t+1} = x_t - α * ∇f(x_t)
    """
    grad = grad_fn(x)
    new_x = x - lr * grad
    return new_x, None

def momentum_step(x: float, grad_fn: Callable[[float], float], lr: float, state: Optional[Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    """
    Momentum optimizer step.
    v_{t+1} = β * v_t + (1 - β) * ∇f(x_t)
    x_{t+1} = x_t - α * v_{t+1}
    β = 0.9
    """
    beta = 0.9
    grad = grad_fn(x)
    if state is None:
        v = np.zeros_like(grad, dtype=float)
    else:
        v = state.get('v', np.zeros_like(grad, dtype=float))
    
    v = beta * v + (1 - beta) * grad
    new_x = x - lr * v
    
    return new_x, {'v': v}

def adam_step(x: float, grad_fn: Callable[[float], float], lr: float, state: Optional[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """
    Adam optimizer step (simplified to 1D).
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    x_{t+1} = x_t - α * m̂_t / (sqrt(v̂_t) + ε)
    β₁ = 0.9, β₂ = 0.999, ε = 1e-8
    """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    grad = grad_fn(x)
    if state is None:
        m = np.zeros_like(grad, dtype=float)
        v = np.zeros_like(grad, dtype=float)
        t = 0
    else:
        m = state.get('m', np.zeros_like(grad, dtype=float))
        v = state.get('v', np.zeros_like(grad, dtype=float))
        t = state.get('t', 0)
    
    t += 1
    
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    new_x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return new_x, {'m': m, 'v': v, 't': t}

OPTIMIZERS: Dict[str, Callable[[float, Callable[[float], float], float, Optional[Any]], Tuple[float, Any]]] = {
    "Gradient Descent": gd_step,
    "Momentum": momentum_step,
    "Adam": adam_step,
}

# Simulation Logic


def _run_optimization_1d(
    f: Callable[[float], float],
    grad: Callable[[float], float],
    opt_name: str,
    x0: float,
    lr: float,
    steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run optimization simulation.
    
    Args:
        f: Function f(x)
        grad: Gradient function grad(x)
        opt_name: Key in OPTIMIZERS dict
        x0: Initial point (float)
        lr: Learning rate (float)
        steps: Number of iterations (int)
    
    Returns:
        xs: np.array of x values over time
        fs: np.array of f(x) values over time
    """
    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    optimizer_step = OPTIMIZERS[opt_name]
    xs = np.zeros(steps + 1)
    fs = np.zeros(steps + 1)
    
    x = x0
    state = None
    
    xs[0] = x
    fs[0] = f(x)
    
    for i in range(steps):
        x, state = optimizer_step(x, grad, lr, state)
        
        # Check for divergence (NaN or extremely large values)
        if np.isnan(x) or np.isinf(x) or abs(x) > 1e10:
            # Truncate arrays to valid portion
            xs = xs[:i+1]
            fs = fs[:i+1]
            break
        
        f_val = f(x)
        if np.isnan(f_val) or np.isinf(f_val) or abs(f_val) > 1e10:
            xs = xs[:i+1]
            fs = fs[:i+1]
            break
        
        xs[i+1] = x
        fs[i+1] = f_val
    
    return xs, fs


def _run_optimization_2d(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    opt_name: str,
    x0: Tuple[float, float],
    lr: float,
    steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run optimization for 2D functions.
    """
    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    optimizer_step = OPTIMIZERS[opt_name]
    x = np.array(x0, dtype=float)
    dim = x.shape[0]

    xs = np.zeros((steps + 1, dim))
    fs = np.zeros(steps + 1)

    state = None
    xs[0] = x
    fs[0] = f(x)

    for i in range(steps):
        x, state = optimizer_step(x, grad, lr, state)

        if np.any(np.isnan(x)) or np.any(np.isinf(x)) or np.linalg.norm(x) > 1e6:
            xs = xs[:i+1]
            fs = fs[:i+1]
            break

        f_val = f(x)
        if not np.isfinite(f_val) or abs(f_val) > 1e12:
            xs = xs[:i+1]
            fs = fs[:i+1]
            break

        xs[i+1] = x
        fs[i+1] = f_val

    return xs, fs


def run_optimization(
    f: Callable,
    grad: Callable,
    opt_name: str,
    x0: Any,
    lr: float,
    steps: int,
    dimension: str = "1D"
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch optimization based on dimensionality."""
    if dimension == "2D":
        return _run_optimization_2d(f, grad, opt_name, x0, lr, steps)
    return _run_optimization_1d(f, grad, opt_name, x0, lr, steps)

def _run_optimization_live_1d(
    f: Callable[[float], float],
    grad: Callable[[float], float],
    opt_name: str,
    x0: float,
    lr: float,
    steps: int,
    func_name: str,
    plot_placeholder,
    metrics_placeholder,
    animation_speed: float = 0.1,
    x_range: Tuple[float, float] = (-5.0, 5.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run optimization simulation with live animation.
    
    Args:
        f: Function f(x)
        grad: Gradient function grad(x)
        opt_name: Key in OPTIMIZERS dict
        x0: Initial point (float)
        lr: Learning rate (float)
        steps: Number of iterations (int)
        func_name: Name of the function for display
        plot_placeholder: Streamlit placeholder for the plot
        metrics_placeholder: Streamlit placeholder for metrics
        animation_speed: Delay between steps in seconds
        x_range: Range of x values for plotting
    
    Returns:
        xs: np.array of x values over time
        fs: np.array of f(x) values over time
    """
    if opt_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    optimizer_step = OPTIMIZERS[opt_name]
    xs = []
    fs = []
    
    x = x0
    state = None
    
    xs.append(x)
    fs.append(f(x))
    
    # Initial plot
    fig = plot_optimization_1d(f, np.array(xs), np.array(fs), func_name, opt_name, x_range=x_range)
    plot_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Initial metrics
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Step", "0")
        with col2:
            st.metric("Current x", f"{x:.4f}")
        with col3:
            st.metric("Current f(x)", f"{fs[0]:.4f}")
    
    for i in range(steps):
        x, state = optimizer_step(x, grad, lr, state)
        
        # Check for divergence (NaN or extremely large values)
        if np.isnan(x) or np.isinf(x) or abs(x) > 1e10:
            break
        
        f_val = f(x)
        if np.isnan(f_val) or np.isinf(f_val) or abs(f_val) > 1e10:
            break
        
        xs.append(x)
        fs.append(f_val)
        
        # Update plot
        fig = plot_optimization_1d(f, np.array(xs), np.array(fs), func_name, opt_name, x_range=x_range)
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Step", f"{i+1}/{steps}")
            with col2:
                st.metric("Current x", f"{x:.4f}")
            with col3:
                st.metric("Current f(x)", f"{f_val:.4f}")
        
        # Small delay for animation
        time.sleep(animation_speed)
    
    return np.array(xs), np.array(fs)


def run_optimization_live(
    f: Callable,
    grad: Callable,
    opt_name: str,
    x0: Any,
    lr: float,
    steps: int,
    func_name: str,
    plot_placeholder,
    metrics_placeholder,
    animation_speed: float = 0.1,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    dimension: str = "1D"
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch live optimization (1D only for now)."""
    if dimension == "2D":
        raise ValueError("Live animation is currently supported only for 1D functions.")
    return _run_optimization_live_1d(
        f, grad, opt_name, x0, lr, steps,
        func_name, plot_placeholder, metrics_placeholder,
        animation_speed, x_range
    )

# ============================================================================
# Custom Function Support
# ============================================================================

def create_function_from_string(func_str: str) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Create a function and its gradient from a string expression.
    Uses sympy for symbolic differentiation.
    
    Args:
        func_str: String expression like "x**2 + 1" or "(x - 2)**2 + 1"
    
    Returns:
        Tuple of (function, gradient_function)
    """
    try:
        x = Symbol('x')
        # Parse the expression - sympy can handle most mathematical expressions
        expr = sp.sympify(func_str, evaluate=False)
        
        # Create the function - sympy automatically maps functions to numpy
        f_lambda = lambdify(x, expr, modules=['numpy'])
        
        # Compute gradient symbolically
        grad_expr = sp.diff(expr, x)
        grad_lambda = lambdify(x, grad_expr, modules=['numpy'])
        
        # Wrap to handle edge cases
        def f(x_val: float) -> float:
            try:
                result = f_lambda(x_val)
                # Handle scalar results
                if isinstance(result, (int, float, np.number)):
                    result = float(result)
                else:
                    result = float(np.array(result).item())
                return result if np.isfinite(result) else np.nan
            except (ValueError, TypeError, ZeroDivisionError) as e:
                return np.nan
        
        def grad(x_val: float) -> float:
            try:
                result = grad_lambda(x_val)
                # Handle scalar results
                if isinstance(result, (int, float, np.number)):
                    result = float(result)
                else:
                    result = float(np.array(result).item())
                return result if np.isfinite(result) else np.nan
            except (ValueError, TypeError, ZeroDivisionError) as e:
                return np.nan
        
        return f, grad
    except Exception as e:
        raise ValueError(f"Error parsing function: {str(e)}. Make sure to use 'x' as the variable and valid Python/sympy syntax.")

# ============================================================================
# Streamlit UI
# ============================================================================

def _compute_y_limits(
    values: np.ndarray,
    trim_threshold: float = 5.0,
    margin: float = 0.1
) -> Optional[Tuple[float, float]]:
    """
    Compute y-axis limits that automatically zoom in when outliers dominate.

    Args:
        values: Array of y-values to analyze.
        trim_threshold: If full range is this many times larger than the trimmed
            (5th-95th percentile) range, use the trimmed range instead.
        margin: Fractional padding to add around the selected range.

    Returns:
        Tuple of (y_min, y_max) with margin applied, or None if values invalid.
    """
    finite_vals = values[np.isfinite(values)]
    if len(finite_vals) == 0:
        return None

    y_min = float(np.min(finite_vals))
    y_max = float(np.max(finite_vals))
    trimmed_min = float(np.percentile(finite_vals, 5))
    trimmed_max = float(np.percentile(finite_vals, 95))
    full_range = y_max - y_min
    trimmed_range = trimmed_max - trimmed_min

    if trimmed_range > 0 and full_range > trim_threshold * trimmed_range:
        y_min, y_max = trimmed_min, trimmed_max

    if y_max == y_min:
        y_min -= 1.0
        y_max += 1.0

    padding = (y_max - y_min) * margin
    return y_min - padding, y_max + padding


def plot_function_1d(
    f: Callable[[float], float],
    func_name: str,
    x_range: Tuple[float, float] = (-5.0, 5.0)
) -> go.Figure:
    """
    Plot just the function without optimization trajectory.
    
    Args:
        f: Function to plot
        func_name: Name of the function
        x_range: Range of x values to plot
    
    Returns:
        Plotly Figure object
    """
    # Plot the function
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    y_plot = np.array([f(x) for x in x_plot])
    
    fig = go.Figure()
    
    # Add function trace
    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        mode='lines',
        name='Function f(x)',
        line=dict(color='blue', width=2.5),
        hovertemplate='x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>'
    ))
    
    # Set layout
    y_limits = _compute_y_limits(y_plot)
    
    fig.update_layout(
        title=dict(text=func_name, font=dict(size=16, color='black')),
        xaxis_title='x',
        yaxis_title='f(x)',
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    if y_limits:
        fig.update_yaxes(range=y_limits)
    
    return fig

def plot_optimization_1d(
    f: Callable[[float], float],
    xs: np.ndarray,
    fs: np.ndarray,
    func_name: str,
    opt_name: str,
    x_range: Tuple[float, float] = (-5.0, 5.0)
) -> go.Figure:
    """
    Plot the function and optimization trajectory.
    
    Args:
        f: Function to plot
        xs: Array of x values from optimization
        fs: Array of f(x) values from optimization
        func_name: Name of the function
        opt_name: Name of the optimizer
        x_range: Range of x values to plot the function
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Always plot the function over the fixed range to keep view consistent
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    y_plot = np.array([f(x) for x in x_plot])
    
    # Add function trace
    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        mode='lines',
        name='Function f(x)',
        line=dict(color='blue', width=2.5),
        hovertemplate='x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>'
    ))
    
    # Calculate reasonable x-axis limits based on function range and optimization path
    x_min_plot = x_range[0]
    x_max_plot = x_range[1]
    
    plot_fs = np.array([])
    if len(xs) > 0:
        # Get valid x values from optimization path
        valid_xs = xs[np.isfinite(xs) & (np.abs(xs) < 1e6)]  # Cap at reasonable values
        
        if len(valid_xs) > 0:
            # Extend x-range slightly to show optimization path, but keep it reasonable
            path_x_min = np.min(valid_xs)
            path_x_max = np.max(valid_xs)
            
            # Only extend if path is within reasonable bounds (not diverged too far)
            if path_x_min > -50 and path_x_max < 50:
                x_min_plot = min(x_range[0], path_x_min - 1)
                x_max_plot = max(x_range[1], path_x_max + 1)
        
        # Filter out extreme values for plotting
        valid_mask = np.abs(xs) < 1e6
        plot_xs = xs[valid_mask]
        plot_fs = fs[valid_mask]
        
        if len(plot_xs) > 0:
            # Add optimization path
            fig.add_trace(go.Scatter(
                x=plot_xs,
                y=plot_fs,
                mode='lines+markers',
                name='Optimization path',
                line=dict(color='red', width=2),
                marker=dict(size=6, color='red', opacity=0.7),
                hovertemplate='Step: %{pointNumber}<br>x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>'
            ))
            
            # Mark the starting point
            if np.abs(xs[0]) < 1e6:
                fig.add_trace(go.Scatter(
                    x=[xs[0]],
                    y=[fs[0]],
                    mode='markers',
                    name='Start',
                    marker=dict(size=12, color='green', symbol='circle'),
                    hovertemplate='Start<br>x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>',
                    showlegend=True
                ))
            
            # Mark the final point
            if len(xs) > 1 and np.abs(xs[-1]) < 1e6:
                fig.add_trace(go.Scatter(
                    x=[xs[-1]],
                    y=[fs[-1]],
                    mode='markers',
                    name='End',
                    marker=dict(size=12, color='red', symbol='square'),
                    hovertemplate='End<br>x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>',
                    showlegend=True
                ))
    
    # Set layout
    title = f'{func_name} - {opt_name}'
    
    # Handle y-axis limits - focus on function range, but show optimization if reasonable
    if len(fs) > 0 and len(plot_fs) > 0:
        combined = np.concatenate([y_plot, plot_fs])
    else:
        combined = y_plot

    y_limits = _compute_y_limits(combined)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='black')),
        xaxis_title='x',
        yaxis_title='f(x)',
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(range=[x_min_plot, x_max_plot])
    )
    
    if y_limits:
        fig.update_yaxes(range=y_limits)
    
    return fig


def _build_surface_data(
    f: Callable[[np.ndarray], float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 55
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute surface grid for 2D functions."""
    x_values = np.linspace(x_range[0], x_range[1], resolution)
    y_values = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_values, y_values)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.apply_along_axis(f, 1, points).reshape(X.shape)
    return X, Y, Z


def plot_function_2d(
    f: Callable[[np.ndarray], float],
    func_name: str,
    xy_range: Tuple[float, float] = (-5.0, 5.0),
    resolution: int = 55
) -> go.Figure:
    """Plot a 2D function surface."""
    X, Y, Z = _build_surface_data(f, xy_range, xy_range, resolution)
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.9,
        showscale=False
    ))
    
    # Calculate z-axis range from surface data to match optimization plot
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    z_range = z_max - z_min
    # Add small padding (same as optimization plot)
    z_padding = z_range * 0.1 if z_range > 0 else 1.0
    z_axis_range = [z_min - z_padding, z_max + z_padding]
    
    fig.update_layout(
        title=dict(text=func_name, font=dict(size=16)),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x, y)',
            xaxis=dict(range=[xy_range[0], xy_range[1]], autorange=False),
            yaxis=dict(range=[xy_range[0], xy_range[1]], autorange=False),
            zaxis=dict(range=z_axis_range, autorange=False)  # Fix z-axis to match optimization plot, disable auto
        ),
        template='plotly_dark',
        margin=dict(l=20, r=20, t=60, b=20),
        height=520
    )
    return fig


def plot_optimization_2d(
    f: Callable[[np.ndarray], float],
    xs: np.ndarray,
    fs: np.ndarray,
    func_name: str,
    opt_name: str,
    xy_range: Tuple[float, float] = (-5.0, 5.0),
    surface_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    resolution: int = 55
) -> go.Figure:
    """Plot 2D optimization trajectory on a surface."""
    if surface_data is None:
        surface_data = _build_surface_data(f, xy_range, xy_range, resolution)
    X, Y, Z = surface_data
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.85,
        showscale=False
    ))
    
    if xs is not None and len(xs) > 0:
        xs = np.atleast_2d(xs)
        fs = np.asarray(fs)
        fig.add_trace(go.Scatter3d(
            x=xs[:, 0],
            y=xs[:, 1],
            z=fs,
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=4, color='red'),
            name='Optimization path'
        ))
        fig.add_trace(go.Scatter3d(
            x=[xs[0, 0]],
            y=[xs[0, 1]],
            z=[fs[0]],
            mode='markers',
            marker=dict(size=9, color='green'),
            name='Start'
        ))
        fig.add_trace(go.Scatter3d(
            x=[xs[-1, 0]],
            y=[xs[-1, 1]],
            z=[fs[-1]],
            mode='markers',
            marker=dict(size=9, color='red', symbol='square'),
            name='End'
        ))
    
    # Calculate z-axis range from surface data to preserve surface shape
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))
    z_range = z_max - z_min
    # Add small padding (same as preview plot)
    z_padding = z_range * 0.1 if z_range > 0 else 1.0
    z_axis_range = [z_min - z_padding, z_max + z_padding]
    
    fig.update_layout(
        title=dict(text=f"{func_name} - {opt_name}", font=dict(size=16)),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x, y)',
            xaxis=dict(range=[xy_range[0], xy_range[1]], autorange=False),
            yaxis=dict(range=[xy_range[0], xy_range[1]], autorange=False),
            zaxis=dict(range=z_axis_range, autorange=False)  # Fix z-axis to surface range, disable auto
        ),
        template='plotly_dark',
        margin=dict(l=20, r=20, t=60, b=20),
        height=520,
        showlegend=True
    )
    return fig
def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Gradient Descent Playground", layout="wide")

    # Global styling tweaks
    st.markdown(
        """
        <style>
        div[data-baseweb="textarea"] textarea {
            color: #f5f5f5 !important;
            background-color: #11141c !important;
            border-color: #4a4f63 !important;
            caret-color: #f5f5f5 !important;
        }
        div[data-baseweb="textarea"] textarea::placeholder {
            color: #cccccc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("Gradient Descent Playground")
    st.markdown("Visualize how different optimizers behave on various 1D functions")
    
    # Sidebar controls
    custom_func_str: Optional[str] = None
    func_display_name: Optional[str] = None
    selected_meta: Optional[Dict[str, Any]] = None

    with st.sidebar:
        st.header("Controls")
        
        dimension = st.radio("Function dimension", ["1D", "2D"], index=0)
        
        if dimension == "1D":
            func_mode = st.radio(
                "Function Mode",
                options=["Preset Functions", "Custom Function"],
                index=0
            )
        else:
            func_mode = "Preset Functions"
            st.caption("Custom 2D functions coming soon.")
        
        if func_mode == "Preset Functions":
            function_dict = FUNCTIONS_1D if dimension == "1D" else FUNCTIONS_2D
            func_name = st.selectbox(
                "Function",
                options=list(function_dict.keys()),
                index=0
            )
            selected_meta = function_dict[func_name]
            func_display_name = func_name
            custom_func_str = None
        else:
            func_name = "Custom Function"
            selected_meta = None
            st.markdown("**Custom Function (1D)**")
            st.markdown("Enter a function of `x` using Python/sympy syntax:")
            st.code("x**2 + 1\n(x - 2)**2\nx**4 - 3*x**2 + 2\nsin(x) + x**2", language="python")
            
            custom_func_str = st.text_area(
                "Function f(x) =",
                value="x**2 + 1",
                height=100,
                help="Use 'x' as the variable. Supports: +, -, *, /, **, sin, cos, exp, log, abs, sqrt, etc.",
                key="custom_func_input"
            )
            
            generate_graph_button = st.button("Generate Graph", use_container_width=True)
            
            try:
                x_sym = Symbol('x')
                expr = sp.sympify(custom_func_str.strip(), evaluate=False)
                latex_str = sp.latex(expr)
                func_display_name = f"Custom: f(x) = ${latex_str}$"
                st.latex(f"f(x) = {latex_str}")
            except Exception:
                func_display_name = f"Custom: {custom_func_str}"
            
            if generate_graph_button:
                st.session_state['show_custom_preview'] = True
                st.session_state['custom_func_str'] = custom_func_str
            elif 'custom_func_str' in st.session_state and st.session_state['custom_func_str'] != custom_func_str:
                st.session_state['show_custom_preview'] = False
        
        opt_name = st.selectbox(
            "Optimizer",
            options=list(OPTIMIZERS.keys()),
            index=0
        )
        
        if dimension == "1D":
            default_start = selected_meta.get("default_start", -4.0) if selected_meta else -4.0
            x0 = st.number_input(
                "Starting point x₀",
                value=float(default_start),
                step=0.1,
                format="%.2f"
            )
        else:
            default_start = selected_meta.get("default_start", (-2.0, 2.0)) if selected_meta else (-2.0, 2.0)
            col_x, col_y = st.columns(2)
            with col_x:
                x0_x = st.number_input(
                    "Starting x₀ (x)",
                    value=float(default_start[0]),
                    step=0.1,
                    format="%.2f"
                )
            with col_y:
                x0_y = st.number_input(
                    "Starting y₀ (y)",
                    value=float(default_start[1]),
                    step=0.1,
                    format="%.2f"
                )
            x0 = np.array([x0_x, x0_y], dtype=float)
        
        lr = st.number_input(
            "Learning rate α",
            value=0.001,
            min_value=0.001,
            max_value=10.0,
            step=0.01,
            format="%.3f"
        )
        
        steps = st.slider(
            "Number of steps",
            min_value=1,
            max_value=200,
            value=30,
            step=1
        )
        
        default_plot_range = int(selected_meta.get("plot_range", 6)) if selected_meta else 6
        plot_label = "Plot range (±x)" if dimension == "1D" else "Plot range (±x, ±y)"
        plot_range_val = st.slider(
            plot_label,
            min_value=2,
            max_value=30,
            value=default_plot_range,
            step=1,
            help="Controls how wide the plotted range is. Use the toolbar to zoom further."
        )
        plot_range_tuple = (-float(plot_range_val), float(plot_range_val))
        
        if dimension == "1D":
            animation_enabled = st.checkbox(
                "Enable live animation",
                value=True,
                help="Show optimization steps in real-time"
            )
            if animation_enabled:
                animation_speed = st.slider(
                    "Animation speed",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    format="%.2f",
                    help="Delay between steps (seconds)"
                )
            else:
                animation_speed = 0.0
        else:
            animation_enabled = False
            animation_speed = 0.0
            st.caption("Live animation is currently available for 1D functions only.")
        
        run_button = st.button("Run Optimization", type="primary", use_container_width=True)
    
    # Main content area - preview placeholder
    preview_placeholder = st.empty()
    
    if run_button:
        preview_placeholder.empty()
        try:
            if dimension == "1D":
                if func_mode == "Preset Functions":
                    meta = FUNCTIONS_1D[func_name]
                    f = meta["func"]
                    grad = meta["grad"]
                    display_name = func_name
                else:
                    if not custom_func_str or custom_func_str.strip() == "":
                        st.error("Please enter a custom function.")
                        st.stop()
                    f, grad = create_function_from_string(custom_func_str.strip())
                    display_name = func_display_name if func_display_name else f"Custom: {custom_func_str}"
            else:
                meta = FUNCTIONS_2D[func_name]
                f = meta["func"]
                grad = meta["grad"]
                display_name = func_name
        except Exception as e:
            st.error(f"Error creating function: {str(e)}")
            if dimension == "1D":
                st.info("**Tips:** Use `x` as the variable name, `**` for powers, and functions like `sin`, `cos`, `exp`, `log`.")
            st.stop()
        
        try:
            if animation_enabled and dimension == "1D":
                plot_placeholder = st.empty()
                metrics_placeholder = st.empty()
                xs, fs = run_optimization_live(
                    f, grad, opt_name, x0, lr, steps,
                    display_name, plot_placeholder, metrics_placeholder,
                    animation_speed, plot_range_tuple, dimension="1D"
                )
            else:
                xs, fs = run_optimization(f, grad, opt_name, x0, lr, steps, dimension=dimension)
                if dimension == "1D":
                    fig = plot_optimization_1d(f, xs, fs, display_name, opt_name, x_range=plot_range_tuple)
                else:
                    fig = plot_optimization_2d(f, xs, fs, display_name, opt_name, xy_range=plot_range_tuple)
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Optimization Results")
            import pandas as pd
            if dimension == "1D":
                results_data = {
                    'Iteration': list(range(len(xs))),
                    'x_t': xs,
                    'f(x_t)': fs
                }
            else:
                results_data = {
                    'Iteration': list(range(xs.shape[0])),
                    'x_t': xs[:, 0],
                    'y_t': xs[:, 1],
                    'f(x_t, y_t)': fs
                }
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Summary metrics
            if dimension == "1D":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial f(x)", f"{fs[0]:.4f}")
                with col2:
                    st.metric("Final f(x)", f"{fs[-1]:.4f}")
                with col3:
                    st.metric("Improvement", f"{(fs[0] - fs[-1]):.4f}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Initial f(x,y)", f"{fs[0]:.4f}")
                with col2:
                    st.metric("Final f(x,y)", f"{fs[-1]:.4f}")
                with col3:
                    st.metric("Final x", f"{xs[-1, 0]:.3f}")
                with col4:
                    st.metric("Final y", f"{xs[-1, 1]:.3f}")
                st.caption(f"Improvement: {(fs[0] - fs[-1]):.4f}")
            
            if len(fs) < steps + 1:
                st.warning("Optimization diverged or encountered numerical issues. Try reducing the learning rate.")
        
        except Exception as e:
            st.error(f"Error running optimization: {str(e)}")
            st.exception(e)
    else:
        # Show preview when idle
        if dimension == "1D":
            if func_mode == "Preset Functions":
                try:
                    meta = FUNCTIONS_1D[func_name]
                    fig_preview = plot_function_1d(meta["func"], func_name, x_range=plot_range_tuple)
                    preview_placeholder.plotly_chart(fig_preview, use_container_width=True)
                except Exception:
                    preview_placeholder.empty()
            elif func_mode == "Custom Function" and st.session_state.get('show_custom_preview', False):
                custom_expr = st.session_state.get('custom_func_str', 'x**2 + 1')
                try:
                    f_preview, _ = create_function_from_string(custom_expr.strip())
                    x_sym = Symbol('x')
                    expr = sp.sympify(custom_expr.strip(), evaluate=False)
                    latex_str = sp.latex(expr)
                    fig_preview = plot_function_1d(f_preview, f"Custom: f(x) = ${latex_str}$", x_range=plot_range_tuple)
                    preview_placeholder.plotly_chart(fig_preview, use_container_width=True)
                except Exception as e:
                    preview_placeholder.empty()
                    st.error(f"Error plotting function: {str(e)}")
            else:
                preview_placeholder.empty()
                st.info("Adjust the parameters in the sidebar and click 'Run Optimization' to start!")
        else:
            try:
                meta = FUNCTIONS_2D[func_name]
                fig_preview = plot_function_2d(meta["func"], func_name, xy_range=plot_range_tuple)
                preview_placeholder.plotly_chart(fig_preview, use_container_width=True)
            except Exception as e:
                preview_placeholder.empty()
                st.error(f"Error plotting surface: {str(e)}")

if __name__ == "__main__":
    main()

