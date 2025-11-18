"""
Gradient Descent Playground
A Streamlit app for visualizing gradient descent and optimizers on 1D functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Callable, Tuple, Dict, Any, Optional
import sympy as sp
from sympy import lambdify, Symbol
import time

# Enable LaTeX rendering in matplotlib
plt.rcParams['text.usetex'] = False  # Use matplotlib's built-in LaTeX-like rendering
plt.rcParams['mathtext.fontset'] = 'stix'  # Better math font


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

FUNCTIONS: Dict[str, Tuple[Callable[[float], float], Callable[[float], float]]] = {
    "Quadratic (x - 2)² + 1": (f_quad, grad_quad),
    "Non-convex (x⁴ - 3x² + 2)": (f_nonconvex, grad_nonconvex),
    "Abs + linear (|x| + 0.1x)": (f_abs_lin, grad_abs_lin),
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
    if state is None:
        v = 0.0
    else:
        v = state.get('v', 0.0)
    
    grad = grad_fn(x)
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
    
    if state is None:
        m = 0.0
        v = 0.0
        t = 0
    else:
        m = state.get('m', 0.0)
        v = state.get('v', 0.0)
        t = state.get('t', 0)
    
    t += 1
    grad = grad_fn(x)
    
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

# ============================================================================
# Simulation Logic
# ============================================================================

def run_optimization(
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

def run_optimization_live(
    f: Callable[[float], float],
    grad: Callable[[float], float],
    opt_name: str,
    x0: float,
    lr: float,
    steps: int,
    func_name: str,
    plot_placeholder,
    metrics_placeholder,
    animation_speed: float = 0.1
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
    fig = plot_optimization(f, np.array(xs), np.array(fs), func_name, opt_name)
    plot_placeholder.pyplot(fig)
    plt.close(fig)
    
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
        fig = plot_optimization(f, np.array(xs), np.array(fs), func_name, opt_name)
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
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

def plot_function_only(
    f: Callable[[float], float],
    func_name: str,
    x_range: Tuple[float, float] = (-5.0, 5.0)
) -> plt.Figure:
    """
    Plot just the function without optimization trajectory.
    
    Args:
        f: Function to plot
        func_name: Name of the function
        x_range: Range of x values to plot
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the function
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    y_plot = np.array([f(x) for x in x_plot])
    ax.plot(x_plot, y_plot, 'b-', linewidth=2.5, label='Function f(x)', zorder=3)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'{func_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Handle y-axis limits
    valid_y = y_plot[np.isfinite(y_plot) & (np.abs(y_plot) < 1e10)]
    if len(valid_y) > 0:
        y_min = np.min(valid_y)
        y_max = np.max(valid_y)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    return fig

def plot_optimization(
    f: Callable[[float], float],
    xs: np.ndarray,
    fs: np.ndarray,
    func_name: str,
    opt_name: str,
    x_range: Tuple[float, float] = (-5.0, 5.0)
) -> plt.Figure:
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
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Always plot the function over the fixed range to keep view consistent
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    y_plot = np.array([f(x) for x in x_plot])
    ax.plot(x_plot, y_plot, 'b-', linewidth=2.5, label='Function f(x)', zorder=3)
    
    # Calculate reasonable x-axis limits based on function range and optimization path
    # But cap extreme values to keep the view focused
    x_min_plot = x_range[0]
    x_max_plot = x_range[1]
    
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
            # If path has diverged, keep the original range focused on function
            # but still show the path points (they'll be off-screen but that's okay)
        
        # Plot the optimization trajectory with lower zorder so function shows through
        # Filter out extreme values for plotting
        valid_mask = np.abs(xs) < 1e6
        plot_xs = xs[valid_mask]
        plot_fs = fs[valid_mask]
        
        if len(plot_xs) > 0:
            ax.plot(plot_xs, plot_fs, 'ro-', markersize=8, linewidth=2, alpha=0.7, label='Optimization path', zorder=2)
            # Mark the starting point (always show if reasonable)
            if np.abs(xs[0]) < 1e6:
                ax.plot(xs[0], fs[0], 'go', markersize=12, label='Start', zorder=5)
            # Mark the final point (always show if reasonable)
            if len(xs) > 1 and np.abs(xs[-1]) < 1e6:
                ax.plot(xs[-1], fs[-1], 'rs', markersize=12, label='End', zorder=5)
    
    # Set x-axis limits to keep view focused on function's interesting region
    ax.set_xlim(x_min_plot, x_max_plot)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    # Set title with LaTeX rendering if needed
    title = f'{func_name} - {opt_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Handle y-axis limits - focus on function range, but show optimization if reasonable
    valid_y_plot = y_plot[np.isfinite(y_plot) & (np.abs(y_plot) < 1e10)]
    if len(valid_y_plot) > 0:
        y_min_func = np.min(valid_y_plot)
        y_max_func = np.max(valid_y_plot)
        
        if len(fs) > 0:
            valid_fs = fs[np.isfinite(fs) & (np.abs(fs) < 1e6)]  # Cap extreme values
            if len(valid_fs) > 0:
                # Use function range as primary, extend slightly for optimization if reasonable
                y_min = min(y_min_func, np.min(valid_fs))
                y_max = max(y_max_func, np.max(valid_fs))
                # But don't let optimization extremes dominate the view
                if y_max - y_min > 10 * (y_max_func - y_min_func):
                    # Optimization has diverged, focus on function range
                    y_min = y_min_func
                    y_max = y_max_func
            else:
                y_min = y_min_func
                y_max = y_max_func
        else:
            y_min = y_min_func
            y_max = y_max_func
        
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Gradient Descent Playground", layout="wide")
    
    st.title("Gradient Descent Playground")
    st.markdown("Visualize how different optimizers behave on various 1D functions")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Function selection mode
        func_mode = st.radio(
            "Function Mode",
            options=["Preset Functions", "Custom Function"],
            index=0
        )
        
        if func_mode == "Preset Functions":
            func_name = st.selectbox(
                "Function",
                options=list(FUNCTIONS.keys()),
                index=0
            )
            custom_func_str = None
            func_display_name = func_name
        else:
            st.markdown("**Custom Function**")
            st.markdown("Enter a function of `x` using Python/sympy syntax:")
            st.markdown("Examples:")
            st.code("x**2 + 1\n(x - 2)**2\nx**4 - 3*x**2 + 2\nsin(x) + x**2", language="python")
            
            custom_func_str = st.text_area(
                "Function f(x) =",
                value="x**2 + 1",
                height=100,
                help="Use 'x' as the variable. Supports: +, -, *, /, **, sin, cos, exp, log, abs, sqrt, etc.",
                key="custom_func_input"
            )
            
            # Generate Graph button for custom function
            generate_graph_button = st.button("Generate Graph", use_container_width=True)
            
            # Try to create a nice display name with LaTeX
            func_display_name = None
            try:
                x_sym = Symbol('x')
                expr = sp.sympify(custom_func_str.strip(), evaluate=False)
                latex_str = sp.latex(expr)
                func_display_name = f"Custom: f(x) = ${latex_str}$"
                st.latex(f"f(x) = {latex_str}")
            except:
                func_display_name = f"Custom: {custom_func_str}"
            
            # Store state for preview when Generate Graph is clicked
            if generate_graph_button:
                st.session_state['show_custom_preview'] = True
                st.session_state['custom_func_str'] = custom_func_str
            # Clear preview if function text changed
            elif 'custom_func_str' in st.session_state and st.session_state['custom_func_str'] != custom_func_str:
                st.session_state['show_custom_preview'] = False
        
        opt_name = st.selectbox(
            "Optimizer",
            options=list(OPTIMIZERS.keys()),
            index=0
        )
        
        x0 = st.number_input(
            "Starting point x₀",
            value=-4.0,
            step=0.1,
            format="%.2f"
        )
        
        lr = st.number_input(
            "Learning rate α",
            value=0.1,
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
        
        run_button = st.button("Run Optimization", type="primary", use_container_width=True)
    
    # Main content area - Show preview plots (only if not running optimization)
    preview_placeholder = st.empty()
    
    # Run optimization when button is clicked
    if run_button:
        # Clear preview when optimization starts
        preview_placeholder.empty()
        
        # Get selected function and gradient
        try:
            if func_mode == "Preset Functions":
                f, grad = FUNCTIONS[func_name]
                display_name = func_name
            else:
                # Create custom function
                if not custom_func_str or custom_func_str.strip() == "":
                    st.error("Please enter a custom function.")
                    st.stop()
                f, grad = create_function_from_string(custom_func_str.strip())
                display_name = func_display_name if func_display_name else f"Custom: {custom_func_str}"
        except Exception as e:
            st.error(f"Error creating function: {str(e)}")
            st.info("**Tips for custom functions:**\n"
                   "- Use `x` as the variable name\n"
                   "- Use `**` for exponentiation (e.g., `x**2`)\n"
                   "- Use `*` for multiplication (e.g., `3*x`)\n"
                   "- Available functions: `sin`, `cos`, `exp`, `log`, `abs`, `sqrt`, etc.\n"
                   "- Example: `x**4 - 3*x**2 + 2`")
            st.stop()
        
        # Run optimization
        try:
            if animation_enabled:
                # Live animation mode
                plot_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                xs, fs = run_optimization_live(
                    f, grad, opt_name, x0, lr, steps,
                    display_name, plot_placeholder, metrics_placeholder,
                    animation_speed
                )
                
                # Final summary after animation
                st.subheader("Optimization Results")
                
                # Create DataFrame for display
                import pandas as pd
                results_data = {
                    'Iteration': list(range(len(xs))),
                    'x_t': xs,
                    'f(x_t)': fs
                }
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial f(x)", f"{fs[0]:.4f}")
                with col2:
                    st.metric("Final f(x)", f"{fs[-1]:.4f}")
                with col3:
                    improvement = fs[0] - fs[-1]
                    st.metric("Improvement", f"{improvement:.4f}")
                
                # Warning if optimization diverged
                if len(xs) < steps + 1:
                    st.warning("Optimization diverged or encountered numerical issues. Try reducing the learning rate.")
            else:
                # Static mode (original behavior)
                xs, fs = run_optimization(f, grad, opt_name, x0, lr, steps)
                
                # Create plot
                fig = plot_optimization(f, xs, fs, display_name, opt_name)
                st.pyplot(fig)
                
                # Display results table
                st.subheader("Optimization Results")
                
                # Create DataFrame for display
                import pandas as pd
                results_data = {
                    'Iteration': list(range(len(xs))),
                    'x_t': xs,
                    'f(x_t)': fs
                }
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial f(x)", f"{fs[0]:.4f}")
                with col2:
                    st.metric("Final f(x)", f"{fs[-1]:.4f}")
                with col3:
                    improvement = fs[0] - fs[-1]
                    st.metric("Improvement", f"{improvement:.4f}")
                
                # Warning if optimization diverged
                if len(xs) < steps + 1:
                    st.warning("Optimization diverged or encountered numerical issues. Try reducing the learning rate.")
            
        except Exception as e:
            st.error(f"Error running optimization: {str(e)}")
            st.exception(e)
    else:
        # Show preview plots when not running optimization
        # Show preview for preset functions
        if func_mode == "Preset Functions":
            try:
                f_preview, _ = FUNCTIONS[func_name]
                fig_preview = plot_function_only(f_preview, func_name)
                preview_placeholder.pyplot(fig_preview)
                plt.close(fig_preview)
            except:
                preview_placeholder.empty()
        # Show preview for custom functions if Generate Graph was clicked
        elif func_mode == "Custom Function" and st.session_state.get('show_custom_preview', False):
            custom_func_to_plot = st.session_state.get('custom_func_str', 'x**2 + 1')
            try:
                f_preview, _ = create_function_from_string(custom_func_to_plot.strip())
                x_sym = Symbol('x')
                expr = sp.sympify(custom_func_to_plot.strip(), evaluate=False)
                latex_str = sp.latex(expr)
                fig_preview = plot_function_only(f_preview, f"Custom: f(x) = ${latex_str}$")
                preview_placeholder.pyplot(fig_preview)
                plt.close(fig_preview)
            except Exception as e:
                preview_placeholder.empty()
                st.error(f"Error plotting function: {str(e)}")
        else:
            preview_placeholder.empty()
            st.info("Adjust the parameters in the sidebar and click 'Run Optimization' to start!")

if __name__ == "__main__":
    main()

