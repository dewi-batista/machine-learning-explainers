import plotly.graph_objects as go  # type: ignore
import numpy as np

x_1 = np.array([25, 35, 45, 55, 65])
x_2 = np.array([22.5, 26.5, 29.5, 31.5, 33.5])
y = np.array([180, 210, 240, 260, 280])

def f(t_0, t_1, t_2, x_1, x_2, y):
    """
    Compute the loss function over the grid (vectorized version).
    t_1 and t_2 are expected to be 2D arrays from meshgrid.
    """
    # Reshape x_1 and x_2 for broadcasting
    x_1 = x_1.reshape(-1, 1, 1)  # Shape (5, 1, 1)
    x_2 = x_2.reshape(-1, 1, 1)  # Shape (5, 1, 1)
    y = y.reshape(-1, 1, 1)      # Shape (5, 1, 1)

    # Compute residuals and sum squared errors across all data points
    residuals = y - t_0 - (t_1 * x_1) - (t_2 * x_2)
    return np.sum(residuals**2, axis=0)

# Define the range of t_1 and t_2
t_1 = np.linspace(-100, 100, 1000)
t_2 = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(t_1, t_2)

# Compute function values
Z = f(25.68, X, Y, x_1, x_2, y)  # Note: Passing X and Y, not t_1, t_2

t1_star = 0.94
t2_star = 5.79
z_star = f(25.68, np.array([[t1_star]]), np.array([[t2_star]]), x_1, x_2, y)[0, 0]
z_star -= Z.min()
z_star /= (Z.max() - Z.min())

Z = (Z - Z.min()) / (Z.max() - Z.min())

# Plot with Plotly
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

# Extend axis range explicitly if needed
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-100, 100]),
        yaxis=dict(range=[-100, 100]),
        zaxis=dict(range=[Z.min() - 0.1, Z.max() + 0.1])
    )
)

fig.add_trace(go.Scatter3d(
    x=[t1_star], y=[t2_star], z=[z_star],
    mode='markers',
    marker=dict(size=6, color='red', symbol='diamond'),
    name="Selected Point"
))

print(z_star)

fig.show()
