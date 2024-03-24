import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_x_shaped_heatmap(q_table, nrows, ncols, cmap='viridis'):
    """
    Plots a grid where each square is divided by an 'X'. Each quadrant is colored according
    to Q-values. Q-value texts are correctly positioned within their respective sections.
    """
    # Setup for heatmap coloring
    norm = Normalize(vmin=np.min(q_table), vmax=np.max(q_table))
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(ncols, nrows))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()

    for state in range(len(q_table)):
        q_values = q_table[state]
        row, col = divmod(state, ncols)

        # Correct positions for Q-value text annotations
        text_positions = {
            0: (col + 0.2, row + 0.5),  # Left
            1: (col + 0.5, row + 0.8),  # Down
            2: (col + 0.8, row + 0.5),  # Right
            3: (col + 0.5, row + 0.2),  # Up
        }

        # Colors for each quadrant based on Q-values
        colors = [mapper.to_rgba(q_value) for q_value in q_values]

        # Plotting each quadrant with colored patches
        quadrants = [
            [(col, row), (col + 0.5, row + 0.5), (col, row + 1)], # Left
            [(col, row + 1), (col + 0.5, row + 0.5), (col + 1, row + 1)], # Down
            [(col + 1, row), (col + 0.5, row + 0.5), (col + 1, row + 1)], # Right
            [(col, row), (col + 0.5, row + 0.5), (col + 1, row)], # Up
        ]

        for idx, vertices in enumerate(quadrants):
            poly = Polygon(vertices, color=colors[idx], ec='k')
            ax.add_patch(poly)
            tx, ty = text_positions[idx]
            ax.text(tx, ty, f'{q_values[idx]:.2f}', ha='center', va='center', color='white', fontsize=9)

    # Draw grid lines
    ax.set_xticks(np.arange(ncols + 1))
    ax.set_yticks(np.arange(nrows + 1))
    ax.grid(which='major', color='k', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    plt.show()

# Example usage
nrows, ncols = 4, 4  # Grid dimensions
q_table = np.random.rand(nrows * ncols, 4) * 10 - 5  # Example Q-table with varied values
plot_x_shaped_heatmap(q_table, nrows, ncols)
