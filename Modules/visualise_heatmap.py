import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# === CONFIGURATION ===
csv_path = 'Outputs/Mangifera_indica_probability_map.csv'  # Update as needed
output_png = os.path.splitext(csv_path)[0] + '_viz.png'

# Load the data
print(f"Loading: {csv_path}")
df = pd.read_csv(csv_path)

# Set up dark background
plt.style.use('dark_background')

# Define custom color map (gradient from red to blue)
colors = [
    '#FF0000',  # 0.0 - red
    '#FF4500',  # 0.1
    '#FF8C00',  # 0.2
    '#FFD700',  # 0.3
    '#ADFF2F',  # 0.4
    '#00CED1',  # 0.5
    '#4169E1',  # 0.6 - royal blue
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_prob_cmap", colors)

# Plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    df['longitude'], df['latitude'],
    c=df['probability'], cmap=cmap,
    s=2, marker='s', edgecolor='none',
    vmin=0, vmax=1, alpha=0.8  # alpha to reduce brightness
)
cbar = plt.colorbar(sc, label='Probability')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

plt.xlabel('Longitude', color='white')
plt.ylabel('Latitude', color='white')
plt.title('Species Distribution Probability Map', color='white')

plt.grid(False)
plt.tight_layout()
plt.savefig(output_png, dpi=300, facecolor='black')
print(f"✅ Saved visualization to: {output_png}")
plt.show()
