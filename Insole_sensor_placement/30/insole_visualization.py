import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from PIL import Image
import matplotlib.cm as cm

# Load the background image
bg_image = Image.open("background.png")

# Load the data and coordinates
data_df = pd.read_csv("data.csv")
coord_df = pd.read_csv("coordinates.csv").iloc[0:17]  # Skip the header

# Set up the plot
fig, ax = plt.subplots()
ax.imshow(bg_image, extent=[0, 640, 0, 640])

# Create a colormap
cmap = cm.get_cmap('jet')

# Create circles for each area
circles = []
for _, row in coord_df.iterrows():
    circle = Circle((row['x'], row['y']), 30, color='red', alpha=0.5)
    ax.add_patch(circle)
    circles.append(circle)

# Update function for animation
def update(frame):
    for i, circle in enumerate(circles):
        # Map the data value to a color
        value = data_df.iloc[frame, i + 1] / 1024  # Normalize to 0-1
        color = cmap(value)
        circle.set_color(color)
    return circles

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data_df), interval=500, blit=True)

plt.show()
