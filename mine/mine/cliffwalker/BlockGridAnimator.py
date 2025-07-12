import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

class BlockGridAnimator:
    def __init__(self, data):
        """
        Initialize the animator.

        Parameters:
        - data: numpy array of shape (num_frames, rows, cols, 4)
                Each 4-tuple represents [up, right, down, left] values per block
        """
        self.data = np.array(data)
        assert self.data.ndim == 4 and self.data.shape[3] == 4, "Invalid data shape."

        self.num_frames, self.rows, self.cols, _ = self.data.shape
        self.triangles = []

        # Compute log-scale normalized values (symmetric around 0)
        safe_data = np.where(self.data == 0, 1e-6, self.data)
        log_vals = np.log10(np.abs(safe_data)) * np.sign(safe_data)

        max_abs = np.max(np.abs(log_vals))
        self.vmin = -max_abs
        self.vmax = +max_abs

        # Plot initialization
        self.fig, self.ax = plt.subplots(figsize=(self.cols * 1.2 / 2, self.rows * 1.2 / 2))
        self.ax.set_xlim(0, self.cols)
        self.ax.set_ylim(0, self.rows)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.invert_yaxis()

        self._init_patches()

        # Add frame number display
        self.frame_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes,
                                       fontsize=12, color='black', ha='left', va='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def _value_to_color(self, val):
        """
        Convert a value to a color between red (negative) and green (positive),
        using a symmetric logarithmic scale.
        """
        if val == 0:
            norm_val = 0.5
        else:
            log_val = np.log10(abs(val) + 1e-6) * np.sign(val)
            norm_val = (log_val - self.vmin) / (self.vmax - self.vmin)

        norm_val = np.clip(norm_val, 0, 1)
        r = 1 - norm_val
        g = norm_val
        return (r, g, 0)

    def _init_patches(self):
        for i in range(self.rows):
            for j in range(self.cols):
                x0, y0 = j, i
                cx, cy = x0 + 0.5, y0 + 0.5

                tl = (x0, y0)
                tr = (x0 + 1, y0)
                br = (x0 + 1, y0 + 1)
                bl = (x0, y0 + 1)
                c = (cx, cy)

                up = Polygon([tl, tr, c])
                right = Polygon([tr, br, c])
                down = Polygon([bl, br, c])
                left = Polygon([tl, bl, c])

                for tri in [up, right, down, left]:
                    self.ax.add_patch(tri)

                self.triangles.append((up, right, down, left))

    def _update(self, frame):
        for idx, (i, j) in enumerate(np.ndindex(self.rows, self.cols)):
            up, right, down, left = self.triangles[idx]
            values = self.data[frame, i, j]
            up.set_facecolor(self._value_to_color(values[0]))
            right.set_facecolor(self._value_to_color(values[1]))
            down.set_facecolor(self._value_to_color(values[2]))
            left.set_facecolor(self._value_to_color(values[3]))

        self.frame_text.set_text(f"Frame: {frame + 1}/{self.num_frames}")
        return sum(self.triangles, ()) + (self.frame_text,)

    def show(self, interval=300):
        self.ani = FuncAnimation(self.fig, self._update, frames=self.num_frames,
                                 interval=interval, blit=False)
        plt.show()

    def save(self, filename="output.gif", fps=2):
        self.ani = FuncAnimation(self.fig, self._update, frames=self.num_frames,
                                 interval=1000 / fps, blit=False)
        self.ani.save(filename, writer=PillowWriter(fps=fps))
        print(f"✅ GIF saved as: {filename}")
