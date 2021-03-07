import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

class Draw:
    """Include some Drawing functions.
    """
    def plot_using_a_map(self, map, legend_name: str):
        """Plot using a Python dict.

        @param map: the key of map is drawn as x-axis, the value as y-axis.
        @param legend_name: label added to the plot.
        """
        x_axis, y_axis = zip(*sorted(map.items()))
        plt.plot(x_axis,
                 y_axis,
                 label=legend_name)
