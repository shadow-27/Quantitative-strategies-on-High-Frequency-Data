# lets define a function to plot heatmaps
# to visualize results
# of our strategy for different parameters

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmap(df, value_col, index_col, columns_col, title, cmap="RdYlGn", figsize=(10, 6)):
    """
    Creates a heatmap based on a dataframe.

    Arguments:
    - df: pandas dataframe
    - value_col: name of the column with values (str)
    - index_col: name of the column for the Y axis (str)
    - columns_col: name of the column for the X axis (str)
    - title: title of the plot (str)
    """

    # Creating pivot table
    heatmap_data = df.pivot_table(
        values=value_col,
        index=index_col,
        columns=columns_col
    )

    # Symmetric color scaling around zero
    vmax = np.nanmax(heatmap_data.values)
    vmin = np.nanmin(heatmap_data.values)

    # Drawing the plot
    plt.figure(figsize=figsize)
    plt.title(title)

    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=vmin, vmax=vmax,
        mask=heatmap_data.isnull(),  # hide where NaN
        linewidths=0.5
    )

    plt.xlabel(columns_col)
    plt.ylabel(index_col)
    plt.tight_layout()
    plt.show()