import bokeh.io as bio
import bokeh.plotting as bk
import pandas as pd
import seaborn as sns
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from clfgraph.sklearn_baseline.models import wandb_init

import wandb

wandb_init()

def plot_distributions(df):
    sns.set(style="whitegrid")
    plt = sns.pairplot(df, hue="target", palette="Set2", diag_kind="kde")
    plt.fig.suptitle("Pairplot of Features Colored by Target", y=1.02)
    plt.savefig("pairplot.png")
    wandb.log({"Pairplot": wandb.Image("pairplot.png")})

def plot_heatmap(df):
    corr = df.corr()
    plt = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.set_title("Correlation Heatmap")
    plt.figure.savefig("heatmap.png")
    wandb.log({"Correlation Heatmap": wandb.Image("heatmap.png")})

def plot_bokeh_scatter(df, x_col, y_col):
    source = ColumnDataSource(df)
    hover = HoverTool(tooltips=[("Index", "$index"), (x_col, f"@{x_col}"), (y_col, f"@{y_col}")])
    p = bk.figure(title=f"{x_col} vs {y_col}", x_axis_label=x_col, y_axis_label=y_col, tools=[hover, "pan", "wheel_zoom", "box_zoom", "reset"])
    p.circle(x_col, y_col, size=10, source=source, color="navy", alpha=0.5)
    bio.output_file("scatter.html")
    bio.save(p)
    wandb.log({"Scatter Plot": wandb.Html("scatter.html")})

def plot_bokeh_histogram(df, col):
    hist, edges = np.histogram(df[col], bins=50)
    p = bk.figure(title=f"Histogram of {col}", x_axis_label=col, y_axis_label='Count')
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.7)
    bio.output_file("histogram.html")
    bio.save(p)
    wandb.log({"Histogram": wandb.Html("histogram.html")})

def plot_bokeh_grid(df, cols):
    plots = []
    for i in range(0, len(cols), 2):
        if i+1 < len(cols):
            p1 = bk.figure(title=f"{cols[i]} vs {cols[i+1]}")
            p1.circle(df[cols[i]], df[cols[i+1]], size=10, color="green", alpha=0.5)
            p2 = bk.figure(title=f"Histogram of {cols[i]}")
            hist, edges = np.histogram(df[cols[i]], bins=50)
            p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", alpha=0.7)
            plots.append([p1, p2])
    grid = gridplot(plots, toolbar_location="right")
    bio.output_file("gridplot.html")
    bio.save(grid)
    wandb.log({"Grid Plot": wandb.Html("gridplot.html")})

def visualize_all(df):
    plot_distributions(df)
    plot_heatmap(df)
    for col in df.columns[:-1]:
        plot_bokeh_scatter(df, col, df.columns[-1])
        plot_bokeh_histogram(df, col)
    plot_bokeh_grid(df, df.columns[:-1])

df = pd.read_csv(wandb.use_artifact('project/dataset:v0').file())
visualize_all(df)
wandb.finish()