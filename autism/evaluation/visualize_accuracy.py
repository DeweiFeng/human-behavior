import plotly.express as px
import plotly.io as pio
import pandas as pd
import os

def plot_segment_accuracy(by_segment: pd.Series, output_dir: str, filetype: str = "png"):
    fig = px.bar(
        by_segment.reset_index().sort_values('correct', ascending=False),
        x='segment', y='correct',
        title='Accuracy per Test Segment',
        labels={'correct': 'Accuracy', 'segment': 'Test Segment'}
    )
    save_plot(fig, output_dir, "segment_accuracy", filetype)

def plot_module_accuracy(by_module: pd.Series, output_dir: str, filetype: str = "png"):
    fig = px.bar(
        by_module.reset_index().sort_values('correct', ascending=False),
        x='module', y='correct',
        title='Accuracy per Module',
        labels={'correct': 'Accuracy', 'module': 'Module'}
    )
    save_plot(fig, output_dir, "module_accuracy", filetype)


def save_plot(fig, output_dir: str, name: str, filetype: str = "png"):
    os.makedirs(output_dir, exist_ok=True)
    filetype = filetype.lower()
    out_path = os.path.join(output_dir, f"{name}.{filetype}")

    if filetype == "html":
        fig.write_html(out_path)
    elif filetype in {"png", "pdf", "svg", "eps", "jpeg", "jpg", "webp"}:
        # Workaround: use plotly.io directly and specify Kaleido engine
        pio.write_image(fig, out_path, format=filetype, engine="kaleido")
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

