import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import rioxarray as riox
from src import paths


def plot_raster_detections(
    detections_dir=paths.data_processed_dir("raster_detections"),
    out_dir=paths.reports_figures_dir("raster_detections"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    binary_cmap = mcolors.ListedColormap(["lightgreen", "salmon"])
    binary_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], binary_cmap.N)

    for raster_path in sorted(detections_dir.glob("*.tif")):
        img = riox.open_rasterio(raster_path).squeeze(drop=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        if raster_path.name.endswith("_detection_dates.tif"):
            dates = img.where(img >= 0)
            plot = ax.imshow(dates, cmap="viridis")
            fig.colorbar(plot, label="Detection date (days since reference date)")
        else:
            ax.imshow(img, cmap=binary_cmap, norm=binary_norm)
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgreen", markersize=10, label="No change"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="salmon", markersize=10, label="Change"),
            ]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), title="Change detection")
        ax.set_axis_off()
        ax.set_title(raster_path.stem)
        fig.savefig(out_dir / raster_path.with_suffix(".png").name, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    plot_raster_detections()
