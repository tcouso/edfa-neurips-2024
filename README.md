# esn-change-detection

Official implementation of [*Enhanced Detection of Human-Driven Forest Alterations using Echo State Networks*](https://neurips.cc/virtual/2024/100568) (Tomás Couso Coddou, Paula Aguirre, Rodrigo A. Carrasco, Javier Lopatin), NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning.

Central Chile has gone through a decade of severe droughts and fires, with alarming tree mortality. Telling human-driven alterations, such as logging, from natural ones is hard precisely under those conditions, because a drought-stressed forest already looks degraded from space. This work models the vegetation dynamics of every pixel with an Echo State Network (ESN) trained on Landsat NDVI time series, and flags a change when the observed signal sustainably falls below what the network expects.

<p align="center">
  <img src="assets/detection-example.png" width="520" alt="Change detection over the NDVI series of a pixel in a burned polygon">
</p>

<p align="center"><em>Detection over the series of one pixel in a burned polygon. The ESN prediction (blue) follows the denoised signal (orange) until the fire; the signal then leaves the shaded band of <code>k</code> times the pixel's expected error, and the detection (purple) falls inside the annotated event window (red).</em></p>

## Highlights

Reported on the [workshop poster](https://neurips.cc/virtual/2024/100568), over NDVI:

- **0.76 overall accuracy** classifying polygons as changed or not.
- **Fire is detected reliably**, even over drought-affected terrain: 0.86 accuracy over stable terrain and 0.84 over drought terrain.
- Logging is harder (0.67 and 0.69), and is the main room for improvement.
- NDVI and NBR show complementary strengths, with NBR doing better on logging over stable terrain.

| Terrain | Accuracy | F1-score | Recall | Precision |
|---|---|---|---|---|
| Stable, fire | 0.86 | 0.83 | 0.89 | 0.78 |
| Drought, fire | 0.84 | 0.76 | 0.89 | 0.66 |
| Stable, logging | 0.67 | 0.58 | 0.47 | 0.74 |
| Drought, logging | 0.69 | 0.54 | 0.47 | 0.62 |

The method also runs over raster time series, mapping where and when changes are detected:

<p align="center">
  <img src="assets/raster-detections.png" width="720" alt="Change detections over two burned polygons">
</p>

<p align="center"><em>Detections over two burned polygons (black outlines). Left: aerial imagery. Centre: pixels with a detected change, in red. Right: the detection date of each pixel. Maps rendered with QGIS, over a web basemap.</em></p>

The dataset behind these numbers is not public. This repository is the code harness that produces them, with the input format documented and a synthetic dataset to run it.

## How it works

ESNs are a lightweight kind of recurrent network: a fixed sparse random reservoir encodes temporal patterns, and only a linear readout is trained, so fitting is cheap and fast even over thousands of pixels.

<p align="center">
  <img src="assets/esn-architecture.png" width="440" alt="Echo state network architecture">
</p>

<p align="center"><em>Echo state network: a fixed random reservoir, with only the readout <code>W<sup>out</sup></code> trained. The feedback connections <code>W<sup>fb</sup></code> are optional; this code uses the non-feedback network. Figure from <a href="https://ieeexplore.ieee.org/abstract/document/9205191">Cho et al. (2020)</a>.</em></p>

A single ESN is trained on the denoised NDVI series and predicts each pixel's next value. A window of the series is flagged when the signal falls below the prediction by more than `k` times the pixel's RMSE in most of its steps, and a polygon is labelled by voting over its pixels.

## Two versions

The workshop poster is the most recent version of the work, and is the one to start from. The workshop manuscript describes an earlier stage, so each has its own self-contained directory. They share no code and are installed separately.

| | [`poster/`](poster) (most recent) | [`manuscript/`](manuscript) (earlier) |
|---|---|---|
| Corresponds to | [the workshop poster](https://neurips.cc/virtual/2024/100568) | [the workshop manuscript (PDF)](https://ccai-papers.s3.us-east-1.amazonaws.com/neurips2024/77/paper.pdf) |
| Detection | one global ESN; windows where the prediction error exceeds `k` times the RMSE | per-pixel lower bound from an ESN fine-tuned step by step; N consecutive forecasts below `k` times the bound |
| Polygon decision | fraction of pixels detected inside the annotated event window, voting | fraction of flagged pixels against `th` |
| Reservoir / input | 1000 units, 48 values at 16-day sampling | 500 units, 104 weekly values |
| Extras | detection over raster time series | |

Neither version can be reproduced end to end without the private dataset. Each directory documents the input format it expects and includes a generator of synthetic data to exercise the code.

## Acknowledgements

This work was supported by the FONDEF ID21I10102 and FONDECYT 1231245 grants from ANID. Paula Aguirre also acknowledges funding from CENIA.

## References

The change detection scheme adapts the fault detection method of:

A. D. Cho, R. A. Carrasco, G. A. Ruz and J. L. Ortiz, "Slow Degradation Fault Detection in a Harsh Environment," *IEEE Access*, vol. 8, pp. 175904-175920, 2020, doi: [10.1109/ACCESS.2020.3026348](https://doi.org/10.1109/ACCESS.2020.3026348).

## Citation

```bibtex
@inproceedings{couso2024enhanced,
  title     = {Enhanced Detection of Human-Driven Forest Alterations using Echo State Networks},
  author    = {Couso Coddou, Tom{\'a}s and Aguirre, Paula and Carrasco, Rodrigo A. and Lopatin, Javier},
  booktitle = {NeurIPS 2024 Workshop on Tackling Climate Change with Machine Learning},
  year      = {2024},
  url       = {https://neurips.cc/virtual/2024/100568}
}
```
