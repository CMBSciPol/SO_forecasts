
# Sigma(r) Forecast vs Observation Time

This script estimates the uncertainty on the tensor-to-scalar ratio, \( \sigma(r) \), 
as a function e.g. of total observation time for a CMB experiment like Simons Observatory.

## Features

- Converts detector-level NETs and observing time into per-band noise levels (in μK·arcmin).
- Models noise power spectra including low-ℓ atmospheric-like excess (\( 1/\ell^\alpha \)).
- Combines frequency bands using a mixing matrix with realistic foreground components (CMB, dust, synchrotron).
- Computes \( \sigma(r) \) via a Fisher matrix approach, marginalizing over \( A_{\text{lens}} \).
- Allows tuning of:
  - Observation duration
  - Delensing efficiency
  - Detector configuration and noise models

## Requirements

- Python 3.x
- `camb`
- `fgbuster`
- `numpy`, `matplotlib`, `scipy`

Install dependencies:
```bash
pip install camb fgbuster numpy matplotlib scipy
```

## Usage

To run the forecast:
```bash
python fisher_sigma_r.py
```

## Output

A plot of \( \sigma(r) \) versus total observation time is displayed, with a logarithmic y-axis.

## Author

Developed by [SciPol project](https://scipol.in2p3.fr), CNRS / APC

For questions, contact: josquin@apc.in2p3.fr

