Here’s an RMarkdown (`.Rmd`) file for your Python script, formatted for GitHub. It includes Markdown syntax for explanation, Python code blocks, and a description of the process and purpose.

Here’s the `.Rmd` file content reformatted:

---

```rmarkdown
---
title: "Raven Model Climate Simulation with ESPO-G6-R2"
author: "Your Name"
date: "`r Sys.Date()`"
output: github_document
---

## Overview

This script automates the process of running a Raven hydrological model with climate data from the ESPO-G6-R2 dataset. The steps include:

1. Setting up the environment and importing necessary libraries.
2. Downloading the required ESPO-G6-R2 data from the PaVICS system.
3. Running the Raven model for the specified climate model and scenario over the period 1950–2100.

## Prerequisites

Before running this script, ensure the following:

- Python is installed.
- The necessary libraries (`numpy`, `pandas`, `geopandas`, etc.) are installed.
- Access to the PaVICS platform.

## Code

```python
# Import necessary libraries
import os, re, glob, shutil, time, json, zipfile, urllib.request
from itertools import product
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from pyproj import CRS
import rasterio, rioxarray as rio
import cartopy.crs as ccrs
from siphon.catalog import TDSCatalog
from birdy import WPSClient
import ravenpy
from ravenpy.utilities.testdata import get_file

# Set environment variable
os.environ["USE_PYGEOS"] = "0"

# Load the ESPO-G6-R2 application script
exec(requests.get("https://raw.githubusercontent.com/rarabzad/Pavics_tools/refs/heads/main/apply_ESPO_G6_R2/apply_ESPO_G6_R2.py").text)

# Download and extract test data
with open("test.zip", "wb") as file:
    file.write(requests.get("https://github.com/rarabzad/Pavics_tools/raw/refs/heads/main/apply_ESPO_G6_R2/test.zip").content)

with zipfile.ZipFile("test.zip", 'r') as zip_ref:
    zip_ref.extractall("test")
os.remove("test.zip")  # Clean up the ZIP file

# Define file paths and model parameters
hrufile_path = os.path.join(os.getcwd(), "test/hru/finalcat_hru_info.shp")
Raven_model_dir = os.path.join(os.getcwd(), "test/model")
model_name = "TaiESM"
scenario = "ssp370"

# Run the ESPO-G6-R2 application
apply_ESPO_G6_R2(hrufile_path, Raven_model_dir, model_name, scenario)
```

## Notes

- Replace `model_name` and `scenario` variables with desired climate model and scenario names.
- Ensure access to the ESPO-G6-R2 and PaVICS systems for downloading data.

## References

- [PaVICS Platform](https://pavics.ouranos.ca/)
- [ESPO-G6-R2 GitHub Repository](https://github.com/Ouranosinc/ESPO-G)

## Output

The script generates outputs compatible with the Raven hydrological model, based on the selected climate data and scenario. The results can be used for further hydrological and climate impact analyses.
```

Save this as a `.Rmd` file and render it in RStudio or any Markdown-compatible platform to generate a GitHub-ready document.