## Overview

This script automates the process of running a Raven hydrological model with climate data from the ESPO-G6-R2 dataset. The steps include:

1. Setting up the environment and importing necessary libraries.
2. Downloading the required ESPO-G6-R2 data from the PaVICS system.
3. Running the Raven model for the specified climate model and scenario over the period 1950–2100.

## Prerequisites

Before running this script, ensure the following:

- A functional [Raven model](https://raven.uwaterloo.ca/)
- Name of a climate model and scenario. See [here](https://github.com/Ouranosinc/ESPO-G)
- A landscape discretization shapefile: Ideally derived using [Basinmaker](https://hydrology.uwaterloo.ca/basinmaker/), this file is required to calculate grid weights using workflow from [here](https://github.com/julemai/GridWeightsGenerator/tree/main).
- Access to the [PaVICS] (https://pavics.ouranos.ca/) platform.

## Code

```python
import os
import zipfile
import requests
# Set environment variable
os.environ["USE_PYGEOS"] = "0"

# Download and extract test data
with open("test.zip", "wb") as file:
    file.write(requests.get("https://github.com/rarabzad/Pavics_tools/raw/refs/heads/main/apply_ESPO_G6_R2/test.zip").content)

with zipfile.ZipFile("test.zip", 'r') as zip_ref:
    zip_ref.extractall("test")
os.remove("test.zip")  # Clean up the ZIP file

# Define file paths and climate model name/scenario
hrufile_path = os.path.join(os.getcwd(), "test/hru/finalcat_hru_info.shp")
Raven_model_dir = os.path.join(os.getcwd(), "test/model")
model_name = "TaiESM"
scenario = "ssp370"

# Run the ESPO-G6-R2 application
exec(requests.get("https://raw.githubusercontent.com/rarabzad/Pavics_tools/refs/heads/main/apply_ESPO_G6_R2/apply_ESPO_G6_R2.py").text)
```

## Notes

- Replace `model_name` and `scenario` variables with desired climate model and scenario names.
- Ensure access to the ESPO-G6-R2 and PaVICS systems for downloading data.

## References

- [PaVICS Platform](https://pavics.ouranos.ca/)
- [ESPO-G6-R2 GitHub Repository](https://github.com/Ouranosinc/ESPO-G)

## outputs

The script produces outputs tailored for the Raven hydrological model, based on the `*.rvi` files work orders. By default all generated results are saved in the `[Raven_model_dir]/output`.
