# Pavics RDRS: Climate Data Processing and Grid Weights Generation

This repository contains a set of tools to process climate data and generate grid weight files based on a given HRU (Hydrological Response Unit) shapefile. The `pavics_rdrs.py` script facilitates the extraction of climate data from the PAVICS database and applies spatial analysis to compute grid weights, which can be used in hydrological models. The code is optimized for climate data, including precipitation, temperature, and elevation.

## Overview

The workflow begins by downloading and unzipping an HRU shapefile, followed by the execution of climate data processing functions. The core steps of the process are:

1. **Data Extraction**: Retrieve climate data (temperature, precipitation) from the PAVICS database.
2. **Spatial Analysis**: Use the HRU shapefile to extract data for the specific region defined by the shapefile.
3. **Grid Weight Generation**: Generate grid weights using the specified HRU shapefile.
4. **NetCDF Creation**: The script outputs processed climate data as a NetCDF file for use in further analysis or modeling.
5. **Raven Model File Generation**: The script generates a Raven model input file (.rvt) for use with the [Raven hydrological model](https://raven.uwaterloo.ca/).

### Prerequisites

Before using the script, make sure you have the following:

- **Access to PAVICS Platform**: You must have access to the PAVICS platform, which provides climate data through its virtual laboratory. You can access PAVICS at [https://pavics.ouranos.ca/](https://pavics.ouranos.ca/).
- **HRU Shapefile**: You need an HRU shapefile that can be generated using the [BasinMaker tool](https://hydrology.uwaterloo.ca/basinmaker/). The HRU shapefile defines the geographical boundaries for which the climate data will be extracted.

### Dependencies

Before using the script, make sure you have the following Python dependencies installed (Generally Pavics has these installed):

- `xarray`
- `geopandas`
- `numpy`
- `scipy`
- `pandas`
- `tqdm`
- `siphon`
- `urllib`

### Getting Started

Follow these steps to use the workflow to download data, process it, and generate the grid weights.

## Step-by-Step Guide

### 1. Import Libraries and Download External Scripts

The script begins by importing the necessary libraries and downloading an external Python script (`pavics_rdrs.py`) that contains all the essential functions for processing climate data.

```python
import urllib.request, zipfile, os

# Download and execute the script containing functions
exec(urllib.request.urlopen("https://raw.githubusercontent.com/rarabzad/Pavics_tools/refs/heads/main/RDRS/pavics_rdrs.py").read())
```

### 2. Download and Unzip the HRU Shapefile

The next step is to download the HRU shapefile archive (`hru.zip`) from GitHub and unzip it. This shapefile will be used to extract climate data for a specific region.

```python
# Download and unzip the HRU shapefile
urllib.request.urlretrieve("https://github.com/rarabzad/Pavics_tools/raw/refs/heads/main/RDRS/hru.zip", "hru.zip")
with zipfile.ZipFile("hru.zip", 'r') as zip_ref:
    zip_ref.extractall("hru")
os.remove("hru.zip")
```

### 3. Process Climate Data Using HRU Shapefile

After extracting the HRU shapefile, the function `process_climate_data` is called. This function takes the HRU shapefile path as input and processes climate data from the PAVICS database for the specified region.

```python
# Example of calling the function with the HRU shapefile path
shapefile_path = "/notebook_dir/writable-workspace/test/hru/finalcat_hru_info.shp"  # Replace with actual path
process_climate_data(shapefile_path)
```

### 4. Inside the `process_climate_data` Function

The `process_climate_data` function performs several tasks:

1. **Load Data from PAVICS**: The function loads climate data (temperature and precipitation) from the PAVICS database, using the `TDSCatalog` and `xarray` to access the remote data.
2. **Spatial Data Extraction**: The function processes the HRU shapefile (with coordinates in EPSG:4326) to create a bounding box. It then extracts the indices of the grid cells inside the bounding box.
3. **Climate Data Extraction**: Using the grid indices from the HRU shapefile, the function extracts the relevant climate data (e.g., precipitation, temperature) for the region defined by the shapefile.
4. **Elevation Data**: The function retrieves elevation data for the specified region by matching latitude and longitude coordinates from the HRU shapefile with a pre-existing DEM (Digital Elevation Model).
5. **Generate NetCDF File**: The processed data is saved into a NetCDF file (`climate_data.nc`), which includes latitude, longitude, precipitation, temperature, and elevation data.
6. **Grid Weight Generation**: Lastly, the function generates grid weights using the HRU shapefile and stores them in a file named `GridWeights.txt`.
7. **Generate Raven Model File**: The script generates a [Raven model input file](https://raven.uwaterloo.ca/) (.rvt) that includes the climate forcing data (precipitation, temperature) and elevation for use in Raven hydrological simulations.

### Example Execution

Once the HRU shapefile is extracted, and the necessary Python script is executed, you can call the `process_climate_data` function with the path to your HRU shapefile. Here is the full code for running the workflow:

```python
import urllib.request, zipfile, os

# Download and execute the script containing functions
exec(urllib.request.urlopen("https://raw.githubusercontent.com/rarabzad/Pavics_tools/refs/heads/main/RDRS/pavics_rdrs.py").read())

# Download and unzip the HRU shapefile
urllib.request.urlretrieve("https://github.com/rarabzad/Pavics_tools/raw/refs/heads/main/RDRS/hru.zip", "hru.zip")
with zipfile.ZipFile("hru.zip", 'r') as zip_ref:
    zip_ref.extractall("hru")
os.remove("hru.zip")

# Example of calling the function with the HRU shapefile path
shapefile_path = "/notebook_dir/writable-workspace/test/hru/finalcat_hru_info.shp"  # Replace with actual path
process_climate_data(shapefile_path)
```

### 5. Output

After running the script, you will have the following outputs:

1. **NetCDF File**: A file named `climate_data.nc` containing the processed climate data (precipitation, temperature, and elevation).
2. **Grid Weights File**: A file named `GridWeights.txt` containing the grid weights for the HRU regions.
3. **Raven Model File**: A file named `model.rvt` containing the climate forcing data (precipitation, temperature) and elevation, which can be used in the [Raven hydrological model](https://raven.uwaterloo.ca/).

## Additional Information

### PAVICS Database Summary

PAVICS is a virtual laboratory facilitating the analysis of climate data. It provides access to several data collections, including observations, climate projections, and reanalyses. It also offers a Python programming environment to analyze data without the need to download it. The environment is constantly updated with the most efficient libraries for climate data analysis while ensuring quality control on the data and metadata.

For more information, visit: [PAVICS Website](https://pavics.ouranos.ca/)

### Raven Hydrological Model

[Raven](https://raven.uwaterloo.ca/) is a hydrological model developed to support the simulation of streamflow and water balance processes at the catchment scale. The `.rvt` files generated by this script can be used to provide climate forcings for Raven simulations.
