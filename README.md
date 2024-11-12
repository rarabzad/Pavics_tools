# Example: Downloading Climate Data for All Models and Scenarios
This example script automates the process of downloading climate data from the ESPO-G6-R2 dataset across all available model names and scenarios. It uses the ESPO_G6_R2_Downloader function to retrieve data for a specified region (defined by a shapefile) and saves the output in a structured format.

# Prerequisites
Ensure the following Python libraries are installed:

geopandas for spatial data handling
xarray for working with NetCDF files
numpy for numerical operations
siphon for accessing THREDDS Data Server (TDS) catalogs
re for regular expressions
python
Copy code
import geopandas as gpd
import xarray as xr
import numpy as np
from siphon.catalog import TDSCatalog
import re

# Workflow
## Access the TDS Catalog:

Define the URL of the TDS catalog hosting the ESPO-G6-R2 dataset.
Access the catalog using TDSCatalog from the siphon library and retrieve the dataset names.
Extract Model Names and Scenarios:

Using regular expressions, parse the dataset names to identify all unique model names and scenarios.
Example:
Model names: ["CanESM5", "MPI-ESM1-2-HR", ...]
Scenarios: ["ssp126", "ssp245", "ssp585", ...]
Apply the Downloader Function for Each Model-Scenario Combination:

Define the path to the shapefile representing the region of interest.
For each model name and scenario, call the ESPO_G6_R2_Downloader function, which downloads and saves climate data for that specific model and scenario.
Example Code
python
Copy code
# URL for the TDS catalog
url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"

# Access the catalog
cat = TDSCatalog(url)

# Retrieve dataset names (these will be strings, not objects)
datasets = [dataset for dataset in cat.datasets]

# Extract model names and scenarios using regular expressions
all_model_names = list(set(re.search(r'ScenarioMIP_(.*?)_ssp', dataset).group(1) for dataset in datasets if re.search(r'ScenarioMIP_(.*?)_ssp', dataset)))
all_scenarios = list(set(re.search(r'(ssp\d+)', dataset).group(1) for dataset in datasets if re.search(r'(ssp\d+)', dataset)))

# Define the path to the shapefile
shapefile_path = 'outline.shp'  # Replace with the path to your shapefile

# Loop through each model name and scenario, applying the downloader function
for model_name in all_model_names:
    for scenario in all_scenarios:
        ESPO_G6_R2_Downloader(shapefile_path, model_name, scenario)
Explanation of Key Steps
TDS Catalog Access: Connects to the specified TDS catalog URL and lists all datasets available in the ESPO-G6-R2v1.0.0 catalog.
Model and Scenario Extraction: Uses regular expressions to dynamically gather all model names and scenarios, allowing the script to adapt if additional models or scenarios are added to the catalog.
Downloader Function Application: Iterates through each model-scenario pair and runs ESPO_G6_R2_Downloader with the specified shapefile, model name, and scenario. This creates a NetCDF file for each pair in the specified region.
Output
The output consists of multiple NetCDF files, each containing climate variables (tasmin, tasmax, prcp) for a specific model and scenario combination. The files are named as Raven_input_<model_name>_<scenario>.nc.
