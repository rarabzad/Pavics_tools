```markdown
## Step 1: Climate Model Processing Script

This script automates the process of downloading climate models, extracting necessary data, and performing various operations related to hydrological modeling. Below is the step-by-step breakdown of the code.

```python
# Clean up global variables to avoid interference from Python internal variables
for name in list(globals().keys()):
    if name[0] != "_":  # Avoid deleting internal Python variables
        del globals()[name]
```

## Step 2: Import Necessary Libraries

```python
import requests, zipfile, re, os, shutil, cftime, ravenpy
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from siphon.catalog import TDSCatalog
from netCDF4 import Dataset
```

## Step 3: Extract Example Models

This section handles downloading and extracting zip files for the models used in the climate analysis.

```python
# URL for the first grid of models
url = "https://github.com/rarabzad/Pavics_tools/raw/refs/heads/main/SMM_grids_hrus.zip"

# Download the zip file and extract it
with open("SMM_grids_hrus.zip", "wb") as f:
    f.write(requests.get(url).content)

# Extract the zip contents
with zipfile.ZipFile("SMM_grids_hrus.zip", "r") as zip_ref:
    zip_ref.extractall()

# Remove the downloaded zip file
os.remove("SMM_grids_hrus.zip")

# URL for the models
url = "https://github.com/rarabzad/Pavics_tools/raw/refs/heads/main/SMM_models.zip"

# Download and extract the models zip file
with open("SMM_models.zip", "wb") as f:
    f.write(requests.get(url).content)

with zipfile.ZipFile("SMM_models.zip", "r") as zip_ref:
    zip_ref.extractall()

# Remove the zip file after extraction
os.remove("SMM_models.zip")
```

## Step 4: Copy CC Forcing Grid Weights File

The following block copies the grid weights required for the climate models into the appropriate directories for two models.

```python
# Copy the GridWeights.txt file to the appropriate directories
shutil.copy('SMM_grids_hrus/GridWeights.txt', '/notebook_dir/writable-workspace/SMM_models/milk/input/')
shutil.copy('SMM_grids_hrus/GridWeights.txt', '/notebook_dir/writable-workspace/SMM_models/stmary/input/')
```

## Step 5: Download and Process Climate Change Data

The next part of the script fetches the necessary climate data based on a specified URL and processes it for further use.

```python
# Path to the HRU (Hydrological Response Unit) shapefile
hrufile_path = "SMM_grids_hrus/hru.shp"

# Load the function for downloading climate data from a remote source
exec(requests.get("https://raw.githubusercontent.com/rarabzad/Pavics_tools/refs/heads/main/ESPO_G6_R2_Downloader.py").text)

# URL for accessing the TDS catalog of climate simulations
url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"

# Access the catalog and retrieve the list of datasets
cat = TDSCatalog(url)
datasets = [dataset for dataset in cat.datasets]  # Extract the dataset names as strings
```

## Step 6: Helper Function for Replacing Forcing File Names

This function replaces the default forcing file name in configuration files to use the specific model forcing data.

```python
def replace_filename(file_path, old, new):
    """
    Replace the old forcing file name with a new one in a given file.
    """
    with open(file_path, 'r+') as file:
        content = file.read().replace(old, new)
        file.seek(0)
        file.write(content)
```

## Step 7: Iterate Over Climate Models and Update Forcing Data

In this section, the script loops through each climate model dataset, downloads the necessary files, and adjusts configuration files accordingly.

```python
# Initial file names and parameters for the forcing data
old_forcing = 'RavenInput.nc'
old_start_date = 'start_date'
old_end_date = 'end_date'

# Create directories to store the output data
os.mkdir('hydrographs')
os.mkdir('climateForcings')

# Loop through all the datasets
for file in datasets:
    # Extract model name and scenario from the dataset name
    start_index = file.find("ScenarioMIP_NAM_") + len("ScenarioMIP_NAM_")
    end_index = file.find("_ssp")
    model_name = file[start_index:end_index]
    scenario = file[end_index + 1:end_index + 7]

    # Download the climate data using the provided downloader function
    ESPO_G6_R2_Downloader(hrufile_path, model_name, scenario)

    # Define the new forcing file name
    new_forcing = f"Raven_input_{model_name}_{scenario}.nc"

    # Replace old forcing file names in the configuration files
    replace_filename('/notebook_dir/writable-workspace/SMM_models/milk/milk.rvt', old_forcing, new_forcing)
    replace_filename('/notebook_dir/writable-workspace/SMM_models/stmary/stmary.rvt', old_forcing, new_forcing)

    # Copy the new forcing file to the appropriate directories
    shutil.copy(new_forcing, os.path.join('/notebook_dir/writable-workspace/SMM_models/stmary/input/', new_forcing))
    shutil.copy(new_forcing, os.path.join('/notebook_dir/writable-workspace/SMM_models/milk/input/', new_forcing))

    # Open the forcing file to check the calendar type
    nc_file = Dataset(new_forcing, 'r')
    calendar_type = nc_file['time'].getncattr('calendar')

    # Skip processing if the calendar is not 'noleap'
    if calendar_type != 'noleap':
        print(f"Skipping iteration {new_forcing} because calendar is not 'noleap'. Current calendar: {calendar_type}")
        continue

    # Extract the time information from the forcing file
    times = cftime.num2date(nc_file.variables['time'][:], units=nc_file.variables['time'].units)
    new_start_date = times[1].strftime('%Y-%m-%d') + " 00:00:00"
    new_end_date = times[-2].strftime('%Y-%m-%d') + " 00:00:00"
    nc_file.close()

    # Replace the start and end dates in the configuration files
    replace_filename('/notebook_dir/writable-workspace/SMM_models/milk/milk.rvi', old_start_date, new_start_date)
    replace_filename('/notebook_dir/writable-workspace/SMM_models/stmary/stmary.rvi', old_start_date, new_start_date)
    replace_filename('/notebook_dir/writable-workspace/SMM_models/milk/milk.rvi', old_end_date, new_end_date)
    replace_filename('/notebook_dir/writable-workspace/SMM_models/stmary/stmary.rvi', old_end_date, new_end_date)

    # Run the Raven hydrological model for both "milk" and "stmary" scenarios
    ravenpy.run(modelname='milk', configdir='/notebook_dir/writable-workspace/SMM_models/milk/')
    ravenpy.run(modelname='stmary', configdir='/notebook_dir/writable-workspace/SMM_models/stmary/')

    # Copy the generated hydrographs to the output directory
    source_file = '/notebook_dir/writable-workspace/SMM_models/milk/output/Hydrographs.csv'
    shutil.copy(source_file, os.path.join('hydrographs', f"milk_{os.path.splitext(new_forcing.replace('Raven_input_', ''))[0]}.csv"))

    source_file = '/notebook_dir/writable-workspace/SMM_models/stmary/output/Hydrographs.csv'
    shutil.copy(source_file, os.path.join('hydrographs', f"stmary_{os.path.splitext(new_forcing.replace('Raven_input_', ''))[0]}.csv"))

    # Clean up by removing the forcing files after use
    os.remove(os.path.join('/notebook_dir/writable-workspace/SMM_models/stmary/input/', new_forcing))
    os.remove(os.path.join('/notebook_dir/writable-workspace/SMM_models/milk/input/', new_forcing))

    # Update the old forcing and date variables for the next iteration
    old_forcing = new_forcing
    old_start_date = new_start_date
    old_end_date = new_end_date

    # Move the processed forcing file to the climateForcings directory
    shutil.move(new_forcing, 'climateForcings')
```

## Step 8: Archive Hydrographs

Finally, after all iterations are complete, the generated hydrographs are archived into a ZIP file.

```python
# Create a ZIP archive of the hydrographs
shutil.make_archive('hydrographs.zip', 'zip', 'hydrographs')
```

### Jupyter Notebook file

You can use the Jupyter Notebook file archived in this repository to reproduce the entire process.
