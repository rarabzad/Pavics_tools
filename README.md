```markdown
## Example: Downloading Climate Data for All Models and Scenarios

This example script automates the process of downloading climate data from the ESPO-G6-R2 dataset across all available model names and scenarios. It uses the `ESPO_G6_R2_Downloader` function (defined in the external script) to retrieve data for a specified region (defined by a shapefile) and saves the output in a structured format.

### Prerequisites

Ensure the following Python libraries are installed:
- `geopandas` for spatial data handling
- `xarray` for working with NetCDF files
- `numpy` for numerical operations
- `siphon` for accessing THREDDS Data Server (TDS) catalogs
- `requests` for downloading the external script
- `re` for for standard data anlyses
- `shapely` for spatial data analyses


### Importing the `ESPO_G6_R2_Downloader` Function

To download and import the `ESPO_G6_R2_Downloader` function, you can use the following code snippet:

```python
import requests

# URL of the external script containing the ESPO_G6_R2_Downloader function
url = "https://raw.githubusercontent.com/rarabzad/Pavics_tools/refs/heads/main/ESPO_G6_R2_Downloader.py?token=GHSAT0AAAAAACXAX7TPLBYTTUQ35L3BSQ7CZZSU2JA"

# Download the script content
response = requests.get(url)
with open("ESPO_G6_R2_Downloader.py", "w") as f:
    f.write(response.text)

# Import the downloader function
from ESPO_G6_R2_Downloader import ESPO_G6_R2_Downloader
```

### Workflow

1. **Access the TDS Catalog**:
   - Define the URL of the TDS catalog hosting the ESPO-G6-R2 dataset.
   - Access the catalog using `TDSCatalog` from the `siphon` library and retrieve the dataset names.

2. **Extract Model Names and Scenarios**:
   - Using regular expressions, parse the dataset names to identify all unique model names and scenarios.

3. **Apply the Downloader Function for Each Model-Scenario Combination**:
   - Define the path to the shapefile representing the region of interest.
   - For each model name and scenario, call the `ESPO_G6_R2_Downloader` function, which downloads and saves climate data for that specific model and scenario.

### Example Code

```python
import requests, gpd, xr, np, re
from siphon.catalog import TDSCatalog
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

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
```

### Explanation of Key Steps

- **TDS Catalog Access**: Connects to the specified TDS catalog URL and lists all datasets available in the ESPO-G6-R2v1.0.0 catalog.
- **Model and Scenario Extraction**: Uses regular expressions to dynamically gather all model names and scenarios, allowing the script to adapt if additional models or scenarios are added to the catalog.
- **Downloader Function Application**: Iterates through each model-scenario pair and runs `ESPO_G6_R2_Downloader` with the specified shapefile, model name, and scenario. This creates a NetCDF file for each pair in the specified region.

### Output

The output consists of multiple NetCDF files, each containing climate variables (`tasmin`, `tasmax`, `prcp`) for a specific model and scenario combination. The files are named as `Raven_input_<model_name>_<scenario>.nc`.
