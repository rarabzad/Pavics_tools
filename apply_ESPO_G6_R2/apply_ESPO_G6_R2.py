import os, re, glob, shutil, time, json, zipfile, urllib.request, requests
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
from rasterio.enums import Resampling

# Helper function to replace the missing get_file function
def get_file(file_path):
    """Simple replacement for ravenpy.utilities.testdata.get_file"""
    return file_path

# Helper functions to update the template rvt file correction commands based on the model rvt file
def parse_blocks(lines):
    """Extract blocks between :GriddedForcing and :EndGriddedForcing."""
    blocks = {}
    block_name = None
    current_block = []
    for line in lines:
        if line.startswith(":GriddedForcing"):
            parts = line.split()
            if len(parts) > 1:
                block_name = parts[1]  # Get the forcing name (e.g., Precipitation)
            current_block = [line]
        elif line.startswith(":EndGriddedForcing"):
            current_block.append(line)
            if block_name:
                blocks[block_name] = current_block
            block_name = None
            current_block = []
        elif block_name:
            current_block.append(line)
    return blocks

import cftime, numpy as np, datetime as dt

# Helper functions to get start/end date
def get_start_end_dates(ds):
    cal = ds.time.encoding.get("calendar", ds.time.attrs.get("calendar", "gregorian"))
    nt, first = ds.time.shape[0], ds.time.values[0]
    def fmt(t):
        return t.strftime("%Y-%m-%d") if hasattr(t,"strftime") else str(t)[:10]
    if cal in ["noleap","365_day"]:
        f = first.astype("M8[ns]").astype(dt.datetime) if isinstance(first,np.datetime64) else dt.datetime(first.year,first.month,first.day)
        raw_last = f+dt.timedelta(days=nt-1); leaps=0
        for y in range(f.year,raw_last.year+1):
            if y%4==0 and (y%100!=0 or y%400==0):
                if f.date()<=dt.date(y,2,29)<=raw_last.date(): leaps+=1
        last = raw_last-dt.timedelta(days=leaps)
    elif cal=="360_day":
        yrs,days=divmod(nt-1,360)
        last=cftime.Datetime360Day(first.year+yrs,1,1)+days
    else:
        if isinstance(first,np.datetime64):
            f=first.astype("M8[ns]").astype(dt.datetime); last=f+dt.timedelta(days=nt-1)
        elif isinstance(first,dt.datetime): last=first+dt.timedelta(days=nt-1)
        else:
            units=f"days since {first.strftime('%Y-%m-%d')} 00:00:00"
            last=cftime.num2date(cftime.date2num(first,units,cal)+(nt-1),units,cal)
    return fmt(first),fmt(last)


def update_template(template_lines, file_lines):
    """Replace correction factor lines in template with those from the file."""
    template_blocks = parse_blocks(template_lines)
    file_blocks = parse_blocks(file_lines)
    
    for block_name, file_block in file_blocks.items():
        if block_name in template_blocks:
            # Extract correction factor lines from the file block
            correction_lines = [
                line for line in file_block if any(
                    factor in line for factor in [":RainCorrection", ":SnowCorrection", ":TemperatureCorrection"]
                )
            ]
            # Replace correction lines in the template block
            updated_block = []
            correction_replaced = False

            for line in template_blocks[block_name]:
                if any(factor in line for factor in [":RainCorrection", ":SnowCorrection", ":TemperatureCorrection"]):
                    if not correction_replaced:
                        updated_block.extend(correction_lines)  # Insert the extracted correction lines
                        correction_replaced = True
                else:
                    updated_block.append(line)
            # Update the block in the template
            template_blocks[block_name] = updated_block
    
    # Reconstruct the updated template
    updated_template = []
    for block in template_blocks.values():
        updated_template.extend(block)
    return updated_template

def safe_file_operation(operation, *args, **kwargs):
    """Safely perform file operations with error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                raise e
            print(f"File operation failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(1)

# Validate required variables exist
required_vars = ['hrufile_path', 'Raven_model_dir', 'model_name', 'scenario']
for var in required_vars:
    if var not in globals():
        raise ValueError(f"Required variable '{var}' is not defined. Please define it before running this script.")

# Validate paths exist
if not os.path.exists(hrufile_path):
    raise FileNotFoundError(f"HRU shapefile not found: {hrufile_path}")
if not os.path.exists(Raven_model_dir):
    raise FileNotFoundError(f"Raven model directory not found: {Raven_model_dir}")

try:
    # Step 1: Get the model prefix from the .rvi file in the Raven model directory
    rvi_files = [file for file in os.listdir(Raven_model_dir) if file.endswith('.rvi')]
    if not rvi_files:
        raise FileNotFoundError(f"No .rvi files found in {Raven_model_dir}")
    
    model_prefix = [os.path.splitext(file)[0] for file in rvi_files]
    if not model_prefix:
        raise ValueError("Could not extract model prefix from .rvi files")
    
    print(f"Found model prefix: {model_prefix[0]}")

    # Step 2: Define the path for the corresponding .rvt file (Raven input file)
    rvt_file_path = os.path.join(Raven_model_dir, f"{model_prefix[0]}.rvt")
    if not os.path.exists(rvt_file_path):
        raise FileNotFoundError(f"RVT file not found: {rvt_file_path}")

    # Step 3: Open the .rvt file and read its contents
    with open(rvt_file_path, 'r') as file:
        rvt = file.read()

    # Step 4: Remove the existing GriddedForcing blocks from the .rvt content
    redirect_lines = re.sub(r":GriddedForcing.*?:EndGriddedForcing", "", rvt, flags=re.DOTALL).splitlines()

    # Step 5: Split the rvt file contents by lines
    rvt = rvt.splitlines()

    # Step 6: Define the template rvt file with GriddedForcing configurations
    template_rvt = """\
:GriddedForcing            Precipitation
    :ForcingType           PRECIP
    :FileNameNC            forcing_file_path
    :VarNameNC             precipitation
    :DimNamesNC            rlon rlat time
    :RainCorrection        1
    :SnowCorrection        1
    :ElevationVarNameNC    elevation
    :RedirectToFile        GridWeights.txt
:EndGriddedForcing
:GriddedForcing            Maxtemp
    :ForcingType           TEMP_DAILY_MAX
    :FileNameNC            forcing_file_path
    :VarNameNC             max_temperature
    :DimNamesNC            rlon rlat time
    :TemperatureCorrection 1
    :ElevationVarNameNC    elevation
    :RedirectToFile        GridWeights.txt
:EndGriddedForcing
:GriddedForcing            Mintemp
    :ForcingType           TEMP_DAILY_MIN
    :FileNameNC            forcing_file_path
    :VarNameNC             min_temperature
    :DimNamesNC            rlon rlat time
    :TemperatureCorrection 1
    :ElevationVarNameNC    elevation
    :RedirectToFile        GridWeights.txt
:EndGriddedForcing"""
    
    template_rvt = template_rvt.splitlines()
    template_rvt = update_template(template_rvt, rvt)

    # Create temporary file for debugging
    with open("tmp.rvt", "w") as file:
        for line in template_rvt:
            file.write(line + "\n")

    # Step 7: Combine template with redirect lines
    new_rvt = template_rvt + redirect_lines

    # Step 8: Set the output filename for the NetCDF file
    output_file = f"Raven_input_{model_name}_{scenario}.nc"

    # Step 9: Replace placeholders in the content with actual file paths
    new_rvt = [line.replace("forcing_file_path", output_file) for line in new_rvt]

    # Step 10: Write the updated content back to the .rvt file
    safe_file_operation(
        lambda: shutil.copy(rvt_file_path, rvt_file_path + ".backup")
    )  # Create backup
    
    with open(rvt_file_path, "w") as file:
        for line in new_rvt:
            file.write(line + "\n")

    # Step 11: Load the HRU shapefile to define the area of interest
    print("Loading HRU shapefile...")
    try:
        hru = gpd.read_file(hrufile_path)
    except Exception as e:
        raise ValueError(f"Error reading HRU shapefile: {e}")

    # Step 12: Ensure the HRU is in the correct CRS (WGS84 lat/lon)
    if hru.crs != 'EPSG:4326':
        hru = hru.to_crs(epsg=4326)
    print(f"HRU CRS: {hru.crs}, Bounds: {hru.total_bounds}")

    # Step 13: Access the climate data from a remote source using the Siphon library
    print("Accessing climate data catalog...")
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"
    
    try:
        cat = TDSCatalog(url)
    except Exception as e:
        raise ConnectionError(f"Could not access THREDDS catalog: {e}")

    # Step 14: Retrieve the dataset corresponding to the given model and scenario
    datasets = [dataset for dataset in cat.datasets]
    id = [idx for idx, dataset in enumerate(cat.datasets) if model_name in dataset and scenario in dataset]
    
    if not id:
        available_datasets = [d for d in cat.datasets.keys()]
        raise ValueError(f"No matching dataset found for model: {model_name}, scenario: {scenario}. "
                        f"Available datasets: {available_datasets[:10]}...")  # Show first 10
    
    cds = cat.datasets[id[0]]
    print(f"Found dataset: {list(cat.datasets.keys())[id[0]]}")

    # Step 15: Open the dataset using xarray and enable chunking for memory efficiency
    print("Opening climate dataset...")
    try:
        ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")
    except Exception as e:
        raise ConnectionError(f"Could not open dataset via OPENDAP: {e}")

    # Step 16: Define a bounding box around the HRU to extract relevant grid indices
    buffer_size = 0.06  # degrees
    minx, miny, maxx, maxy = hru.total_bounds + np.array([-buffer_size, -buffer_size, buffer_size, buffer_size])
    bounding_box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
    print(f"Bounding box: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")

    # Step 17: Find points within the bounding box
    try:
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
    except KeyError as e:
        available_vars = list(ds.variables.keys())
        raise KeyError(f"Required coordinate variable not found: {e}. Available variables: {available_vars}")
    
    rlat_dim, rlon_dim = lat.shape
    lat_within_bbox = (lat >= miny) & (lat <= maxy)
    lon_within_bbox = (lon >= minx) & (lon <= maxx)
    points_within_bbox = np.where(lat_within_bbox & lon_within_bbox)
    rlat_ids, rlon_ids = points_within_bbox

    if len(rlat_ids) == 0 or len(rlon_ids) == 0:
        raise ValueError("No climate data points found within the HRU bounding box. Check coordinate systems.")

    # Step 18: Find the min/max indices for latitude and longitude within the bounding box
    row_indices = range(rlat_ids.min(), rlat_ids.max() + 1)
    col_indices = range(rlon_ids.min(), rlon_ids.max() + 1)
    lat_idx_min = np.min(row_indices)
    lat_idx_max = np.max(row_indices) + 1
    lon_idx_min = np.min(col_indices)
    lon_idx_max = np.max(col_indices) + 1
    
    print(f"Grid subset indices - lat: {lat_idx_min}:{lat_idx_max}, lon: {lon_idx_min}:{lon_idx_max}")

    # Step 19: Extract the relevant data (temperature and precipitation) from the climate dataset
    print("Extracting climate variables...")
    required_vars = ['tasmin', 'tasmax', 'pr']
    for var in required_vars:
        if var not in ds.variables:
            available_vars = [v for v in ds.variables.keys() if not v in ['lat', 'lon', 'time', 'rlat', 'rlon']]
            raise KeyError(f"Required variable '{var}' not found in dataset. Available data variables: {available_vars}")
    
    tasmin_values = ds.tasmin.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    tasmax_values = ds.tasmax.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    prcp_values = ds.pr.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lat_values = ds.lat.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lon_values = ds.lon.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))

    # Create bounding box geojson for elevation data
    points = gpd.GeoDataFrame(
        {"geometry": [Point(lon, lat) for lon, lat in zip(lon_values.values.flatten(), lat_values.values.flatten())]},
        crs="EPSG:4326",
    )

    polygon = points.union_all().convex_hull
    buffered_polygon = polygon.buffer(0.06)
    buffered_polygon_gdf = gpd.GeoDataFrame({"geometry": [buffered_polygon]}, crs="EPSG:4326")
    
    geojson_file_path = os.path.join(os.getcwd(), "bounding_box.geojson")
    buffered_polygon_gdf.to_file(geojson_file_path, driver="GeoJSON")

    # Step 20: Trigger computation to load data into memory (using Dask for parallel processing)
    print("Computing climate data arrays...")
    try:
        tasmin_values = tasmin_values.compute()
        tasmax_values = tasmax_values.compute()
        prcp_values = prcp_values.compute()
        lat_values = lat_values.compute()
        lon_values = lon_values.compute()
    except Exception as e:
        raise RuntimeError(f"Error computing climate data: {e}")

    # Step 21: Derive the elevation data for grids from WPSClient server
    print("Fetching elevation data...")
    wps_url = os.environ.get("WPS_URL", "https://pavics.ouranos.ca/twitcher/ows/proxy/raven/wps")
    
    try:
        wps = WPSClient(wps_url)
    except Exception as e:
        raise ConnectionError(f"Could not connect to WPS server: {e}")

    # Step 22: Fetch elevation data for all grid points within the HRU region
    try:
        feature_url = get_file(geojson_file_path)
        terrain_resp = wps.terrain_analysis(shape=feature_url, select_all_touching=True, projected_crs=3978)
        properties, dem = terrain_resp.get(asobj=True)
        
        # Reproject to lat/lon
        dem_latlon = dem.rio.reproject(CRS.from_epsg(4326))
        
        # Resample to coarser resolution if needed
        new_resolution = 0.05  # degrees
        dem_resampled = dem_latlon.rio.reproject(
            dem_latlon.rio.crs,
            resolution=(new_resolution, new_resolution),
            resampling=Resampling.bilinear
        )
        dem_resampled = dem_resampled.sortby(['x', 'y'])
        
    except Exception as e:
        print(f"Warning: Could not fetch elevation data: {e}")
        print("Using default elevation values...")
        # Use default elevation if WPS fails
        elevation_values = np.full(lat_values.values.shape, 100.0)  # Default 100m elevation
    else:
        # Extract elevation values for each grid point
        flat_lon = lon_values.values.flatten()
        flat_lat = lat_values.values.flatten()
        extracted_values = []
        
        for lon, lat in zip(flat_lon, flat_lat):
            try:
                value = dem_resampled.sel(x=lon, y=lat, method="nearest").values
                # Handle potential NaN values
                if np.isnan(value):
                    value = 100.0  # Default elevation
                extracted_values.append(value)
            except Exception:
                extracted_values.append(100.0)  # Default if extraction fails
        
        elevation_values = np.array(extracted_values).reshape(lat_values.values.shape)

    # Step 23: Create a new xarray Dataset with the extracted variables
    print("Creating output NetCDF dataset...")
    
    # Validate data shapes
    expected_shape = (len(ds['time']), lat_values.shape[0], lat_values.shape[1])
    for var_name, var_data in [('tasmin', tasmin_values), ('tasmax', tasmax_values), ('pr', prcp_values)]:
        if var_data.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {var_name}: expected {expected_shape}, got {var_data.shape}")
    
    new_ds = xr.Dataset(
        {
            'min_temperature': (['time', 'rlat', 'rlon'], tasmin_values.values - 273.15),  # Convert from K to °C
            'max_temperature': (['time', 'rlat', 'rlon'], tasmax_values.values - 273.15),  # Convert from K to °C
            'precipitation':   (['time', 'rlat', 'rlon'], prcp_values.values * 86400),    # Convert from m/s to mm/day
            'lat':             (['rlat', 'rlon'],         lat_values.values),
            'lon':             (['rlat', 'rlon'],         lon_values.values),
            'elevation':       (['rlat', 'rlon'],         elevation_values),
        },
        coords={
            'time': ds['time'],
            'rlat': ds['rlat'][lat_idx_min:lat_idx_max],
            'rlon': ds['rlon'][lon_idx_min:lon_idx_max],
        }
    )

    # Step 24: Set attributes for each variable in the new dataset
    new_ds.min_temperature.attrs['units'] = "degC"
    new_ds.min_temperature.attrs['long_name'] = "Daily minimum temperature"
    new_ds.max_temperature.attrs['units'] = "degC"
    new_ds.max_temperature.attrs['long_name'] = "Daily maximum temperature"
    new_ds.precipitation.attrs['units'] = "mm"
    new_ds.precipitation.attrs['long_name'] = "Daily precipitation"
    new_ds.lat.attrs['units'] = "degrees_north"
    new_ds.lat.attrs['long_name'] = "Latitude"
    new_ds.lon.attrs['units'] = "degrees_east"
    new_ds.lon.attrs['long_name'] = "Longitude"
    new_ds.elevation.attrs['units'] = "m"
    new_ds.elevation.attrs['long_name'] = "Elevation above sea level"

    # Step 25: Save the new dataset to a NetCDF file
    nc_file_path = os.path.join(Raven_model_dir, output_file)
    print(f"Saving NetCDF file: {nc_file_path}")
    
    try:
        new_ds.to_netcdf(nc_file_path)
    except Exception as e:
        raise IOError(f"Could not save NetCDF file: {e}")

    # Step 26: Update the .rvi file with the correct start and end dates
    rvi_file_path = os.path.join(Raven_model_dir, f"{model_prefix[0]}.rvi")
    
    with open(rvi_file_path, 'r') as file:
        rvi = file.read().splitlines()

    # Extract time information - handle cftime objects properly
    time = new_ds['time'].values
    
    # Handle cftime objects (common in climate data with non-standard calendars)
    if hasattr(time[0], 'year'):  # cftime object
        time_as_datetime = [pd.Timestamp(year=t.year, month=t.month, day=t.day) for t in time]
    else:  # numpy datetime64 or other format
        time_as_datetime = [pd.Timestamp(t) for t in time]
    
    start_date, end_date = get_start_end_dates(new_ds)
   
    print(f"Time range: {start_date} to {end_date}")

    # Update the start and end dates in the rvi file
    try:
        start_date_index = next(i for i, line in enumerate(rvi) if line.startswith(":StartDate"))
        end_date_index = next(i for i, line in enumerate(rvi) if line.startswith(":EndDate"))
        
        rvi[start_date_index] = f":StartDate {start_date}"
        rvi[end_date_index] = f":EndDate {end_date}"
        
        # Ensure proper line endings
        rvi = [line.rstrip() + '\n' for line in rvi]

        # Save the updated rvi file
        with open(rvi_file_path, 'w') as file:
            file.writelines(rvi)
            
    except StopIteration:
        print("Warning: Could not find :StartDate or :EndDate lines in .rvi file")

    # Step 27: Copy the HRU shapefile and related files to the Raven model directory
    print("Copying shapefile components...")
    shapefile_base = os.path.splitext(os.path.basename(hrufile_path))[0]
    files_to_copy = glob.glob(os.path.join(os.path.dirname(hrufile_path), shapefile_base + '*'))
    
    if not files_to_copy:
        raise FileNotFoundError(f"No shapefile components found for {shapefile_base}")
    
    for file in files_to_copy:
        try:
            shutil.copy(file, Raven_model_dir)
        except Exception as e:
            print(f"Warning: Could not copy {file}: {e}")

    # Step 28: Download and execute the grid weights generator script
    print("Generating grid weights...")
    cwd = os.getcwd()
    
    try:
        os.chdir(Raven_model_dir)
        
        script_filename = "derive_grid_weights.py"
        script_url = "https://raw.githubusercontent.com/julemai/GridWeightsGenerator/refs/heads/main/derive_grid_weights.py"
        
        try:
            urllib.request.urlretrieve(script_url, script_filename)
        except Exception as e:
            raise ConnectionError(f"Could not download grid weights generator: {e}")
        
        # Verify required files exist
        nc_path = output_file
        shapefile_path = shapefile_base + ".shp"
        
        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
        if not os.path.exists(shapefile_path):
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        
        # Set up grid weights generation command
        key_col = "HRU_ID"
        dimname = "rlat,rlon"
        varname = "lon,lat"
        grid_weights_path = "GridWeights.txt"
        
        command = (
            f'python {script_filename} '
            f'-i {nc_path} '
            f'-d "{dimname}" '
            f'-v "{varname}" '
            f'-r {shapefile_path} '
            f'-a --doall '
            f'-c {key_col} '
            f'-o {grid_weights_path}'
        )
        
        print(f"Running command: {command}")
        exit_code = os.system(command)
        
        if exit_code != 0:
            print(f"Warning: Grid weights generation returned non-zero exit code: {exit_code}")
        
        # Verify grid weights file was created
        if not os.path.exists(grid_weights_path):
            raise FileNotFoundError("GridWeights.txt file was not generated successfully")
            
    finally:
        os.chdir(cwd)

    # Step 29: Clean up by removing the HRU shapefile and related files from the Raven model directory
    print("Cleaning up temporary files...")
    for file in glob.glob(os.path.join(Raven_model_dir, shapefile_base + '*')):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove {file}: {e}")

    # Clean up temporary files
    temp_files = ["bounding_box.geojson", "tmp.rvt"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove {temp_file}: {e}")

    # Step 30: Run the Raven model with the updated configuration
    print("Running Raven model...")
    try:
        result = ravenpy.run(modelname=model_prefix[0], configdir=Raven_model_dir)
        print("Raven model completed successfully!")
        print("Workflow completed!")
    except Exception as e:
        raise RuntimeError(f"Raven model execution failed: {e}")

except Exception as e:
    print(f"Workflow failed with error: {e}")
    raise
