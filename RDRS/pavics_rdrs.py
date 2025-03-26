from datetime import datetime
import numpy as np
import xarray as xr
import geopandas as gpd
from siphon.catalog import TDSCatalog
from tqdm import tqdm
import pandas as pd
from scipy.spatial import cKDTree
import urllib.request
import os

def process_climate_data(shapefile_path):
    # -----------------------------
    # Step 1: Load NetCDF Dataset using xarray
    # -----------------------------
    print("Loading NetCDF dataset from PAVICS...")
    catUrl = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/reanalyses/catalog.xml"
    datasetName = "day_RDRSv2.1_NAM.ncml"
    catalog = TDSCatalog(catUrl)
    ds = catalog.datasets[datasetName].remote_access()
    lat = ds['lat']
    lon = ds['lon']
    print("Dataset loaded successfully.")
    
    # -----------------------------
    # Step 2: Load and Transform Shapefile
    # -----------------------------
    print("Loading and transforming shapefile...")
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    buffer = 0.06
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds + [-buffer, -buffer, buffer, buffer]
    print(f"Shapefile bounds: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
    
    # -----------------------------
    # Step 3: Find Grid Indices Inside Bounding Box
    # -----------------------------
    print("Finding grid indices inside bounding box...")
    lat_mask = (lat >= min_lat) & (lat <= max_lat)
    lon_mask = (lon >= min_lon) & (lon <= max_lon)
    row_indices, col_indices = np.where(lat_mask & lon_mask)
    time_indices = generate_ids("1980-01-01", "2018-12-31")
    print(f"Found {len(row_indices)} grid points inside the bounding box.")
    
    # -----------------------------
    # Step 4: Extract Climate Data Inside The Bounding Box
    # -----------------------------
    print("Extracting climate data...")
    tasmin = ds["tasmin"]
    tasmax = ds["tasmax"]
    pr = ds["pr"]
    tasmin = extract_masked_data(tasmin, time_indices, row_indices, col_indices) - 273.15  # Convert to Celsius
    tasmax = extract_masked_data(tasmax, time_indices, row_indices, col_indices) - 273.15  # Convert to Celsius
    pr = extract_masked_data(pr, time_indices, row_indices, col_indices) * 86400  # Convert to mm/day
    print("Climate data extracted successfully.")
    
    # -----------------------------
    # Step 5: Extract Elevation Data Inside The Bounding Box
    # -----------------------------
    print("Extracting elevation data...")
    csv_url = "https://raw.githubusercontent.com/rarabzad/Pavics_tools/refs/heads/main/RDRS/NA_HYDROSHEDS_DEM.csv"
    dem_df = pd.read_csv(csv_url)
    lat_grid = lat[row_indices, col_indices]
    lon_grid = lon[row_indices, col_indices]
    grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    dem_points = np.column_stack((dem_df["lat"].values, dem_df["lon"].values))
    tree = cKDTree(dem_points)
    _, nearest_idx = tree.query(grid_points, k=1)
    dem_values = dem_df.iloc[nearest_idx]["DEM"].values
    elevation = dem_values.reshape(lat_grid.shape)
    print("Elevation data extracted successfully.")
    
    # -----------------------------
    # Step 6: Writing Data into a NetCDF File
    # -----------------------------
    print("Writing data into NetCDF file...")
    time_len, rlat, rlon = pr.shape
    time_range = pd.date_range(start="1980-01-01", periods=time_len, freq="D")
    ds_out = xr.Dataset(
        {
            "elevation": (["rlat", "rlon"], elevation),
            "pr": (["time", "rlat", "rlon"], pr),
            "tasmax": (["time", "rlat", "rlon"], tasmax),
            "tasmin": (["time", "rlat", "rlon"], tasmin),
            "lat": (["rlat", "rlon"], lat[row_indices, col_indices]),
            "lon": (["rlat", "rlon"], lon[row_indices, col_indices]),
        },
        coords={
            "time": ("time", time_range),
            "rlat": ("rlat", np.arange(rlat)),
            "rlon": ("rlon", np.arange(rlon)),
        },
    )
    ds_out["lat"].attrs["long_name"] = "Latitude"
    ds_out["lat"].attrs["units"] = "degrees_north"
    ds_out["lon"].attrs["long_name"] = "Longitude"
    ds_out["lon"].attrs["units"] = "degrees_east"
    ds_out["elevation"].attrs["long_name"] = "Elevation"
    ds_out["elevation"].attrs["units"] = "meters"
    ds_out["pr"].attrs["long_name"] = "Precipitation"
    ds_out["pr"].attrs["units"] = "mm/day"
    ds_out["tasmax"].attrs["long_name"] = "Maximum Temperature"
    ds_out["tasmax"].attrs["units"] = "Celsius"
    ds_out["tasmin"].attrs["long_name"] = "Minimum Temperature"
    ds_out["tasmin"].attrs["units"] = "Celsius"
    ds_out.attrs["created_by"] = "Rezgar Arabzadeh"
    ds_out.attrs["development_date"] = "March 2025"
    ds_out.attrs["data_source"] = "Processed data from PAVICS database"
    ds_out.attrs["pavics_summary"] = (
        "PAVICS is a virtual laboratory facilitating the analysis of climate data. "
        "It provides access to several data collections ranging from observations, climate projections, and reanalyses. "
        "It also provides a Python programming environment to analyze this data without the need to download it. "
        "This working environment is constantly updated with the most efficient libraries for climate data analysis, "
        "in addition to ensuring quality control on the provided data and associated metadata."
    )
    ds_out.attrs["pavics_url"] = "https://pavics.ouranos.ca/"
    ds_out.attrs["contact_email"] = "rarabzad@uwaterloo.ca"
    ds_out.to_netcdf("climate_data.nc", format="NETCDF4")
    print("NetCDF file written successfully.")
    
    # -----------------------------
    # Step 7: Writing "model.rvt" file
    # -----------------------------
    print("Writing model.rvt file...")
    template_rvt = """\
    :GriddedForcing            Precipitation
        :ForcingType           PRECIP
        :FileNameNC            climate_data.nc
        :VarNameNC             pr
        :DimNamesNC            rlon rlat time
        :RainCorrection        1
        :SnowCorrection        1
        :ElevationVarNameNC    elevation
        :RedirectToFile        GridWeights.txt
    :EndGriddedForcing
    :GriddedForcing            Maxtemp
        :ForcingType           TEMP_DAILY_MAX
        :FileNameNC            climate_data.nc
        :VarNameNC             tasmax
        :DimNamesNC            rlon rlat time
        :TemperatureCorrection 1
        :ElevationVarNameNC    elevation
        :RedirectToFile        GridWeights.txt
    :EndGriddedForcing
    :GriddedForcing            Mintemp
        :ForcingType           TEMP_DAILY_MIN
        :FileNameNC            climate_data.nc
        :VarNameNC             tasmin
        :DimNamesNC            rlon rlat time
        :TemperatureCorrection 1
        :ElevationVarNameNC    elevation
        :RedirectToFile        GridWeights.txt
    :EndGriddedForcing"""
    template_rvt = template_rvt.splitlines()

    with open("model.rvt", "w") as file:
        for line in template_rvt:
            file.write(line + "\n")
    print("model.rvt file written successfully.")
    
    # -----------------------------
    # Step 8: Generating Grid Weights
    # -----------------------------       
    print("Generating grid weights...")
    script_filename = "derive_grid_weights.py"
    urllib.request.urlretrieve("https://raw.githubusercontent.com/julemai/GridWeightsGenerator/refs/heads/main/derive_grid_weights.py", script_filename)
    grid_weights_generator_path = script_filename
    nc_path = os.path.join(os.getcwd(), "climate_data.nc")
    shapefile_path = os.path.join(os.getcwd(), shapefile_path)
    key_col = "HRU_ID"
    dimname = "rlat,rlon"
    varname = "lon,lat"
    grid_weights_path = "GridWeights.txt"
    command = (
        f'python {grid_weights_generator_path} '
        f'-i {nc_path} '
        f'-d "{dimname}" '
        f'-v "{varname}" '
        f'-r {shapefile_path} '
        f'-a --doall '
        f'-c {key_col} '
        f'-o {grid_weights_path}'
    )
    os.system(command + " > /dev/null 2>&1")
    print("Grid weights generated successfully.")

def generate_ids(start_date, end_date, ref_date=datetime(1980, 1, 1)):
    start_id = (datetime.strptime(start_date, "%Y-%m-%d") - ref_date).days
    end_id = (datetime.strptime(end_date, "%Y-%m-%d") - ref_date).days
    return np.arange(start_id, end_id + 1)

def extract_masked_data(nc_variable, time_indices, lat_indices, lon_indices):
    time_indices = np.array(time_indices)
    lat_indices = np.array(lat_indices)
    lon_indices = np.array(lon_indices)
    extracted_data = np.full((len(time_indices), len(lat_indices), len(lon_indices)), np.nan)
    total_cells = len(lat_indices) * len(lon_indices)
    progress_bar = tqdm(total=total_cells, desc="Extracting data", unit="cell")
    for lat_idx, lat in enumerate(lat_indices):
        for lon_idx, lon in enumerate(lon_indices):
            extracted_data[:, lat_idx, lon_idx] = nc_variable[time_indices, lat, lon]
            progress_bar.update(1)
    progress_bar.close()
    return extracted_data
