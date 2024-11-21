def apply_ESPO_G6_R2(hrufile_path, Raven_model_dir, model_name, scenario):
    # Step 1: Get the model prefix from the .rvi file in the Raven model directory
    # This assumes there is a matching .rvi file in the directory, and extracts the model prefix
    model_prefix = [match.group(1) for file in os.listdir(Raven_model_dir) if (match := re.match(r"^(.*)\.rvi$", file))]
    
    # Step 2: Define the path for the corresponding .rvt file (Raven input file)
    rvt_file_path = os.path.join(Raven_model_dir, f"{model_prefix[0]}.rvt")
    
    # Step 3: Open the .rvt file and read its contents
    with open(rvt_file_path, 'r') as file:
        content = file.read()

    # Step 4: Remove the existing GriddedForcing blocks from the .rvt content
    # This will remove any previous GriddedForcing configurations for precipitation and temperature
    content_without_gridded_forcing = re.sub(r":GriddedForcing.*?:EndGriddedForcing", "", content, flags=re.DOTALL)
    
    # Step 5: Extract any existing :RedirectToFile entries from the .rvt file
    redirect_lines = re.findall(r"^(?!.*:GriddedForcing).*(:RedirectToFile[^\n]*)", content_without_gridded_forcing, flags=re.MULTILINE)
    
    # Step 6: Prepend new GriddedForcing configurations for Precipitation, Max Temp, and Min Temp
    prepend_content = """\
    :GriddedForcing            Precipitation
        :ForcingType           PRECIP
        :FileNameNC            forcing_file_path
        :VarNameNC             precipitation
        :DimNamesNC            rlon rlat time
        :ElevationVarNameNC    elevation
        :RedirectToFile        GridWeights.txt
    :EndGriddedForcing
    :GriddedForcing            Maxtemp
        :ForcingType           TEMP_DAILY_MAX
        :FileNameNC            forcing_file_path
        :VarNameNC             max_temperature
        :DimNamesNC            rlon rlat time
        :ElevationVarNameNC    elevation
        :RedirectToFile        GridWeights.txt
    :EndGriddedForcing
    :GriddedForcing            Mintemp
        :ForcingType           TEMP_DAILY_MIN
        :FileNameNC            forcing_file_path
        :VarNameNC             min_temperature
        :DimNamesNC            rlon rlat time
        :ElevationVarNameNC    elevation
        :RedirectToFile        GridWeights.txt
    :EndGriddedForcing"""
    
    # Split the prepend content into lines and add the redirect lines extracted above
    new_rvt = prepend_content.splitlines()
    new_rvt += redirect_lines
    
    # Step 7: Set the output filename for the NetCDF file
    output_file = f"Raven_input_{model_name}_{scenario}.nc"
    
    # Step 8: Replace placeholders in the prepend content with actual file paths
    new_rvt = [line.replace("forcing_file_path", output_file) for line in new_rvt]
    new_rvt = "\n".join(new_rvt)

    # Step 9: Write the updated content back to the .rvt file
    with open(rvt_file_path, 'w') as file:
        file.write(new_rvt)

    # Step 10: Load the HRU shapefile to define the area of interest (catchment or subbasin)
    hru = gpd.read_file(hrufile_path)
    
    # Step 11: Ensure the HRU is in the correct CRS (WGS84 lat/lon)
    hru = hru.to_crs(epsg=4326)

    # Step 12: Access the climate data from a remote source using the Siphon library
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"
    cat = TDSCatalog(url)
    
    # Step 13: Retrieve the dataset corresponding to the given model and scenario
    datasets = [dataset for dataset in cat.datasets]
    id = [idx for idx, dataset in enumerate(cat.datasets) if model_name in dataset and scenario in dataset]
    
    if not id:
        print(f"No matching dataset found for model: {model_name}, scenario: {scenario}")
        return
    cds = cat.datasets[id[0]]

    # Step 14: Open the dataset using xarray and enable chunking for memory efficiency
    ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")

    # Step 15: Define a bounding box around the HRU to extract relevant grid indices from the climate data
    minx, miny, maxx, maxy = hru.total_bounds + np.array([-0.5, -0.5, 0.5, 0.5])  # Add buffer to the bounding box
    bounding_box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
    
    # Step 16: Get latitude and longitude values from the dataset and flatten them for easier processing
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    rlat_dim, rlon_dim = lat.shape
    lat_flat = lat.values.flatten()
    lon_flat = lon.values.flatten()
    points = [Point(lon_flat[i], lat_flat[i]) for i in range(len(lat_flat))]
    
    # Step 17: Identify which grid points lie within the bounding box defined by the HRU
    inside_indices = []
    for i, point in enumerate(points):
        if point.within(bounding_box):
            rlat_idx = i // rlon_dim
            rlon_idx = i % rlon_dim
            inside_indices.append((rlat_idx, rlon_idx))
    inside_indices = np.array(inside_indices)
    rlat_ids = inside_indices[:, 0]
    rlon_ids = inside_indices[:, 1]
    
    # Step 18: Find the min/max indices for latitude and longitude within the bounding box
    row_indices = range(rlat_ids.min(), rlat_ids.max() + 1)
    col_indices = range(rlon_ids.min(), rlon_ids.max() + 1)
    lat_idx_min = np.min(row_indices)
    lat_idx_max = np.max(row_indices)
    lon_idx_min = np.min(col_indices)
    lon_idx_max = np.max(col_indices)

    # Step 19: Extract the relevant data (temperature and precipitation) from the climate dataset
    tasmin_values = ds.tasmin.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    tasmax_values = ds.tasmax.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    prcp_values = ds.pr.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lat_values = ds.lat.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lon_values = ds.lon.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))

    # Step 20: Trigger computation to load data into memory (using Dask for parallel processing)
    tasmin_values = tasmin_values.compute()
    tasmax_values = tasmax_values.compute()
    prcp_values = prcp_values.compute()
    lat_values = lat_values.compute()
    lon_values = lon_values.compute()

    # Step 21: Define a function to fetch elevation data for a specific lat/lon point using OpenElevation API
    def get_elevation(lat, lon):
        """Fetch elevation for a given latitude and longitude from OpenElevation API."""
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url).json()
        if "results" in response:
            return response['results'][0]['elevation']
        else:
            print(f"Error fetching elevation for {lat}, {lon}")
            return np.nan  # Return NaN if an error occurs
    
    # Step 22: Fetch elevation data for all grid points within the HRU region
    elevation_values = np.full(lat_values.shape, np.nan)
    for i in range(lat_values.shape[0]):
        for j in range(lat_values.shape[1]):
            lat = lat_values.values[i, j]
            lon = lon_values.values[i, j]
            elevation_values[i, j] = get_elevation(lat, lon)
            if (i * lat_values.shape[1] + j) % 100 == 0:  # Optional: Print progress every 100th location
                print(f"Fetching elevation for index {(i, j)}: {lat}, {lon}")

    # Step 23: Create a new xarray Dataset with the extracted variables (temperature, precipitation, etc.)
    new_ds = xr.Dataset(
        {
            'min_temperature': (['time', 'rlat', 'rlon'], tasmin_values.values - 273.15),  # Convert from K to °C
            'max_temperature': (['time', 'rlat', 'rlon'], tasmax_values.values - 273.15),  # Convert from K to °C
            'precipitation':   (['time', 'rlat', 'rlon'], prcp_values.values * 86400),    # Convert from m/s to mm
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
    new_ds.max_temperature.attrs['units'] = "degC"
    new_ds.precipitation.attrs['units'] = "mm"
    new_ds.lat.attrs['units'] = "degrees_north"
    new_ds.lon.attrs['units'] = "degrees_east"
    new_ds.elevation.attrs['units'] = "m"

    # Step 25: Save the new dataset to a NetCDF file
    nc_file_path = os.path.join(Raven_model_dir, output_file)
    new_ds.to_netcdf(nc_file_path)

    # Step 26: Update the .rvi file with the correct start and end dates based on the time range of the new dataset
    rvi_file_path = os.path.join(Raven_model_dir, f"{model_prefix[0]}.rvi")
    with open(rvi_file_path, 'r') as file:
        rvi = file.read().splitlines()

    time = new_ds['time'].values
    time_as_datetime = [pd.Timestamp(t.year, t.month, t.day) for t in time]
    start_date = min(time_as_datetime).strftime('%Y-%m-%d')  # First date in 'yyyy-mm-dd' format
    end_date = max(time_as_datetime).strftime('%Y-%m-%d')    # Last date in 'yyyy-mm-dd' format

    # Update the start and end dates in the rvi file
    start_date_index = next(i for i, line in enumerate(rvi) if line.startswith(":StartDate"))
    end_date_index = next(i for i, line in enumerate(rvi) if line.startswith(":EndDate"))
    rvi[start_date_index] = f":StartDate {start_date}\n"
    rvi[end_date_index] = f":EndDate {end_date}\n"
    rvi = [line + '\n' if not line.endswith('\n') else line for line in rvi]
    
    # Save the updated rvi file
    with open(rvi_file_path, 'w') as file:
        file.writelines(rvi)

    # Step 27: Copy the HRU shapefile and related files to the Raven model directory
    shapefile_base = os.path.splitext(os.path.basename(hrufile_path))[0]
    files_to_copy = glob.glob(os.path.join(os.path.dirname(hrufile_path), shapefile_base + '*'))
    for file in files_to_copy:
        shutil.copy(file, Raven_model_dir)

    # Step 28: Download the grid weights generator script and execute it to generate the grid weights
    cwd = os.getcwd()
    os.chdir(Raven_model_dir)
    script_filename = "derive_grid_weights.py"
    urllib.request.urlretrieve("https://raw.githubusercontent.com/julemai/GridWeightsGenerator/refs/heads/main/derive_grid_weights.py", script_filename)
    
    grid_weights_generator_path = script_filename
    nc_path = output_file
    shapefile_path = os.path.join(shapefile_base + ".shp")
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

    # Step 29: Clean up by removing the HRU shapefile and related files from the Raven model directory
    for file in glob.glob(os.path.join(Raven_model_dir, shapefile_base + '*')):
        os.remove(file)

    # Step 30: Run the Raven model with the updated configuration
    ravenpy.run(modelname=model_prefix[0], configdir=Raven_model_dir)