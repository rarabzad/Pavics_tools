def ESPO_G6_R2_Downloader(hrufile_path, model_name, scenario):
    # Step 1: Load the Shapefile (hrus of interest)
    hru = gpd.read_file(hrufile_path)
    # Ensure the hru is in the same CRS as the climate data (lat/lon, WGS84)
    hru = hru.to_crs(epsg=4326)
    # Get the min/max bounds of the hru (lat/lon)
    # Step 2: Access climate data using Siphon
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip6/ouranos/ESPO-G/ESPO-G6-R2v1.0.0/catalog.xml"  # Change to your dataset URL
    cat = TDSCatalog(url)
    # Retrieve the datasets directly (these will be strings, not objects)
    datasets = [dataset for dataset in cat.datasets]
    # Find the dataset matching the model_name and scenario
    id = [idx for idx, dataset in enumerate(cat.datasets) if model_name in dataset and scenario in dataset]
    # If no matching dataset is found, skip the iteration
    if not id:
        print(f"No matching dataset found for model: {model_name}, scenario: {scenario}")
        return
    cds = cat.datasets[id[0]]
    # Open the dataset using xarray (this will use Dask for chunking)
    ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")  # Adjust chunking as needed
    # Step 3: Find the corresponding grid indices for min/max lat/lon
    minx, miny, maxx, maxy = hru.total_bounds + np.array([-0.5, -0.5, 0.5, 0.5])  # returns (minx, miny, maxx, maxy) with a buffer 0.6 degree
    bounding_box = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    rlat_dim, rlon_dim = lat.shape
    lat_flat = lat.values.flatten()
    lon_flat = lon.values.flatten()
    points = [Point(lon_flat[i], lat_flat[i]) for i in range(len(lat_flat))]
    inside_indices = []
    for i, point in enumerate(points):
        if point.within(bounding_box):
            rlat_idx = i // rlon_dim
            rlon_idx = i % rlon_dim
            inside_indices.append((rlat_idx, rlon_idx))
    inside_indices = np.array(inside_indices)
    rlat_ids = inside_indices[:, 0]
    rlon_ids = inside_indices[:, 1]
    row_indices = range(rlat_ids.min(), rlat_ids.max() + 1)
    col_indices = range(rlon_ids.min(), rlon_ids.max() + 1)
    lat_idx_min = np.min(row_indices)
    lat_idx_max = np.max(row_indices)
    lon_idx_min = np.min(col_indices)
    lon_idx_max = np.max(col_indices)
    # Step 4: Extract the subgrid (box-constrained hru) for the entire time range
    tasmin_values = ds.tasmin.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max)) - 273.15
    tasmax_values = ds.tasmax.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max)) - 273.15
    prcp_values   = ds.pr.isel    (rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max)) * 86400
    lat_values    = ds.lat.isel   (rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    lon_values    = ds.lon.isel(rlat=slice(lat_idx_min, lat_idx_max), rlon=slice(lon_idx_min, lon_idx_max))
    # Step 5: Compute the data (if necessary)
    tasmin_values = tasmin_values.compute()  # Trigger computation
    tasmax_values = tasmax_values.compute()  # Similarly for tasmax
    prcp_values = prcp_values.compute()  # Similarly for prcp
    lat_values = lat_values.compute()  # Similarly for lat
    lon_values = lon_values.compute()  # Similarly for lon
    # Step 6: Write the extracted data to a NetCDF file
    # Create a new NetCDF file with the same dimensions
    output_file = f"Raven_input_{model_name}_{scenario}.nc"
    # Create a new xarray Dataset for the output data
    new_ds = xr.Dataset(
        {
            'min_temperature': (['time', 'rlat', 'rlon'], tasmin_values.values - 273.15),  # Use the extracted tasmin values
            'max_temperature': (['time', 'rlat', 'rlon'], tasmax_values.values - 273.15),  # Use the extracted tasmax values
            'precipitation':   (['time', 'rlat', 'rlon'], prcp_values.values * 86400),     # Use the extracted prcp values
            'lat':             (['rlat', 'rlon'],         lat_values.values),              # Use the extracted lat values
            'lon':             (['rlat', 'rlon'],         lon_values.values),              # Use the extracted lon values
        },
        coords={
            'time': ds['time'],  # Time coordinate (same as in the original dataset)
            'rlat': ds['rlat'][lat_idx_min:lat_idx_max],  # Latitude coordinate (subset)
            'rlon': ds['rlon'][lon_idx_min:lon_idx_max],  # Longitude coordinate (subset)
        }
    )
    new_ds.min_temperature.attrs['units'] = "degC"
    new_ds.max_temperature.attrs['units'] = "degC"
    new_ds.precipitation.attrs['units'] = "mm"
    new_ds.lat.attrs['units'] = "degrees_north"
    new_ds.lon.attrs['units'] = "degrees_east"
    # Save the output to NetCDF
    new_ds.to_netcdf(output_file)
    # Print success message
    print(f"File {output_file} written successfully.")
