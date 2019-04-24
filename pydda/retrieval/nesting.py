import numpy as np
import pyart
import os
import gc
import glob

from distributed import Client, wait
from scipy.interpolate import griddata
from copy import deepcopy
from datetime import datetime
from .wind_retrieve import get_dd_wind_field
from scipy import sparse


# Reduces the resolution of a PyART grid
def _reduce_pyart_grid_res(Grid, skip_factor):
    Grid2 = deepcopy(Grid)
    field_dict = {}
    for field_name in Grid2.fields.keys():
        field_dict[field_name] = Grid2.fields[field_name].copy()
        field_dict[field_name]["data"] = Grid2.fields[field_name]["data"][
            :, ::skip_factor, ::skip_factor]

    x = Grid2.x
    x["data"] = x["data"][::skip_factor]
    y = Grid2.y
    y["data"] = y["data"][::skip_factor]
    z = Grid2.z
    z["data"] = z["data"]
    metadata = Grid2.metadata
    origin_latitude = Grid2.origin_latitude
    origin_longitude = Grid2.origin_longitude
    origin_altitude = Grid2.origin_altitude
    projection = Grid2.projection
    radar_latitude = Grid2.radar_latitude
    radar_longitude = Grid2.radar_longitude
    radar_altitude = Grid2.radar_altitude
    radar_time = Grid2.radar_time
    radar_name = Grid2.radar_name
    gtime = Grid2.time
    new_grid = pyart.core.Grid(
        gtime, field_dict, metadata, origin_latitude, origin_longitude,
        origin_altitude, x, y, z, projection, radar_latitude, radar_longitude,
        radar_altitude, radar_time, radar_name)
    del Grid2
    return new_grid


# Splits a Py-ART Grid
def _split_pyart_grid(Grid, split_factor, axis=1):
    grid_splits = []
    split_field = {}
    Grid2 = deepcopy(Grid)
    for field_name in Grid2.fields.keys():
        if isinstance(Grid2.fields[field_name]["data"], np.ma.MaskedArray):
            no_mask = Grid2.fields[field_name]["data"].filled(np.nan).copy()
        else:
            no_mask = Grid2.fields[field_name]["data"].copy()
        split_field[field_name] = np.array_split(
            no_mask, split_factor, axis=axis)
        if isinstance(Grid2.fields[field_name]["data"], np.ma.MaskedArray):
            split_field[field_name] = [np.ma.masked_where(
                np.isnan(arr), arr) for arr in split_field[field_name]]
    x = Grid2.x
    y = Grid2.y
    z = Grid2.z
    x_split = np.array_split(x["data"], split_factor)
    y_split = np.array_split(y["data"], split_factor)
    z_split = np.array_split(z["data"], split_factor)
    gtime = Grid2.time
    metadata = Grid2.metadata
    origin_latitude = Grid2.origin_latitude
    origin_longitude = Grid2.origin_longitude
    origin_altitude = Grid2.origin_altitude
    projection = Grid2.projection
    radar_latitude = Grid2.radar_latitude
    radar_longitude = Grid2.radar_longitude
    radar_altitude = Grid2.radar_altitude
    radar_time = Grid2.radar_time
    radar_name = Grid2.radar_name
    for i in range(split_factor):
        grid_dic = {}

        for field_name in Grid2.fields.keys():
            grid_dic[field_name] = Grid2.fields[field_name].copy()
            grid_dic[field_name]["data"] = split_field[field_name][i]
        x_dic = x.copy()
        y_dic = y.copy()
        z_dic = z.copy()
        if(axis == 1):
            y_dic["data"] = y_split[i]
        elif(axis == 2):
            x_dic["data"] = x_split[i]
        elif(axis == 0):
            z_dic["data"] = z_split[i]

        new_grid = pyart.core.Grid(
            gtime, grid_dic, metadata, origin_latitude, origin_longitude,
            origin_altitude, x_dic, y_dic, z_dic, projection, radar_latitude,
            radar_longitude, radar_altitude, radar_time, radar_name)
        grid_splits.append(new_grid)

    return grid_splits


# Concatenates Py-ART Grids
def _concatenate_pyart_grids(grid_list, axis=1):
    new_grid = deepcopy(grid_list[0])
    for field_name in new_grid.fields.keys():
        new_grid.fields[field_name]["data"] = np.ma.concatenate(
            [x.fields[field_name]["data"] for x in grid_list], axis=axis)
    if(axis == 2):
        new_grid.x["data"] = np.ma.concatenate(
            [x.x["data"] for x in grid_list])
        new_grid.nx = np.sum([x.nx for x in grid_list])
    elif(axis == 1):
        new_grid.y["data"] = np.ma.concatenate(
            [x.y["data"] for x in grid_list])
        new_grid.ny = np.sum([x.ny for x in grid_list])
    elif(axis == 0):
        new_grid.z["data"] = np.ma.concatenate(
            [x.z["data"] for x in grid_list])
        new_grid.nz = np.sum([x.nz for x in grid_list])
    return new_grid


# Procedure: 1. Do first pass of retrieval on reduced resolution grid
# 2. Then, we use the reduced resolution retrieval as an input to the
# high resolution retrieval in each region
# Finally, we check for continuity at the boundaries
def get_dd_wind_field_nested(grid_list, u_init, v_init, w_init, client=None,
                             reduction_factor=2, num_splits=1, **kwargs):
    """
    This function performs a wind retrieval using a nested domain.
    This is useful for grids that are larger than about 500 by 500
    by 40 points, since the use of larger grids on a single machine
    will exceed memory limitations.

    This procedure relies on a dask distributed cluster to be set up.
    The retrieval is first performed at a resolution that is coarser
    than the analysis grid by reduction_factor. This provides the
    initial state for the nested loop.

    The domain is split into num_splits**2 sub-domains for the nested
    retrieval step, and each nested retrieval is mapped onto a distributed
    worker for parallel processing. If NumPy and SciPy are already set up to
    use parallel numerical analysis libraries, it is recommended that a single
    machine be dedicated to each nest rather than a single core
    for best performance.

    Parameters
    ==========
    grid_list: list
       A list of Py-ART grids for each radar to use in the retrieval.
    u_init: 3D NumPy array
       The initial guess of the zonal wind field. This has to be in the same
       shape as the analysis grid.
    v_init: 3D NumPy array
       The initial guess of the meridional wind field. This has to be in the
       same shape as the analysis grid.
    w_init: 3D NumPy array
       The initial guess of the vertical wind field. This has to be in the same
       shape as the analysis grid.
    client: dask distributed Client
       The distributed Client that is linked to a distributed cluster. The
       :cluster must be running before get_dd_wind_field_nested is called.
       The retrieval on each nest will be mapped onto each worker. Since
       the optimization loop already takes advantage of parallelism, it's
       best to allow at least 16 cores per one worker.
    reduction_factor: int
       How much to reduce the factor of the analysis grid by when doing the
       initial retrieval on the entire grid.
    num_splits: int
       The number of splits to make through each axis when doing the nesting.

    **kwargs: dict
        This function will take the same keyword arguments as
        get_dd_wind_field, as these arguments are passed into each call of
        get_dd_wind_field. See get_dd_wind_field for more information on the
    """
    # First, we do retrieval on whole grid with fraction of resolution
    grid_lo_res_list = [_reduce_pyart_grid_res(G, reduction_factor)
                        for G in grid_list]
    coarse_Co = kwargs.get('Co')*reduction_factor**2
    coarse_Cm = kwargs.get('Cm')*reduction_factor**2

    new_kwargs = kwargs.copy()
    new_kwargs['Co'] = coarse_Co
    new_kwargs['Cm'] = coarse_Cm
    first_pass = get_dd_wind_field(
        grid_lo_res_list, u_init[::, ::reduction_factor, ::reduction_factor],
        v_init[::, ::reduction_factor, ::reduction_factor],
        w_init[::, ::reduction_factor, ::reduction_factor],
        **new_kwargs)

    # Take the first pass field and regrid to analysis field
    reduced_x = first_pass[0].point_x["data"].flatten()
    reduced_y = first_pass[0].point_y["data"].flatten()
    reduced_z = first_pass[0].point_z["data"].flatten()
    x = grid_list[0].point_x["data"].flatten()
    y = grid_list[0].point_y["data"].flatten()
    z = grid_list[0].point_z["data"].flatten()
    u_init_new = griddata((reduced_z, reduced_y, reduced_x),
                          first_pass[0].fields["u"]["data"].flatten(),
                          (z, y, x), method='nearest')
    v_init_new = griddata((reduced_z, reduced_y, reduced_x),
                          first_pass[0].fields["v"]["data"].flatten(),
                          (z, y, x), method='nearest')
    w_init_new = griddata((reduced_z, reduced_y, reduced_x),
                          first_pass[0].fields["w"]["data"].flatten(),
                          (z, y, x), method='nearest')
    u_init_new = np.reshape(u_init_new, u_init.shape)
    v_init_new = np.reshape(v_init_new, v_init.shape)
    w_init_new = np.reshape(w_init_new, w_init.shape)

    # Finally, split the analysis into num_splits**2 pieces and save
    # as temporary files
    tempfile_name_base = datetime.now().strftime('%y%m%d.%H%M%S')
    tiny_grids = []
    k = 0
    for G in grid_list:
        cur_list = []
        split_grids_x = _split_pyart_grid(G, num_splits, axis=2)
        i = 0
        for sgrid in split_grids_x:
            g_list = _split_pyart_grid(sgrid, num_splits)
            grid_fns = []
            j = 0
            for g in g_list:
                fn = (tempfile_name_base + str(k) + '.' +
                      str(i) + '.' + str(j) + '.nc')
                pyart.io.write_grid((tempfile_name_base + str(k) +
                                     '.' + str(i) + '.' + str(j) + '.nc'), g)
                j = j + 1
                grid_fns.append(fn)
            cur_list.append(grid_fns)
            i = i + 1
        del split_grids_x, g_list

        k = k + 1
        tiny_grids.append(cur_list)

    # Temporarily save the tiny grids and free up memory...we want to
    # load these when we are running it on the cluster

    u_init_split_x = np.array_split(u_init_new, num_splits, axis=2)
    u_init_split = [np.array_split(ux, num_splits, axis=1)
                    for ux in u_init_split_x]
    w_init_split_x = np.array_split(w_init_new, num_splits, axis=2)
    w_init_split = [np.array_split(wx, num_splits, axis=1)
                    for wx in w_init_split_x]
    v_init_split_x = np.array_split(v_init_new, num_splits, axis=2)
    v_init_split = [np.array_split(vx, num_splits, axis=1)
                    for vx in v_init_split_x]

    # Clear out unneeded variables (do not need lo-res grids in memory anymore)
    del u_init_split_x, w_init_split_x, v_init_split_x
    del first_pass, reduced_x, reduced_y, reduced_z, x, y, z, grid_lo_res_list
    gc.collect()

    # Serial just for testing, need to use dask in future
    tiny_retrieval = []

    def do_tiny_retrieval(i, j):
        tgrids = [pyart.io.read_grid(tiny_grids[k][i][j])
                  for k in range(len(grid_list))]
        new_grids = get_dd_wind_field(
            tgrids, u_init_split[i][j], v_init_split[i][j],
            w_init_split[i][j], **kwargs)
        del tgrids
        gc.collect()
        return new_grids

    if client is not None:
        futures_array = []
        for i in range(num_splits):
            for j in range(num_splits):
                futures_array.append(client.submit(do_tiny_retrieval, i, j))
        print("Waiting for nested grid to be retrieved...")
        wait(futures_array)
        tiny_retrieval2 = client.gather(futures_array)
    else:
        tiny_retrieval2 = []
        for i in range(num_splits):
            for j in range(num_splits):
                tiny_retrieval2.append(do_tiny_retrieval(i, j))

    tiny_retrieval = []

    for i in range(num_splits):
        new_grid_list = []

        for j in range(len(grid_list)):
            new_grid_list.append(_concatenate_pyart_grids(
                [tiny_retrieval2[k+i*num_splits][j]
                 for k in range(0, num_splits)], axis=1))
        tiny_retrieval.append(new_grid_list)

    new_grid_list = []
    for i in range(len(grid_list)):
        new_grid_list.append(_concatenate_pyart_grids(
            [tiny_retrieval[k][i] for k in range(num_splits)], axis=2))

    tempfile_list = glob.glob(tempfile_name_base + "*")
    for fn in tempfile_list:
        os.remove(fn)

    return new_grid_list
