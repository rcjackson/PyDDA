import xarray as xr
import xradar as xd
import numpy as np

from glob import glob
from datatree import DataTree
from pyart.core.transforms import cartesian_to_geographic

from ..retrieval.angles import add_azimuth_as_field, add_elevation_as_field


def read_grid(file_name, **kwargs):
    """
    Opens a Cf-compliant netCDF grid object produced by a utility like
    PyART or RadxGrid. This will add all variables PyDDA needs
    from these grids to create a PyDDA Grid (based on the xarray :func:`xr.Dataset`).
    This will open the grid as a parent node of a DataTree structure.

    Parameters
    ----------
    file_name: str or list
        The name of the file to open. If a list of files is provided, the grids
        are opened as a parent node with all grid datasets concatenated together.

    Returns
    -------
    root: DataTree
        The dataset with the list of grids as the parent node.
    """

    root = xr.open_dataset(file_name, decode_times=False)

    # Find first grid field name
    root.attrs["first_grid_name"] = ""
    for vars in list(root.variables.keys()):
        if root[vars].dims == ("time", "z", "y", "x"):
            root.attrs["first_grid_name"] = vars
            break
    if root.attrs["first_grid_name"] == "":
        raise IOError("NetCDF file does not contain any valid radar grid fields!")

    root["point_x"] = xr.DataArray(_point_data_factory(root, "x"), dims=("z", "y", "x"))
    root.attrs["units"] = root["x"].units
    root.attrs["long_name"] = "Point x location"
    root["point_y"] = xr.DataArray(_point_data_factory(root, "y"), dims=("z", "y", "x"))
    root.attrs["units"] = root["y"].units
    root.attrs["long_name"] = "Point y location"
    root["point_z"] = xr.DataArray(_point_data_factory(root, "z"), dims=("z", "y", "x"))
    root.attrs["units"] = root["z"].units
    root.attrs["long_name"] = "Point z location"
    root["point_altitude"] = xr.DataArray(
        _point_altitude_data_factory(root), dims=("z", "y", "x")
    )
    root.attrs["units"] = root["z"].units
    root.attrs["long_name"] = "Point altitude"
    lon = _point_lon_lat_data_factory(root, 0)
    lat = _point_lon_lat_data_factory(root, 1)
    root["point_longitude"] = xr.DataArray(lon, dims=("z", "y", "x"))
    root["point_longitude"].attrs["units"] = root["radar_longitude"].attrs["units"]
    root["point_longitude"].attrs["long_name"] = "Point longitude"
    root["point_latitude"] = xr.DataArray(lat, dims=("z", "y", "x"))
    root["point_latitude"].attrs["units"] = root["radar_latitude"].attrs["units"]
    root["point_latitude"].attrs["long_name"] = "Point latitude"
    add_azimuth_as_field(root)
    add_elevation_as_field(root)

    return root


def read_from_pyart_grid(Grid):
    """

    Converts a Py-ART Grid to a PyDDA Dataset with the necessary variables

    Parameters
    ----------
    Grid: Py-ART Grid
        The Py-ART Grid to convert to a PyDDA Grid

    Returns
    -------
    new_grid: PyDDA Dataset
        The xarray Dataset with the additional parameters PyDDA needs
    """
    new_grid = Grid
    radar_latitude = Grid.radar_latitude
    radar_longitude = Grid.radar_longitude
    radar_altitude = Grid.radar_altitude
    origin_latitude = Grid.origin_latitude
    origin_longitude = Grid.origin_longitude
    origin_altitude = Grid.origin_altitude

    if len(list(Grid.fields.keys())) > 0:
        first_grid_name = list(Grid.fields.keys())[0]
    else:
        first_grid_name = ""
    projection = Grid.get_projparams()
    new_grid = new_grid.to_xarray()

    new_grid["projection"] = xr.DataArray(1, dims=(), attrs=projection)

    if "lat_0" in projection.keys():
        new_grid["projection"].attrs["_include_lon_0_lat_0"] = "true"
    else:
        new_grid["projection"].attrs["_include_lon_0_lat_0"] = "false"

    if "units" not in new_grid["time"].attrs.keys():
        new_grid["time"].attrs["units"] = (
            "seconds since %s"
            % new_grid["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").values[0]
        )
    new_grid.attrs["first_grid_name"] = first_grid_name
    x = radar_latitude.pop("data").squeeze()
    new_grid["radar_latitude"] = xr.DataArray(
        np.atleast_1d(x), dims=("nradar"), attrs=radar_latitude
    )
    x = radar_longitude.pop("data").squeeze()
    new_grid["radar_longitude"] = xr.DataArray(
        np.atleast_1d(x), dims=("nradar"), attrs=radar_longitude
    )
    x = radar_altitude.pop("data").squeeze()
    new_grid["radar_altitude"] = xr.DataArray(
        np.atleast_1d(x), dims=("nradar"), attrs=radar_altitude
    )
    x = origin_latitude.pop("data").squeeze()
    new_grid["origin_latitude"] = xr.DataArray(
        np.atleast_1d(x), dims=("nradar"), attrs=origin_latitude
    )
    x = origin_longitude.pop("data").squeeze()
    new_grid["origin_longitude"] = xr.DataArray(
        np.atleast_1d(x), dims=("nradar"), attrs=origin_longitude
    )
    x = origin_altitude.pop("data").squeeze()
    new_grid["origin_altitude"] = xr.DataArray(
        np.atleast_1d(x), dims=("nradar"), attrs=origin_altitude
    )
    new_grid["point_x"] = xr.DataArray(
        _point_data_factory(new_grid, "x"), dims=("z", "y", "x")
    )
    new_grid.attrs["units"] = new_grid["x"].units
    new_grid.attrs["long_name"] = "Point x location"
    new_grid["point_y"] = xr.DataArray(
        _point_data_factory(new_grid, "y"), dims=("z", "y", "x")
    )
    new_grid.attrs["units"] = new_grid["y"].units
    new_grid.attrs["long_name"] = "Point y location"
    new_grid["point_z"] = xr.DataArray(
        _point_data_factory(new_grid, "z"), dims=("z", "y", "x")
    )
    new_grid.attrs["units"] = new_grid["z"].units
    new_grid.attrs["long_name"] = "Point z location"
    new_grid["point_altitude"] = xr.DataArray(
        _point_altitude_data_factory(new_grid), dims=("z", "y", "x")
    )
    new_grid.attrs["units"] = new_grid["z"].units
    new_grid.attrs["long_name"] = "Point altitude"
    lon = _point_lon_lat_data_factory(new_grid, 0)
    lat = _point_lon_lat_data_factory(new_grid, 1)
    new_grid["point_longitude"] = xr.DataArray(
        lon,
        dims=("z", "y", "x"),
    )
    if "units" in radar_longitude.keys():
        new_grid["point_longitude"].attrs["units"] = radar_longitude["units"]
    new_grid["point_longitude"].attrs["long_name"] = "Point longitude"
    new_grid["point_latitude"] = xr.DataArray(lat, dims=("z", "y", "x"))
    if "units" in radar_latitude.keys():
        new_grid["point_latitude"].attrs["units"] = radar_latitude["units"]
    new_grid["point_latitude"].attrs["long_name"] = "Point latitude"
    add_azimuth_as_field(new_grid)
    add_elevation_as_field(new_grid)

    return new_grid


def _point_data_factory(grid, coordinate):
    """The function which returns the locations of all points."""
    reg_x = grid["x"].values
    reg_y = grid["y"].values
    reg_z = grid["z"].values
    if coordinate == "x":
        return np.tile(reg_x, (len(reg_z), len(reg_y), 1)).swapaxes(2, 2)
    elif coordinate == "y":
        return np.tile(reg_y, (len(reg_z), len(reg_x), 1)).swapaxes(1, 2)
    else:
        assert coordinate == "z"
        return np.tile(reg_z, (len(reg_x), len(reg_y), 1)).swapaxes(0, 2)


def _point_lon_lat_data_factory(grid, coordinate):
    """The function which returns the geographic point locations."""
    x = grid["point_x"].values
    y = grid["point_y"].values
    projparams = grid["projection"].attrs

    if "_include_lon_0_lat_0" in projparams.keys():
        if projparams["_include_lon_0_lat_0"] == "true":
            projparams["lon_0"] = grid["origin_longitude"].values
            projparams["lat_0"] = grid["origin_latitude"].values

    geographic_coords = cartesian_to_geographic(x, y, projparams)

    return geographic_coords[coordinate]


def _point_altitude_data_factory(grid):
    return grid["origin_altitude"].values[0] + grid["point_z"].values
