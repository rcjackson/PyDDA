#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:50:58 2018

@author: rjackson
"""

import pydda
import pyart
import numpy as np
import pytest
from datetime import datetime


def test_make_const_wind_field():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)

    Grid = pydda.initialization.make_constant_wind_field(
        Grid, wind=(2.0, 3.0, 4.0), vel_field="zero_field"
    )

    assert np.all(Grid["u"].values == 2.0)
    assert np.all(Grid["v"].values == 3.0)
    assert np.all(Grid["w"].values == 4.0)


def test_make_wind_field_from_profile():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 10000), (-10000, 10000), (-10000, 10000))
    )

    # a zero field
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    height = np.arange(0, 10000, 100)
    u_sound = np.ones(height.shape)
    v_sound = np.ones(height.shape)

    profile = pyart.core.HorizontalWindProfile.from_u_and_v(height, u_sound, v_sound)

    Grid = pydda.initialization.make_wind_field_from_profile(
        Grid, profile, vel_field="zero_field"
    )

    assert np.all(np.round(Grid["u"].values) == 1)
    assert np.all(np.round(Grid["v"].values) == 1)
    assert np.all(Grid["w"].values == 0.0)


def _make_small_grid():
    """5x5x5 Cartesian grid for fast IDW tests."""
    Grid = pyart.testing.make_empty_grid(
        (5, 5, 5), ((0, 10000), (-5000, 5000), (-5000, 5000))
    )
    Grid.add_field("zero_field", {"data": np.zeros((5, 5, 5)), "_FillValue": -9999.0})
    return pydda.io.read_from_pyart_grid(Grid)


def test_iem_idw_single_station_no_sounding():
    """With one station and no sounding, all grid points inherit the station wind."""
    Grid = _make_small_grid()
    station_obs = [{"x": 0.0, "y": 0.0, "z": 0.0, "u": 5.0, "v": 3.0, "w": 0.0}]
    Grid = pydda.initialization.make_initialization_from_iem_obs(Grid, station_obs)
    # One station → all normalised IDW weights = 1 → uniform result everywhere
    assert np.allclose(Grid["u"].values, 5.0)
    assert np.allclose(Grid["v"].values, 3.0)
    assert np.all(Grid["w"].values == 0.0)


def test_iem_idw_sounding_only():
    """Profile only, no stations: output should match sounding at each z level."""
    Grid = _make_small_grid()
    grid_z = Grid["z"].values
    u_snd = grid_z / 1000.0  # linearly increasing with height
    v_snd = np.full_like(grid_z, 2.0)
    profile = pyart.core.HorizontalWindProfile.from_u_and_v(grid_z, u_snd, v_snd)

    Grid = pydda.initialization.make_initialization_from_iem_obs(
        Grid, [], profile=profile
    )

    for iz in range(len(grid_z)):
        assert np.allclose(Grid["u"].values[0, iz, :, :], u_snd[iz])
        assert np.allclose(Grid["v"].values[0, iz, :, :], v_snd[iz])
    assert np.all(Grid["w"].values == 0.0)


def test_iem_idw_sounding_plus_station_anomaly():
    """Station anomaly is added on top of the sounding background at every level."""
    Grid = _make_small_grid()
    grid_z = Grid["z"].values
    u_back = 3.0
    profile = pyart.core.HorizontalWindProfile.from_u_and_v(
        grid_z, np.full_like(grid_z, u_back), np.zeros_like(grid_z)
    )
    # Station u=7 at z=0; sounding at z=0 is 3; anomaly = +4
    station_obs = [{"x": 0.0, "y": 0.0, "z": 0.0, "u": 7.0, "v": 0.0, "w": 0.0}]
    Grid = pydda.initialization.make_initialization_from_iem_obs(
        Grid, station_obs, profile=profile
    )
    # Single station → anomaly spread is uniform → u = 3 + 4 = 7 everywhere
    assert np.allclose(Grid["u"].values, u_back + 4.0)
    assert np.all(Grid["w"].values == 0.0)


def test_iem_idw_midpoint_average():
    """Two equidistant stations: the midpoint should receive their average."""
    Grid = _make_small_grid()
    grid_x = Grid["x"].values  # e.g. [-5000, -2500, 0, 2500, 5000]
    xa, xb = float(grid_x[0]), float(grid_x[-1])
    assert np.isclose(abs(xa), abs(xb)), "grid must be symmetric about x=0"
    ny2, nx2 = len(Grid["y"].values) // 2, len(Grid["x"].values) // 2

    station_obs = [
        {"x": xa, "y": 0.0, "z": 0.0, "u": 10.0, "v": 0.0, "w": 0.0},
        {"x": xb, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0},
    ]
    Grid = pydda.initialization.make_initialization_from_iem_obs(Grid, station_obs)
    # At (x=0, y=0): equal distances to both stations → average u = 5
    u_mid = Grid["u"].values[0, :, ny2, nx2]
    assert np.allclose(u_mid, 5.0, atol=1e-6)


def test_iem_idw_higher_power_weights_nearest():
    """Higher IDW power concentrates weight on the nearest station."""
    Grid = _make_small_grid()
    grid_x = Grid["x"].values
    xa, xb = float(grid_x[0]), float(grid_x[-1])
    ny2 = len(Grid["y"].values) // 2

    station_obs = [
        {"x": xa, "y": 0.0, "z": 0.0, "u": 10.0, "v": 0.0, "w": 0.0},
        {"x": xb, "y": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "w": 0.0},
    ]
    G1 = pydda.initialization.make_initialization_from_iem_obs(
        Grid.copy(deep=True), station_obs, power=1
    )
    G2 = pydda.initialization.make_initialization_from_iem_obs(
        Grid.copy(deep=True), station_obs, power=2
    )
    # At x[1] (closer to station A than to B), power=2 should give higher u
    u_p1 = G1["u"].values[0, 0, ny2, 1]
    u_p2 = G2["u"].values[0, 0, ny2, 1]
    assert u_p2 > u_p1


def test_iem_idw_output_shape_and_dims():
    """u, v, w must have shape (1, nz, ny, nx) with dims (time, z, y, x)."""
    Grid = _make_small_grid()
    nz = len(Grid["z"].values)
    ny = len(Grid["y"].values)
    nx = len(Grid["x"].values)
    station_obs = [{"x": 0.0, "y": 0.0, "z": 0.0, "u": 2.0, "v": 1.0, "w": 0.0}]
    Grid = pydda.initialization.make_initialization_from_iem_obs(Grid, station_obs)
    expected_shape = (1, nz, ny, nx)
    for field in ("u", "v", "w"):
        assert Grid[field].values.shape == expected_shape
        assert Grid[field].dims == ("time", "z", "y", "x")


def test_iem_idw_raises_when_no_data():
    """Empty station_obs with no profile must raise ValueError."""
    Grid = _make_small_grid()
    with pytest.raises(ValueError):
        pydda.initialization.make_initialization_from_iem_obs(Grid, [])


def test_get_iem_data():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 100000.0), (-100000.0, 100000.0), (-100000.0, 100000.0))
    )
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    station_obs = pydda.constraints.get_iem_obs(Grid)
    names = [x["site_id"] for x in station_obs]
    assert "P28" in names
    assert "WLD" in names
    assert "WDG" in names
    assert "SWO" in names
    assert "END" in names


def test_hrrr_data():
    Grid = pyart.testing.make_empty_grid(
        (20, 20, 20), ((0, 100000.0), (-100000.0, 100000.0), (-100000.0, 100000.0))
    )
    fdata3 = np.zeros((20, 20, 20))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    Grid = pydda.constraints.add_hrrr_constraint_to_grid(
        Grid, pydda.tests.get_sample_file("ruc2anl_130_20110520_0800_001.grb2")
    )

    assert Grid["U_hrrr"].max() > 15
    assert Grid["V_hrrr"].max() > 15
    assert Grid["W_hrrr"].max() > 0
