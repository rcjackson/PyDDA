#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:35:43 2018

@author: rjackson
"""

import pydda
import pyart
import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from xarray import DataTree

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    # The Jax engine needs jaxopt for the solver, not just jax itself.
    import jax
    import jaxopt

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from distributed import Client, LocalCluster
from copy import deepcopy


def test_make_updraft_from_convergence_field():
    """Do we have an updraft in a region of convergence and divergence?"""

    Grid = pyart.testing.make_empty_grid(
        (20, 40, 40), ((0, 10000), (-20000, 20000), (-20000, 20000))
    )

    # a zero field
    fdata3 = np.ma.zeros((20, 40, 40))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    odata3 = np.ma.ones((20, 40, 40))
    Grid.add_field("one_field", {"data": odata3, "_FillValue": -9999.0})

    wind_vel = 10.0
    z_ground = 500.0
    z_top = 5000.0
    radius = 3000.0
    back_u = 10.0
    back_v = 10.0
    x_center = 0.0
    y_center = 0.0
    Grid = pydda.io.read_from_pyart_grid(Grid)
    u, v, w = pydda.tests.make_test_divergence_field(
        Grid, wind_vel, z_ground, z_top, radius, back_u, back_v, x_center, y_center
    )

    new_grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid],
        u,
        v,
        w,
        Co=0.0,
        Cz=0,
        Cm=500.0,
        Cmod=0.0,
        mask_outside_opt=False,
        vel_name="one_field",
        refl_field="one_field",
    )
    new_w = new_grids[0]["w"].values

    # We should have a pretty strong updraft in the retrieval!
    assert np.ma.max(new_w > 3)


def _twpice_grids():
    """The TWP-ICE grid pair used by the retrieval tests, with a first guess
    taken from the sounding (which has w = 0 everywhere)."""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)
    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )
    return Grid0, Grid1


# Common arguments for the echo top boundary condition tests. The low pass
# filter and the masking are turned off so that the retrieved w can be compared
# against the first guess point by point.
ECHO_TOP_KWARGS = dict(
    Co=100,
    Cm=1500.0,
    max_iterations=20,
    Cz=0,
    Cmod=0.0,
    vel_name="corrected_velocity",
    wind_tol=0.1,
    refl_field="reflectivity",
    frz=5000.0,
    upper_bc=2,
    above=2.0,
    low_pass_filter=False,
    mask_outside_opt=False,
    mask_w_outside_opt=False,
)


def _assert_impermeable_above_echo_top(Grids, parameters):
    """w must be untouched (i.e. still zero) wherever the echo top condition
    applies, and the retrieval must still be physically sensible elsewhere."""
    # The TensorFlow engines store these as tensors rather than numpy arrays.
    mask = np.asarray(parameters.upper_bc_mask)
    z = np.asarray(parameters.z)
    assert mask.dtype == bool
    # The condition should cover a substantial part of, but not all of, the grid.
    assert 0.0 < mask.mean() < 1.0
    assert not mask[z <= 2000.0].any()

    w = Grids[0]["w"].values.squeeze()
    # A permanently zero gradient component is never moved by L-BFGS-B, so the
    # first guess of w = 0 survives exactly.
    np.testing.assert_array_equal(w[mask], 0.0)
    # ...but the retrieval still produces updrafts where the radars see echoes.
    assert np.nanmax(w[~mask]) > 5

    u_mean = np.nanmean(Grids[0]["u"].values)
    v_mean = np.nanmean(Grids[0]["v"].values)
    assert u_mean > 0
    assert v_mean < 0


def test_twpice_case_upper_bc_echo_top():
    """The echo top impermeability condition holds w fixed above the echo top."""
    Grid0, Grid1 = _twpice_grids()
    Grids, parameters = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1], engine="scipy", **ECHO_TOP_KWARGS
    )
    _assert_impermeable_above_echo_top(Grids, parameters)


def test_twpice_case_upper_bc_modes_differ():
    """The three upper boundary conditions give three different wind fields."""
    kwargs = dict(ECHO_TOP_KWARGS)
    del kwargs["upper_bc"]
    # This test only needs the three fields to be distinguishable, not
    # converged, so it runs for fewer iterations than the others.
    kwargs["max_iterations"] = 10

    results = {}
    for upper_bc in (0, 1, 2):
        Grid0, Grid1 = _twpice_grids()
        Grids, parameters = pydda.retrieval.get_dd_wind_field(
            [deepcopy(Grid0), deepcopy(Grid1)],
            engine="scipy",
            upper_bc=upper_bc,
            **kwargs,
        )
        results[upper_bc] = (Grids[0]["w"].values.squeeze(), parameters)

    w_none, params_none = results[0]
    w_grid_top, params_grid_top = results[1]
    w_echo_top, _ = results[2]

    # No mask is built unless it is needed.
    assert params_none.upper_bc_mask is None
    assert params_grid_top.upper_bc_mask is None

    # upper_bc=1 pins only the model top; this is the regression guard for
    # `upper_bc is True`, which used to skip the condition for integer modes.
    np.testing.assert_array_equal(w_grid_top[-1, :, :], 0.0)
    assert np.any(w_none[-1, :, :] != 0)

    assert np.any(w_echo_top != w_grid_top)
    assert np.any(w_echo_top != w_none)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_twpice_case_upper_bc_echo_top_jax():
    """The echo top impermeability condition also works with the Jax engine."""
    Grid0, Grid1 = _twpice_grids()
    Grids, parameters = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1], engine="jax", **ECHO_TOP_KWARGS
    )
    _assert_impermeable_above_echo_top(Grids, parameters)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_twpice_case_upper_bc_echo_top_tensorflow():
    """The echo top impermeability condition also works with the TensorFlow
    engine."""
    Grid0, Grid1 = _twpice_grids()
    Grids, parameters = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1], engine="tensorflow", **ECHO_TOP_KWARGS
    )
    _assert_impermeable_above_echo_top(Grids, parameters)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_twpice_case_jax():
    """Use a test case from TWP-ICE"""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )

    Grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1],
        Co=100,
        Cm=1500.0,
        wind_tol=0.1,
        max_iterations=20,
        Cz=0,
        Cmod=0.0,
        vel_name="corrected_velocity",
        refl_field="reflectivity",
        frz=5000.0,
        engine="jax",
        mask_outside_opt=True,
        upper_bc=1,
    )

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0]["u"].values)
    v_mean = np.nanmean(Grids[0]["v"].values)
    w_max = np.nanmax(Grids[0]["w"].values)

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 5


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_twpice_case_tensorflow():
    """Use a test case from TWP-ICE"""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )
    Grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1],
        Co=100,
        Cm=1500.0,
        max_iterations=20,
        Cz=0,
        Cmod=0.0,
        vel_name="corrected_velocity",
        wind_tol=0.1,
        refl_field="reflectivity",
        frz=5000.0,
        engine="tensorflow",
        mask_outside_opt=True,
        upper_bc=1,
    )

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0]["u"].values)
    v_mean = np.nanmean(Grids[0]["v"].values)
    w_max = np.nanmax(Grids[0]["w"].values)

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 5


def test_twpice_case():
    """Use a test case from TWP-ICE"""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )
    Grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0, Grid1],
        Co=100,
        Cm=1500.0,
        max_iterations=20,
        Cz=0,
        Cmod=0.0,
        vel_name="corrected_velocity",
        wind_tol=0.1,
        refl_field="reflectivity",
        frz=5000.0,
        engine="scipy",
        mask_outside_opt=True,
        upper_bc=1,
    )

    # In this test grid, we expect the mean flow to be to the southeast
    # Maximum updrafts should be at least 10 m/s
    u_mean = np.nanmean(Grids[0]["u"].values)
    v_mean = np.nanmean(Grids[0]["v"].values)
    w_max = np.nanmax(Grids[0]["w"].values)

    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 5


def test_twpice_case_parallel():
    """TWP-ICE case with parallel=True should produce physically consistent results
    and match the serial retrieval."""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
    Grid1 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR1)
    sounding = pyart.io.read_arm_sonde(pydda.tests.SOUNDING_PATH)

    Grid0 = pydda.initialization.make_wind_field_from_profile(
        Grid0, sounding[1], vel_field="corrected_velocity"
    )

    common_kwargs = dict(
        Co=100,
        Cm=1500.0,
        max_iterations=20,
        Cz=0,
        Cmod=0.0,
        vel_name="corrected_velocity",
        wind_tol=0.1,
        refl_field="reflectivity",
        frz=5000.0,
        engine="scipy",
        mask_outside_opt=True,
        upper_bc=1,
    )

    Grids_serial, _ = pydda.retrieval.get_dd_wind_field(
        [deepcopy(Grid0), deepcopy(Grid1)], **common_kwargs, parallel=False
    )
    Grids_parallel, _ = pydda.retrieval.get_dd_wind_field(
        [deepcopy(Grid0), deepcopy(Grid1)], **common_kwargs, parallel=True
    )

    # Physical sanity: mean flow to the southeast, updrafts present
    u_mean = np.nanmean(Grids_parallel[0]["u"].values)
    v_mean = np.nanmean(Grids_parallel[0]["v"].values)
    w_max = np.nanmax(Grids_parallel[0]["w"].values)
    assert u_mean > 0
    assert v_mean < 0
    assert w_max > 5

    # Numerical equivalence with serial
    np.testing.assert_allclose(
        Grids_parallel[0]["u"].values, Grids_serial[0]["u"].values, rtol=1e-5
    )
    np.testing.assert_allclose(
        Grids_parallel[0]["v"].values, Grids_serial[0]["v"].values, rtol=1e-5
    )
    np.testing.assert_allclose(
        Grids_parallel[0]["w"].values, Grids_serial[0]["w"].values, rtol=1e-5
    )


def test_smoothing():
    """A field of random numbers from 0 to 1
    should smooth out to near 0.5"""
    Grid = pyart.testing.make_empty_grid(
        (20, 40, 40), ((0, 10000), (-20000, 20000), (-20000, 20000))
    )

    # a zero field
    fdata3 = np.ma.zeros((20, 40, 40))
    Grid.add_field("zero_field", {"data": fdata3, "_FillValue": -9999.0})
    odata3 = np.ma.ones((20, 40, 40))
    Grid.add_field("one_field", {"data": odata3, "_FillValue": -9999.0})
    Grid = pydda.io.read_from_pyart_grid(Grid)
    u = np.random.random((20, 40, 40))
    v = np.random.random((20, 40, 40))
    w = np.zeros((20, 40, 40))
    new_grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid],
        u_init=u,
        v_init=v,
        w_init=w,
        Co=0.0,
        Cx=1e-4,
        Cy=1e-4,
        Cm=0.0,
        Cmod=0.0,
        mask_outside_opt=False,
        vel_name="one_field",
        refl_field="one_field",
    )
    new_u = new_grids[0]["u"].values
    new_v = new_grids[0]["v"].values
    assert new_u.std() < u.std()
    assert new_v.std() < v.std()


def test_model_constraint():
    """A retrieval with just the model constraint should converge
    to the model constraint."""
    Grid0 = pydda.io.read_grid(pydda.tests.EXAMPLE_RADAR0)

    """ Make fake model grid of just U = 1 m/s everywhere"""
    Grid0["U_fakemodel"] = xr.ones_like(Grid0["corrected_velocity"])
    Grid0["V_fakemodel"] = xr.ones_like(Grid0["corrected_velocity"])
    Grid0["W_fakemodel"] = xr.ones_like(Grid0["corrected_velocity"])

    u_init = np.zeros_like(Grid0["U_fakemodel"].values).squeeze()
    v_init = np.zeros_like(Grid0["U_fakemodel"].values).squeeze()
    w_init = np.zeros_like(Grid0["U_fakemodel"].values).squeeze()

    new_grids, _ = pydda.retrieval.get_dd_wind_field(
        [Grid0],
        u_init,
        v_init,
        w_init,
        Co=0.0,
        Cx=0.0,
        Cy=0.0,
        Cm=0.0,
        Cmod=1.0,
        mask_outside_opt=False,
        vel_name="corrected_velocity",
        refl_field="reflectivity",
        model_fields=["fakemodel"],
    )

    np.testing.assert_allclose(
        new_grids[0]["u"].values, Grid0["U_fakemodel"].values, atol=1e-2
    )
    np.testing.assert_allclose(
        new_grids[0]["v"].values, Grid0["V_fakemodel"].values, atol=1e-2
    )


@pytest.mark.mpl_image_compare(tolerance=50)
def test_nested_retrieval():
    test_coarse0 = pydda.io.read_grid(pydda.tests.get_sample_file("test_coarse0.nc"))
    test_coarse1 = pydda.io.read_grid(pydda.tests.get_sample_file("test_coarse1.nc"))
    test_fine0 = pydda.io.read_grid(pydda.tests.get_sample_file("test_fine0.nc"))
    test_fine1 = pydda.io.read_grid(pydda.tests.get_sample_file("test_fine1.nc"))

    test_coarse0 = pydda.initialization.make_constant_wind_field(
        test_coarse0, (0.0, 0.0, 0.0)
    )

    kwargs_dict = dict(
        Cm=256.0,
        Co=1e-2,
        Cx=150.0,
        Cy=150.0,
        Cz=150.0,
        Cmod=1e-5,
        model_fields=["hrrr"],
        refl_field="DBZ",
        wind_tol=0.5,
        max_iterations=50,
        filter_order=3,
        engine="scipy",
    )

    test_coarse1["time"] = test_coarse0["time"]
    test_fine0["time"] = test_coarse0["time"]
    test_fine1["time"] = test_coarse1["time"]

    tree_dict = {
        "/nest_0/radar_ktlx": test_coarse0,
        "/nest_0/radar_kict": test_coarse1,
        "/nest_1/radar_ktlx": test_fine0,
        "/nest_1/radar_kict": test_fine1,
    }

    tree = DataTree.from_dict(tree_dict)
    tree["/nest_0/"].attrs = kwargs_dict
    tree["/nest_1/"].attrs = kwargs_dict

    grid_tree = pydda.retrieval.get_dd_wind_field_nested(tree)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    pydda.vis.plot_horiz_xsection_quiver(
        grid_tree["nest_0"],
        ax=ax[0],
        level=5,
        cmap="ChaseSpectral",
        vmin=-10,
        vmax=80,
        quiverkey_len=10.0,
        background_field="DBZ",
        bg_grid_no=1,
        w_vel_contours=[1, 2, 5, 10],
        quiver_spacing_x_km=50.0,
        quiver_spacing_y_km=50.0,
        quiverkey_loc="bottom_right",
    )
    pydda.vis.plot_horiz_xsection_quiver(
        grid_tree["nest_1"],
        ax=ax[1],
        level=5,
        cmap="ChaseSpectral",
        vmin=-10,
        vmax=80,
        quiverkey_len=10.0,
        background_field="DBZ",
        bg_grid_no=1,
        w_vel_contours=[1, 2, 5, 10],
        quiver_spacing_x_km=50.0,
        quiver_spacing_y_km=50.0,
        quiverkey_loc="bottom_right",
    )
    return fig


# Common arguments for the VVAD constraint tests. The filtering and masking are
# turned off so the retrieved winds can be compared point by point.
VVAD_KWARGS = dict(
    Co=1.0,
    Cm=256.0,
    Cx=1e-2,
    Cy=1e-2,
    Cz=1e-2,
    max_iterations=30,
    vel_name="corrected_velocity",
    refl_field="reflectivity",
    low_pass_filter=False,
    mask_outside_opt=False,
    mask_w_outside_opt=False,
)


def _twpice_vvad_grids():
    """The TWP-ICE grid pair with the VVAD constraint fields attached."""
    Grid0, Grid1 = _twpice_grids()
    return pydda.constraints.make_constraint_from_vvad(
        [Grid0, Grid1], vel_field="corrected_velocity"
    )


def test_vvad_constraint_on_real_grids():
    """The VVAD reconstruction on real radar data must be physically
    plausible. This is the regression guard on the acceptance criteria: a
    level fit from a narrow sector of azimuth yields deformation terms orders
    of magnitude too large, which the reconstruction turns into winds of
    hundreds of m/s."""
    Grids = _twpice_vvad_grids()
    u_vad = Grids[0]["U_vvad"].values.squeeze()
    v_vad = Grids[0]["V_vvad"].values.squeeze()

    # The constraint must actually cover a useful part of the domain.
    assert 0.25 < np.isfinite(u_vad).mean() < 1.0
    assert np.nanmax(np.abs(u_vad)) < 60.0
    assert np.nanmax(np.abs(v_vad)) < 60.0

    for Grid in Grids:
        profiles = pydda.constraints.vvad_retrieval(
            Grid, vel_field="corrected_velocity"
        )
        ok = profiles["valid"].values
        assert ok.sum() > 10
        # A mean divergence over a 150 km domain is O(1e-4) per second; 1e-3
        # would extrapolate to winds of over 100 m/s at the domain edge.
        assert np.all(np.abs(profiles["DIV"].values[ok]) < 1e-3)
        assert np.all(np.abs(profiles["U0"].values[ok]) < 60.0)


def test_vvad_weights_match_ivad_switch():
    """parameters.vad_weights must implement i_vad of Eq. (19) of Protat et
    al. (2024) exactly: on where at most one radar contributes and the VVAD
    has a value, off everywhere else."""
    Grids = _twpice_vvad_grids()
    u_vad = Grids[0]["U_vvad"].values.squeeze()
    v_vad = Grids[0]["V_vvad"].values.squeeze()

    _, parameters = pydda.retrieval.get_dd_wind_field(
        deepcopy(Grids), Cvad=1.0, engine="scipy", **VVAD_KWARGS
    )

    n_rad = np.sum(parameters.weights, axis=0)
    expected = np.logical_and(
        n_rad <= 1, np.logical_and(np.isfinite(u_vad), np.isfinite(v_vad))
    )
    np.testing.assert_array_equal(parameters.vad_weights > 0, expected)

    # The TWP-ICE pair genuinely has both regimes, so this is a real test.
    assert (n_rad > 1).any() and (n_rad <= 1).any()
    assert not np.any(parameters.vad_weights[n_rad > 1] > 0)


def test_vvad_constraint_pulls_analysis_toward_vad():
    """Turning the VVAD constraint on must move the analysis toward the VVAD
    winds where the constraint is active."""
    Grids = _twpice_vvad_grids()
    u_vad = np.nan_to_num(Grids[0]["U_vvad"].values.squeeze())
    v_vad = np.nan_to_num(Grids[0]["V_vvad"].values.squeeze())

    off, _ = pydda.retrieval.get_dd_wind_field(
        deepcopy(Grids), Cvad=0.0, engine="scipy", **VVAD_KWARGS
    )
    on, parameters = pydda.retrieval.get_dd_wind_field(
        deepcopy(Grids), Cvad=1.0, engine="scipy", **VVAD_KWARGS
    )

    active = parameters.vad_weights > 0
    assert active.any()

    def departure(Grids_out):
        u = Grids_out[0]["u"].values.squeeze()
        v = Grids_out[0]["v"].values.squeeze()
        return np.mean(
            np.sqrt((u[active] - u_vad[active]) ** 2 + (v[active] - v_vad[active]) ** 2)
        )

    assert departure(on) < departure(off)


def test_vvad_requires_constraint_fields():
    """Cvad without the VVAD fields is a clear error rather than a crash."""
    Grid0, Grid1 = _twpice_grids()
    with pytest.raises(ValueError, match="U_vvad"):
        pydda.retrieval.get_dd_wind_field(
            [Grid0, Grid1], Cvad=1.0, engine="scipy", **VVAD_KWARGS
        )


@pytest.mark.parametrize("engine", ["tensorflow", "auglag"])
def test_vvad_unsupported_engines(engine):
    """The VVAD constraint is implemented for the scipy and jax engines only."""
    Grids = _twpice_vvad_grids()
    with pytest.raises(NotImplementedError, match="VVAD"):
        pydda.retrieval.get_dd_wind_field(Grids, Cvad=1.0, engine=engine, **VVAD_KWARGS)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="Jax not installed")
def test_vvad_constraint_jax():
    """The jax engine must reach the same solution as scipy with the VVAD
    constraint active."""
    Grids = _twpice_vvad_grids()
    scipy_out, _ = pydda.retrieval.get_dd_wind_field(
        deepcopy(Grids), Cvad=1.0, engine="scipy", **VVAD_KWARGS
    )
    jax_out, _ = pydda.retrieval.get_dd_wind_field(
        deepcopy(Grids), Cvad=1.0, engine="jax", **VVAD_KWARGS
    )

    u_scipy = np.asarray(scipy_out[0]["u"].values.squeeze())
    u_jax = np.asarray(jax_out[0]["u"].values.squeeze())
    assert np.nanmean(np.abs(u_scipy - u_jax)) < 0.5
