.. _visualizing-winds:

Visualizing the wind retrieval
==============================

PyDDA's :mod:`pydda.vis` module provides routines for plotting horizontal and
vertical cross-sections of retrieved wind fields overlaid on a gridded radar
background field (e.g., reflectivity). Three wind-vector styles are supported:

* **Quivers** — arrow length proportional to wind speed.
* **Barbs** — meteorological wind barbs.
* **Streamlines** — continuous flow lines.

Each style has three geometry variants:

+---------------------------------------------------+---------------------------------------------+
| Function                                          | Cross-section                               |
+===================================================+=============================================+
| :func:`pydda.vis.plot_horiz_xsection_quiver`      | Horizontal (constant altitude)              |
+---------------------------------------------------+---------------------------------------------+
| :func:`pydda.vis.plot_xz_xsection_quiver`         | Vertical, east–west (constant y)            |
+---------------------------------------------------+---------------------------------------------+
| :func:`pydda.vis.plot_yz_xsection_quiver`         | Vertical, north–south (constant x)          |
+---------------------------------------------------+---------------------------------------------+
| :func:`pydda.vis.plot_horiz_xsection_barbs`       | Horizontal (constant altitude)              |
+---------------------------------------------------+---------------------------------------------+
| :func:`pydda.vis.plot_xz_xsection_barbs`          | Vertical, east–west (constant y)            |
+---------------------------------------------------+---------------------------------------------+
| :func:`pydda.vis.plot_yz_xsection_barbs`          | Vertical, north–south (constant x)          |
+---------------------------------------------------+---------------------------------------------+
| :func:`pydda.vis.plot_horiz_xsection_streamline`  | Horizontal (constant altitude)              |
+---------------------------------------------------+---------------------------------------------+

---------------------------------
Horizontal cross-section (quiver)
---------------------------------

The most common visualization is a horizontal cross-section of horizontal winds
overlaid on reflectivity. The example below plots the wind field at vertical
level 15, with updraft contours at 1, 2, 5, and 10 m/s and quivers spaced
25 km apart.

.. code-block:: python

    pydda.vis.plot_horiz_xsection_quiver(
        grids_out,
        level=15,
        cmap="ChaseSpectral",
        vmin=-10,
        vmax=80,
        quiverkey_len=10.0,
        background_field="DBZ",
        bg_grid_no=1,
        w_vel_contours=[1, 2, 5, 10],
        quiver_spacing_x_km=25.0,
        quiver_spacing_y_km=25.0,
        quiverkey_loc="bottom_right",
    )

.. plot::

    import warnings

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np

    import pyart
    import pydda

    warnings.filterwarnings("ignore")

    ktlx_file = pydda.tests.get_sample_file("cfrad.20110520_081431.542_to_20110520_081813.238_KTLX_SUR.nc")
    kict_file = pydda.tests.get_sample_file("cfrad.20110520_081444.871_to_20110520_081914.520_KICT_SUR.nc")
    radar_ktlx = pyart.io.read_cfradial(ktlx_file)
    radar_kict = pyart.io.read_cfradial(kict_file)

    vel_tex_ktlx = pyart.retrieve.calculate_velocity_texture(radar_ktlx, vel_field='VEL')
    vel_tex_kict = pyart.retrieve.calculate_velocity_texture(radar_kict, vel_field='VEL')
    radar_ktlx.add_field('velocity_texture', vel_tex_ktlx, replace_existing=True)
    radar_kict.add_field('velocity_texture', vel_tex_kict, replace_existing=True)

    gatefilter_ktlx = pyart.filters.GateFilter(radar_ktlx)
    gatefilter_ktlx.exclude_above('velocity_texture', 3)
    gatefilter_kict = pyart.filters.GateFilter(radar_kict)
    gatefilter_kict.exclude_above('velocity_texture', 3)

    vel_dealias_ktlx = pyart.correct.dealias_region_based(
        radar_ktlx, vel_field='VEL', centered=True, gatefilter=gatefilter_ktlx)
    vel_dealias_kict = pyart.correct.dealias_region_based(
        radar_kict, vel_field='VEL', centered=True, gatefilter=gatefilter_kict)

    radar_kict.add_field('corrected_velocity', vel_dealias_kict, replace_existing=True)
    radar_ktlx.add_field('corrected_velocity', vel_dealias_ktlx, replace_existing=True)

    grid_limits = ((0., 15000.), (-300000., -100000.), (-250000., 0.))
    grid_shape = (31, 201, 251)

    grid_ktlx = pyart.map.grid_from_radars(
        [radar_ktlx], grid_limits=grid_limits, grid_shape=grid_shape,
        gatefilter=gatefilter_ktlx,
        grid_origin=(radar_kict.latitude['data'].filled(),
                     radar_kict.longitude['data'].filled()))
    grid_kict = pyart.map.grid_from_radars(
        [radar_kict], grid_limits=grid_limits, grid_shape=grid_shape,
        gatefilter=gatefilter_kict,
        grid_origin=(radar_kict.latitude['data'].filled(),
                     radar_kict.longitude['data'].filled()))

    grid_ktlx = pydda.io.read_from_pyart_grid(grid_ktlx)
    grid_kict = pydda.io.read_from_pyart_grid(grid_kict)
    grid_kict = pydda.constraints.add_hrrr_constraint_to_grid(
        grid_kict, pydda.tests.get_sample_file('ruc2anl_130_20110520_0800_001.grb2'),
        method='linear')
    grid_kict = pydda.initialization.make_constant_wind_field(grid_kict, (0.0, 0.0, 0.0))

    grids_out, _ = pydda.retrieval.get_dd_wind_field(
        [grid_kict, grid_ktlx],
        Cm=256.0, Co=1e-2, Cx=1, Cy=1, Cz=1, Cmod=1e-5,
        model_fields=["hrrr"], refl_field='DBZ', wind_tol=0.5,
        max_iterations=50, filter_window=15, filter_order=3, engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(
        grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
        quiverkey_len=10.0, background_field='DBZ', bg_grid_no=1,
        w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=25.0,
        quiver_spacing_y_km=25.0, quiverkey_loc='bottom_right')

Key parameters
--------------

``level``
    The index of the vertical level (z dimension) to plot for horizontal
    cross-sections.

``background_field``
    The name of the gridded variable to show as the color-filled background
    (e.g., ``"DBZ"`` for reflectivity).

``bg_grid_no``
    Which grid in the input list to use for the background field. Defaults to 0.

``w_vel_contours``
    List of vertical velocity values (m/s) at which to draw contours. Set to
    ``None`` to suppress vertical velocity contours.

``quiver_spacing_x_km`` / ``quiver_spacing_y_km``
    Horizontal spacing between quiver arrows in kilometers. Larger values
    produce a less cluttered plot.

``quiverkey_len``
    The reference wind speed (m/s) represented by the quiver key arrow.

For full parameter descriptions see the :ref:`user` reference guide.
