.. _retrieving_winds:

Retrieving your first wind field
================================

------------------------
3D variational technique
------------------------

PyDDA minimizes a cost function :math:`J` that corresponds to various penalties including:

+----------------------------------------------------------------+----------------------------+
|                     Cost Function                              |       Interpretation       |
+----------------------------------------------------------------+----------------------------+
| :math:`J_{m} = \nabla \cdot V + w \frac{d\rho}{dz}`            |  Mass continuity equation  |
+----------------------------------------------------------------+----------------------------+
| :math:`J_{o} = \sum_{radar} [V_{ar} - \textbf{V}]^2`           |  Radar winds               |
+----------------------------------------------------------------+----------------------------+
| :math:`J_{mod} = \sum_{domain} [V_{model} - \textbf{V}]^2`       |  Model winds               |
+----------------------------------------------------------------+----------------------------+
| :math:`J_{b} = \sum_{background} [V_{sounding} - \textbf{V}]^2`|  Sounding background       |
+----------------------------------------------------------------+----------------------------+
| :math:`J_{s} = \nabla^2 V`                                     |  Wind field smoothness     |
+----------------------------------------------------------------+----------------------------+
| :math:`J_{vad} = \sum_{i_{vad}} [V_{VVAD} - \textbf{V}]^2`      |  VVAD winds                |
+----------------------------------------------------------------+----------------------------+

The cost function to be minimized is a weighted sum of the various cost functions in PyDDA and are represented in Equation (1):

:math:`J = c_{m}J_{m} + c_{o}J_{o} + c_{b}J_{b} + c_{s}J_{s} + ...` (1)

---------------------------------------------
Filling data voids with the VVAD constraint
---------------------------------------------

The dual-Doppler problem is only well posed where two radars observe the same
point from sufficiently different directions. Outside the dual-Doppler lobes,
and at the low levels that lie below the lowest radar gate, the horizontal wind
is underdetermined by the radial velocities alone.

:math:`J_{vad}` addresses this following the SWIRL system of
`Protat et al. (2024) <https://doi.org/10.1175/JTECH-D-23-0155.1>`_. A
variational velocity azimuth display (VVAD) fits vertical profiles of a linear
wind model - :math:`U_0`, :math:`V_0`, the divergence, and the stretching and
shearing deformations - to each radar's Doppler velocities. Those profiles
reconstruct a horizontal wind at *every* point within range of the radar,
including points with no observation at all, which is what lets the term fill
data voids.

Following Eq. (19) of Protat et al. (2024), the term carries a switch
:math:`i_{vad}` that is zero wherever more than one radar contributes, so the
VVAD never competes with genuine multi-Doppler information. PyDDA computes the
switch automatically from the same radar coverage arrays it uses to weight the
radial velocity term.

.. code-block:: python

    Grids = pydda.constraints.make_constraint_from_vvad(Grids)
    Grids, parameters = pydda.retrieval.get_dd_wind_field(Grids, Cvad=1.0)

Protat et al. (2024) use a weight of 1, equal to that of the radial velocity
constraint. The constraint is available for the ``"scipy"`` and ``"jax"``
engines.

Two caveats are worth keeping in mind. First, every VAD technique assumes the
horizontal wind is linear across the domain, which is reasonable in stratiform
precipitation but not in convection. Second, PyDDA applies two acceptance
criteria beyond the point count used in the paper - a minimum azimuthal
coverage and a maximum condition number - because on a PyDDA analysis grid a
height level can hold hundreds of valid velocities confined to a narrow sector
of azimuth, which cannot determine the fit. See
:py:func:`pydda.constraints.vvad_retrieval` for details.

PyDDA does not implement the double VAD (DVAD) that Protat et al. (2024) use
where two suitably spaced radars overlap, nor the optical flow term of their
Eq. (19).

-------------------------------
Doing your first wind retrieval
-------------------------------

The next step is to use PyDDA to retrieve the input winds! In order to perform a wind
retrieval, there are many aspects that must be considered. After the data processing has
finished, it is now important to constrain the wind field further by adding in either sounding,
point, or model data as a weak constraint in order to increase the chance that PyDDA will
provide a solution that converges to a physically realistic wind field. For this particular example,
we are lucky enough to have model data from the Rapid Update Cycle (RUC; the predecessor to the current Rapid Refresh/RAP) that can be used as a constraint.

------------------------
Using PyDDA's data model
------------------------

As of PyDDA 2.0, PyDDA uses `xarray Datasets <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_
to represent the underlying datastructure. This makes it easier to integrate PyDDA into xarray-based workflows
that are the standard for use in the Geoscientific Python Community. Therefore, anyone using PyART Grids will need
to convert their grids to PyDDA Grids using the :meth:`pydda.io.read_from_pyart_grid` helper function.

.. code-block:: python

    grid_ktlx = pydda.io.read_from_pyart_grid(grid_ktlx)
    grid_kict = pydda.io.read_from_pyart_grid(grid_kict)

In addition, :meth:`pydda.io.read_grid` will read a cf-Compliant radar grid into a PyDDA Grid.

----------------------------
Using models for constraints
----------------------------

In this example, we will use model data from the Rapid Update Cycle (RUC) in order to
provide a constraint on the horizontal winds. This helps to constrain the background
area where there is suboptimal coverage from the radar network outside of the dual
Doppler lobes. To add either a RUC or HRRR model time period as a constraint, we need
to have the original model data in GRIB format and then use the following line to
load the model data into a PyDDA grid for processing.

.. code-block:: python

    # Add constraints
    grid_kict = pydda.constraints.add_hrrr_constraint_to_grid(grid_kict,
        pydda.tests.get_sample_file('ruc2anl_130_20110520_0800_001.grb2'), method='linear')

--------------------------
Set up your initialization
--------------------------

In addition to adding more constraints, PyDDA requires a first guess of the wind field in order
to start performing the optimization loop that finds the wind field that minimizes :math:`J`. For this
example, we will start with a zero initial wind field. The following code snippet will initialize PyDDA
with a zero wind field:

.. code-block:: python

    grid_kict = pydda.initialization.make_constant_wind_field(grid_kict, (0.0, 0.0, 0.0))

--------------------------------
Retrieving your first wind field
--------------------------------

We will then take a first attempt at retrieving a wind field using PyDDA. This is done using the
:meth:`pydda.retrieval.get_dd_wind_field` function. This function takes in a minimum of one input, a list
of input PyART Grids. We will also specify the constants for the constraints. In this case, we are using
the mass continuity, radar observation, smoothness, and model constraints.

.. code-block:: python

    grids_out, _ = pydda.retrieval.get_dd_wind_field([grid_kict, grid_ktlx],
                                                Cm=256.0, Co=1e-2, Cx=1, Cy=1,
                                                Cz=1, Cmod=1e-5, model_fields=["hrrr"],
                                                refl_field='DBZ', wind_tol=0.5,
                                                max_iterations=50, filter_window=15,
                                                filter_order=3, engine='scipy')

-------------------------
Plotting the output winds
-------------------------

PyDDA contains visualization routines to create barb, quiver, and streamline plots of your wind fields
overlaid on gridded radar variables. Further detail about these visualization routines is contained in the
:ref:`visualizing-winds` section. For this example, we will use :code:`pydda.vis.plot_horiz_xsection_quiver` to
create a quiver plot of the wind field. In this example, we are plotting winds at the 15th vertical level of the
grid, with the background field being reflectivity with the colormap spanning -10 to 80 ref. We will specify 25 km
horizontal spacing between quivers and a quiver key length of 10 m/s. Finally, we specify that we will place the quiver
key on the bottom right interior of the plot.


.. code-block:: python

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                     quiverkey_len=10.0, background_field='DBZ', bg_grid_no=1,
                                     w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=25.0,
                                     quiver_spacing_y_km=25.0, quiverkey_loc='bottom_right')

.. plot::

    import warnings

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np

    import pyart
    import pydda
    from pyart.testing import get_test_data

    warnings.filterwarnings("ignore")

    # read in the data from both XSAPR radars
    ktlx_file = pydda.tests.get_sample_file("cfrad.20110520_081431.542_to_20110520_081813.238_KTLX_SUR.nc")
    kict_file = pydda.tests.get_sample_file("cfrad.20110520_081444.871_to_20110520_081914.520_KICT_SUR.nc")
    radar_ktlx = pyart.io.read_cfradial(ktlx_file)
    radar_kict = pyart.io.read_cfradial(kict_file)


    # Calculate the Velocity Texture and apply the PyART GateFilter Utility
    vel_tex_ktlx = pyart.retrieve.calculate_velocity_texture(radar_ktlx,
                                                           vel_field='VEL',
                                                           )
    vel_tex_kict = pyart.retrieve.calculate_velocity_texture(radar_kict,
                                                           vel_field='VEL',
                                                           )

    ## Add velocity texture to the radar objects
    radar_ktlx.add_field('velocity_texture', vel_tex_ktlx, replace_existing=True)
    radar_kict.add_field('velocity_texture', vel_tex_kict, replace_existing=True)

    # Apply a GateFilter
    gatefilter_ktlx = pyart.filters.GateFilter(radar_ktlx)
    gatefilter_ktlx.exclude_above('velocity_texture', 3)
    gatefilter_kict = pyart.filters.GateFilter(radar_kict)
    gatefilter_kict.exclude_above('velocity_texture', 3)

    # Apply Region Based DeAlising Utiltiy
    vel_dealias_ktlx = pyart.correct.dealias_region_based(radar_ktlx,
                                                        vel_field='VEL',
                                                        centered=True,
                                                        gatefilter=gatefilter_ktlx
                                                        )

    # Apply Region Based DeAlising Utiltiy
    vel_dealias_kict = pyart.correct.dealias_region_based(radar_kict,
                                                        vel_field='VEL',
                                                        centered=True,
                                                        gatefilter=gatefilter_kict
                                                        )

    # Add our data dictionary to the radar object
    radar_kict.add_field('corrected_velocity', vel_dealias_kict, replace_existing=True)
    radar_ktlx.add_field('corrected_velocity', vel_dealias_ktlx, replace_existing=True)

    grid_limits = ((0., 15000.), (-300000., -100000.), (-250000., 0.))
    grid_shape = (31, 201, 251)

    grid_ktlx = pyart.map.grid_from_radars([radar_ktlx], grid_limits=grid_limits,
                                 grid_shape=grid_shape, gatefilter=gatefilter_ktlx,
                                    grid_origin=(radar_kict.latitude['data'].filled(),
                                                 radar_kict.longitude['data'].filled()))
    grid_kict = pyart.map.grid_from_radars([radar_kict], grid_limits=grid_limits,
                                 grid_shape=grid_shape, gatefilter=gatefilter_kict,
                                    grid_origin=(radar_kict.latitude['data'].filled(),
                                                 radar_kict.longitude['data'].filled()))
    grid_ktlx = pydda.io.read_from_pyart_grid(grid_ktlx)
    grid_kict = pydda.io.read_from_pyart_grid(grid_kict)
    grid_kict = pydda.constraints.add_hrrr_constraint_to_grid(grid_kict,
        pydda.tests.get_sample_file('ruc2anl_130_20110520_0800_001.grb2'), method='linear')

    grid_kict = pydda.initialization.make_constant_wind_field(grid_kict, (0.0, 0.0, 0.0))
    grids_out, _ = pydda.retrieval.get_dd_wind_field([grid_kict, grid_ktlx],
        Cm=256.0, Co=1e-2, Cx=1, Cy=1, Cz=1, Cmod=1e-5, model_fields=["hrrr"],
        refl_field='DBZ', wind_tol=0.5, max_iterations=50, filter_window=15, filter_order=3,
        engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                     quiverkey_len=10.0, background_field='DBZ', bg_grid_no=1,
                                     w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=25.0,
                                     quiver_spacing_y_km=25.0, quiverkey_loc='bottom_right')

We can see in this figure that PyDDA is resolving numerous updrafts in the mid-levels. One thing to be aware of
is the vertical motion at the edge of the Dual Doppler lobe in the top right corner. This vertical motion is likely
caused by the wind source changing from primarily the radar data to the RUC model run outside of the Dual Doppler
lobes, causing a slight shift in winds that results in horizontal convergence. This convergence will result in an
updraft in the domain that is an artifact of this switch in data sources. It is therefore recommended to not
use vertical velocity data in updrafts that are touching the Dual Doppler lobe edges to mitigate this issue. In
addition, prescribing a stronger background constraint or filtering the data more often may also help mitigate this
issue. We will go into this further in :ref:`optimizing-wind-retrieval`.
