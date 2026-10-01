.. _optimizing-wind-retrieval:


Optimizing your wind retrieval
==============================

In the :ref:`retrieving_winds` section, we showed how to perform an example wind retrieval
with PyDDA. However, there were some issues to be resolved in the wind retrieval, including
artificial updrafts at the boundaries of the Dual Doppler lobes caused by discontinuities
in the horizontal winds from changing data sources from the radar network to the
model constraint. In this section, we will show how to adjust the parameters of your
wind retrieval in order to minimize such artifacts. First, we will show the wind retrieval
that we did in :ref:`retrieving_winds`.


.. code-block:: python

    grids_out, _ = pydda.retrieval.get_dd_wind_field([grid_kict, grid_ktlx],
                                                Cm=256.0, Co=1e-2, Cx=1, Cy=1,
                                                Cz=1, Cmod=1e-5, model_fields=["hrrr"],
                                                refl_field='DBZ', wind_tol=0.5,
                                                max_iterations=100, filter_window=15,
                                                filter_order=3, engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                     quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                     w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                     quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

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
                                     quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                    w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                    quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

We can see several potential issues with the wind retrieval. First, there are artifacts at the
Dual Doppler lobe edges where updrafts are being produced by the optimization code simply
because of a discontinuity in the horizontal winds at the edges of the lobes. In addition,
there are other discontinuities in the horizontal winds that should be addressed. One thing
we can do to mitigate these discontinuities is to increase the weight of the horizontal
smoothness constraints. Therefore, let's prescribe :code:`Cx = 100.` and :code:`Cy = 100`
to the above retrieval.

.. code-block:: python

    grids_out, _ = pydda.retrieval.get_dd_wind_field([grid_kict, grid_ktlx],
                                                Cm=256.0, Co=1e-2, Cx=100, Cy=100,
                                                Cz=1, Cmod=1e-5, model_fields=["hrrr"],
                                                refl_field='DBZ', wind_tol=0.5,
                                                max_iterations=100, filter_window=15,
                                                filter_order=3, engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                     quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                     w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                     quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

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
        Cm=256.0, Co=1e-2, Cx=100, Cy=100, Cz=1, Cmod=1e-5, model_fields=["hrrr"],
        refl_field='DBZ', wind_tol=0.5, max_iterations=100, filter_window=15, filter_order=3,
        engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                        quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                        w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                        quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

As we can see, the artifact at the edge of the Dual Doppler lobe has reduced in size. However, we
also have lost some detail on the updraft structure at this level because the wind field
has been smoothed out. This therefore coarsens the effective resolution of the retrieval.
Let's see what happens when we increase the level of smoothing.

.. code-block:: python

    grids_out, _ = pydda.retrieval.get_dd_wind_field([grid_kict, grid_ktlx],
                                                Cm=256.0, Co=1e-2, Cx=250., Cy=250.,
                                                Cz=250.0, Cmod=1e-5, model_fields=["hrrr"],
                                                refl_field='DBZ', wind_tol=0.5,
                                                max_iterations=100, filter_window=15,
                                                filter_order=3, engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                     quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                     w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                     quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

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
        Cm=256.0, Co=1e-2, Cx=250., Cy=250., Cz=250., Cmod=1e-5, model_fields=["hrrr"],
        refl_field='DBZ', wind_tol=0.5, max_iterations=1050, filter_window=15, filter_order=3,
        engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                        quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                        w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                        quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

In the above retrieval, the updrafts appear to be smoothed out. To help the optimization loop
resolve the updrafts, we recommend, from here, decreasing the tolerance required for the
optimization loop to converge. In addition, decreasing the smoothness will allow more details
of the resolved wind field to appear. In the below example, we observe this, though part of the
artifact near the edge of the Dual Doppler lobe re-appears.

.. code-block:: python

    grids_out, _ = pydda.retrieval.get_dd_wind_field([grid_kict, grid_ktlx],
                                                Cm=256.0, Co=1e-2, Cx=150., Cy=150.,
                                                Cz=150.0, Cmod=1e-5, model_fields=["hrrr"],
                                                refl_field='DBZ', wind_tol=0.1,
                                                max_iterations=400, filter_window=15,
                                                filter_order=3, engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                     quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                     w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                     quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

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
        Cm=256.0, Co=1e-2, Cx=150., Cy=150., Cz=150., Cmod=1e-5, model_fields=["hrrr"],
        refl_field='DBZ', wind_tol=0.5, max_iterations=1050, filter_window=15, filter_order=3,
        engine='scipy')

    pydda.vis.plot_horiz_xsection_quiver(grids_out, level=15, cmap='ChaseSpectral', vmin=-10, vmax=80,
                                        quiverkey_len=20.0, background_field='DBZ', bg_grid_no=1,
                                        w_vel_contours=[1, 2, 5, 10], quiver_spacing_x_km=10.0,
                                        quiver_spacing_y_km=10.0, quiverkey_loc='bottom_right')

Generally, these parameters need to be tuned for your particular radar configuration in
order to obtain the most optimal wind retrieval for your situation. If you are placing
more importance on horizontal winds compared to updraft velocities, then you may be willing
to tolerate more errors in the vertical velocity field so that finer details of the horizontal
wind field can be generated. The above parameters are examples that apply to a 1 km
resolution grid from two NEXRADs and vary for given radar configurations and storm coverages.

------------------------------------
Choosing an upper boundary condition
------------------------------------

PyDDA constrains the vertical velocity at the boundaries of the analysis domain by
zeroing the gradient of the cost function with respect to :math:`w` there, so that
:math:`w` is held at its first guess. At the surface this impermeability condition is
always applied. At the top of the domain it is controlled by the :code:`upper_bc`
keyword of :meth:`pydda.retrieval.get_dd_wind_field`, which takes one of three values:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - :code:`upper_bc`
      - Condition applied
    * - :code:`0`
      - No condition at the top of the domain.
    * - :code:`1`
      - :math:`w = 0` at the top vertical level of the grid.
    * - :code:`2`
      - :math:`w = 0` above the echo top, i.e. wherever no radar reports an
        observation and the point is higher than *above* km in the grid's
        vertical coordinate.

The classic choice is :code:`upper_bc=1`, which imposes :math:`w = 0` at the top of the
analysis domain. That is only physically defensible when the domain top is genuinely
above the storm. If the grid is truncated through the middle of deep convection, the
condition forces the mass continuity constraint to close the divergence profile at an
arbitrary height and pushes a spurious compensating signal down into the levels you
actually care about.

Setting :code:`upper_bc=2` instead applies the impermeability condition at the *echo
top*: every grid point at which none of the radars report a valid radial velocity is
treated as being outside the storm, and :math:`w` is held at its first guess there.
Because the first guess for :math:`w` is normally zero, this is equivalent to requiring
that no mass crosses the top of the observed echo. The method is described in Thompson
et al. (2026), https://doi.org/10.5194/egusphere-2026-4631.

The :code:`above` keyword sets the lowest altitude, in km, at which the echo top
condition may be applied. It exists so that clear air at low levels -- gaps between
cells, the cone of silence, the far edge of the Dual Doppler lobes -- does not pin
:math:`w` to zero close to the surface, which would suppress the very updrafts you are
trying to retrieve. The default of 2 km is a reasonable starting point; raise it if
your radars have poor low-level coverage.

.. code-block:: python

    grids_out, _ = pydda.retrieval.get_dd_wind_field([grid_kict, grid_ktlx],
                                                Cm=256.0, Co=1e-2, Cx=1, Cy=1,
                                                Cz=1, Cmod=1e-5, model_fields=["hrrr"],
                                                refl_field='DBZ', wind_tol=0.5,
                                                max_iterations=50, filter_window=15,
                                                filter_order=3, engine='scipy',
                                                upper_bc=2, above=2.0)

Both :code:`upper_bc` and :code:`above` are supported by all of PyDDA's engines
(:code:`"scipy"`, :code:`"jax"`, :code:`"tensorflow"` and :code:`"auglag"`). The set of
points at which the condition is applied is calculated once at the start of the
retrieval by :meth:`pydda.cost_functions.calculate_echo_top_mask` and is returned on the
:code:`upper_bc_mask` attribute of the parameters object, so it can be inspected
afterwards:

.. code-block:: python

    grids_out, parameters = pydda.retrieval.get_dd_wind_field(
        [grid_kict, grid_ktlx], upper_bc=2, above=2.0, ...)
    print("Fraction of the domain held impermeable: %.2f"
          % parameters.upper_bc_mask.mean())

.. note::
    For backwards compatibility :code:`upper_bc=True` and :code:`upper_bc=False` are
    still accepted and mean the same as :code:`1` and :code:`0`.
