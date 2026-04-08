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