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

fig = plt.figure(figsize=(16, 6))

# Plot the southwestern radar
ax = plt.subplot(121, projection=ccrs.PlateCarree())
disp1 = pyart.graph.RadarMapDisplay(radar_ktlx)
disp1.plot_ppi_map("corrected_velocity",
                   sweep=1,
                   ax=ax,
                   vmin=-35,
                   vmax=35,
                   min_lat=36,
                   max_lat=37,
                   min_lon=-98,
                   max_lon=-97,
                   lat_lines=np.arange(36, 37.25, 0.25),
                   lon_lines=np.arange(-98, -96.75, 0.25),
                   cmap=plt.get_cmap('twilight_shifted')
)

# Plot the southeastern radar
ax2 = plt.subplot(122, projection=ccrs.PlateCarree())
disp2 = pyart.graph.RadarMapDisplay(radar_kict)
disp2.plot_ppi_map("corrected_velocity",
                   sweep=1,
                   ax=ax2,
                   vmin=-35,
                   vmax=35,
                   min_lat=36,
                   max_lat=37,
                   min_lon=-98,
                   max_lon=-97,
                   lat_lines=np.arange(36, 37.25, 0.25),
                   lon_lines=np.arange(-98, -96.75, 0.25),
                   cmap=plt.get_cmap('twilight_shifted')
)