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


fig = plt.figure(figsize=(16, 6))
ax = plt.subplot(121, projection=ccrs.PlateCarree())

# Plot the southwestern radar
disp1 = pyart.graph.RadarMapDisplay(radar_ktlx)
disp1.plot_ppi_map(
    "DBZ",
    sweep=1,
    ax=ax,
    vmin=-20,
    vmax=70,
    min_lat=36,
    max_lat=37,
    min_lon=-98,
    max_lon=-97,
    lat_lines=np.arange(36, 37.25, 0.25),
    lon_lines=np.arange(-98, -96.75, 0.25),
)
# Plot the southeastern radar
ax2 = plt.subplot(122, projection=ccrs.PlateCarree())
disp2 = pyart.graph.RadarMapDisplay(radar_kict)
disp2.plot_ppi_map(
    "DBZ",
    sweep=1,
    ax=ax2,
    vmin=-20,
    vmax=70,
    min_lat=36,
    max_lat=37,
    min_lon=-98,
    max_lon=-97,
    lat_lines=np.arange(36, 37.25, 0.25),
    lon_lines=np.arange(-98, -96.75, 0.25),
)