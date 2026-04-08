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

fig = plt.figure(figsize=(8, 6))
display = pyart.graph.RadarDisplay(radar_ktlx)
display.plot_ppi('velocity_texture',
                     sweep=0,
                     vmin=0,
                     vmax=10,
                     cmap=plt.get_cmap('twilight_shifted')
                     )

# Plot a histogram of the velocity textures
fig = plt.figure(figsize=[8, 8])
hist, bins = np.histogram(radar_ktlx.fields['velocity_texture']['data'],
                          bins=np.linspace(0, 20, 150))
bins = (bins[1:]+bins[:-1])/2.0
plt.plot(bins,
         hist,
         label='Velocity Texture Frequency'
         )
plt.axvline(3,
            color='r',
            label='Proposed Velocity Texture Threshold'
            )
plt.xlabel('Velocity texture')
plt.ylabel('Count')
plt.legend()