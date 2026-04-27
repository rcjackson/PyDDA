"""
===========================================
pydda.initialization (pydda.initialization)
===========================================

.. currentmodule:: pydda.initialization

The module containing the core techniques for the multiple doppler
wind retrieval. All of these techniques take in data from
a desired format and will return a 3-tuple of numpy arrays that
are in the same shape as the input Py-ART Grid object used
for analysis. If you wish to add another initialization here,
add a procedure that takes in a Py-Art Grid that is used for
the grid specification (shape, x, y, z) and the dataset of your
choice. Your output from the function should then be a 3-tuple
of numpy arrays with the same shape as the fields in Grid.

.. autosummary::
    :toctree: generated/

    make_constant_wind_field
    make_wind_field_from_profile
    make_background_from_wrf
    make_initialization_from_era5
    make_initialization_from_iem_obs

"""

from .wind_fields import make_constant_wind_field
from .wind_fields import make_wind_field_from_profile
from .wind_fields import make_background_from_wrf
from .wind_fields import make_initialization_from_era5
from .wind_fields import make_initialization_from_iem_obs
