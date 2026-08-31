"""
========================================
pydda.constraints (pydda.constraints)
========================================

.. currentmodule:: pydda.constraints

The procedures in this module are used to add spatial fields from non-radar
based datasets for use as a constraint. The model cost function uses the
observations inserted into the Grid object from these procedures as a
constraint. In order to develop your own custom constraint here, simply
create a function that adds 3 fields into the input Py-ART Grid with names
"u_(name)", "v_(name)", and "w_(name)" where (name) is the name of your
dataset. Then, in order to have PyDDA use this dataset as a constraint,
simply add (name) into the model_fields option of
:py:func:`get_dd_wind_field`.

.. autosummary::
    :toctree: generated/

     make_constraint_from_wrf
     add_hrrr_constraint_to_grid
     make_constraint_from_era5
     download_needed_era_data
     get_iem_obs
     make_constraint_from_vvad
     vvad_retrieval
     vvad_horizontal_wind

"""

from .model_data import make_constraint_from_wrf
from .model_data import add_hrrr_constraint_to_grid
from .model_data import make_constraint_from_era5
from .model_data import download_needed_era_data
from .station_data import get_iem_obs
from .vad_data import make_constraint_from_vvad
from .vad_data import vvad_retrieval
from .vad_data import vvad_horizontal_wind
