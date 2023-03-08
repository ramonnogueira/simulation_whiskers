# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:01:05 2023

@author: danie
"""

import os
import h5py
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False
    
