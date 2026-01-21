#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : armand, éléonore
Main script to run angle estimation from simulated TOD.
"""

#%% Imports
import tod_simulation as ts
import angle_estimation as ae

import jax.numpy as jnp

#%% TOD and angle estimation

CALCULATE = True
if CALCULATE:
    _, _, tods = ts.detectors_output()
    tod = tods[0]  # Select the TOD of the first detector
else:
    tod = jnp.load(f'detectors_res{int(ts.realistic_hwp)}.npy')[0]  # Load precomputed TOD for faster testing

print("*** Detector TOD loaded. ***")

alpha_reconstructed = ae.estimangle(tod)

print(f"Reconstructed angle (deg):", alpha_reconstructed)
print(f"Delta (deg):", alpha_reconstructed - jnp.degrees(ts.alpha_drone))