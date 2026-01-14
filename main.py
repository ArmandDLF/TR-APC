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
    tod = jnp.load('detectors_res.npy')[0]  # Load precomputed TOD for faster testing


alpha_reconstructed = ae.estimangle(tod)

print(f"Reconstructed angle (deg):", alpha_reconstructed)
print("True angle (deg):", jnp.degrees(ts.alpha_drone))