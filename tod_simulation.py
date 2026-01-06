#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author : armand, éléonore

TOD detector simulation,
furax integration of Anouk's code "Testing Spectral Likelihood for Drone Data Analysis"
"""

#%% Imports
import sys
import numpy as np
import healpy as hp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


from furax.obs import LinearPolarizerOperator, ActualLinearPolarizerOperator
from furax.obs import HWPOperator, WPOperator
from furax.core import HomothetyOperator

from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import Stokes
from furax.mapmaking.pointing import PointingOperator
from furax.math.quaternion import from_lonlat_angles

from scipy import signal

#%% General parameters

arcmin2rad = 1.0 / 60 * jnp.pi / 180.0  # conversion factor arcmin to radians
Nside = 512  # Healpix resolution parameter (number of pixels = 12*Nside^2)

I0 = 1000  # reference intensity
noise_level = I0 / 100  # noise amplitude

scan_type = 1  # 0: staring at source; 1: scanning at constant elevation
azel_source_input = (jnp.pi / 2, 45 * jnp.pi / 180.0)  # source azimuth, elevation (radians)

# Drone amplitude variations parameters
drone_amplitude = I0 / 10  # drone amplitude
delta_variations1 = 0.10  # 10% amplitude variation
delta_variations2 = 0.75  # 75% amplitude variation
variations_omega1 = 0.2  # frequency 1 (Hz)
variations_omega2 = 1.0  # frequency 2 (Hz)

# Healpy expects (colatitude, longitude)  : colatitude = pi/2 - elevation, and longitude = azimuth.
ipix_source_input = hp.ang2pix(Nside, 0.5 * jnp.pi - azel_source_input[1], azel_source_input[0])

# Drone emission
beam_fwhm = 60  # beam FWHM in arcminutes
alpha_drone = 10.0 * jnp.pi / 180  # drone polarization angle in radians
true_det_angle = 5.0 * jnp.pi / 180  # true detector angle in radians
source_type = 'ic'

# HWP and chopper parameters
f_chop = 37  # chopping frequency (Hz)
phi_chop = 0.0 # chopping phase (radians) 

# Constants for VPM and HWP
z_0 = 6.e-3  # modulation amplitude (m)
f_vpm = 10.  # VPM frequency (Hz)
f_hwp = 2.0  # frequency of half-wave plate rotation (Hz)
demod_mode = 4  # demodulation mode
hwp, vpm = True, False # Use one or the other

# Telescope parameters 
amp_scan = 0.02  # scanning amplitude
freq_scan = 0.1 # telescope scanning frequency (Hz)
n_detectors = 1


# Time
k = 15 # in seconds
sample_rate = 100  # in Hz
t_v = np.arange(0, k, 1.0/sample_rate) 

#%% Constructing normalized source map

def beam_conv(map_in, beam_fwhm=beam_fwhm):
    """
    Convolve input map with beam kernel (Gaussian smoothing).

    Parameters:
        map_in (jnp.array): input Healpix map
        beam_fwhm (float): beam full width half max in arcminutes

    Returns:
        jnp.array: beam-smoothed map
    """
    smoothed = hp.smoothing(np.asarray(map_in), beam_fwhm * arcmin2rad)
    return jax.device_put(smoothed)

def conv_source(beam_fwhm=beam_fwhm, Nside=Nside, ipix_source_input=ipix_source_input):
    """
    Convolve input source map with beam and normalize.    
    
    Parameters:
        beam_fwhm (float): beam full width half max in arcminutes
        Nside (int): Healpix Nside parameter
        ipix_source_input (int): pixel index of source location

    Returns:
        StokesIQUV: convolved and normalized source map
    """

    m_input_beam_conv = jnp.zeros((4,12*Nside**2))
    m_input_beam_conv = m_input_beam_conv.at[0,ipix_source_input].set(1.0)

    # Convolve
    for i in range(4):
        m_input_beam_conv = m_input_beam_conv.at[i].set(
            beam_conv(m_input_beam_conv[i], beam_fwhm)
        )

    # Normalize
    global_max = jnp.max(jnp.abs(m_input_beam_conv))
    m_input_beam_conv = m_input_beam_conv / global_max

    # Write and return
    hp.write_map('beam_drone.fits', m_input_beam_conv, overwrite=True)
    
    return Stokes.from_stokes(*m_input_beam_conv)

#%% Telescope pointing

def build_mapping_quaternions(t_v, scan_type=scan_type, teles_amp_scan=amp_scan, teles_freq_scan=freq_scan):
    """
    Build boresight and detector quaternions for telescope pointing.
    Parameters:
        t_v (jnp.array): time vector
        scan_type (int): type of scan (0: fixed, 1: scanning)
        ampscan (float): scanning amplitude (radians)
        freqscan (float): scanning frequency (Hz)
    
    Returns:
        qbore (jnp.array): boresight quaternions
        qdet (jnp.array): detector quaternions
    """
    # Create (az, el) traj, considering fixed elevation in all cases for now
    els = jnp.ones_like(t_v) * azel_source_input[1] # Same elevation as the source

    azs = None 
    if scan_type == 0:
        azs = jnp.ones_like(t_v) * azel_source_input[0]  # Fixed azimuth to source
    elif scan_type == 1:
        azs = jnp.pi / 2 + teles_amp_scan * jnp.cos(2 * jnp.pi * teles_freq_scan * t_v)

    # Convert to quaternions
    psi = jnp.zeros_like(t_v) # no rotation around line of sight
    qbore = jnp.array([from_lonlat_angles(a, e, p) for a, e, p in zip(azs, els, psi)])
    
    # Detectors aligned with boresight (identity quaternion)
    qdet = jnp.array([[1.0, 0.0, 0.0, 0.0] for _ in range(n_detectors)])

    return qbore, qdet

def telescope_pointing(landscape, t_v, scan_type=scan_type, teles_amp_scan=amp_scan):
    """
    Create PointingOperator for telescope.
    
    Parameters:
        landscape (HealpixLandscape): sky landscape
        t_v (jnp.array): time vector
        scan_type (int): type of scan (0: fixed, 1: scanning)
        teles_ampscan (float): scanning amplitude (radians)

    Returns:
        PointingOperator: telescope pointing operator
    """
    # Build boresight and detector quaternions
    qbore, qdet = build_mapping_quaternions(t_v, scan_type=scan_type, teles_amp_scan=teles_amp_scan)
    
    return PointingOperator(
        landscape=landscape,
        qbore=qbore,
        qdet=qdet,
        _in_structure=landscape.structure,
        _out_structure=Stokes.class_for('IQUV').structure_for((n_detectors, len(t_v))),
        chunk_size=16,
    )

#%% Source mapping

def chopper(t_v, f_chop=f_chop, phi_chop=phi_chop):
    """
    Generate a chopping signal using a square wave.

    Parameters:
        t_v (jnp.array): time(s)
        f_chop (float): chopping frequency (Hz)
        phi_chop (float): chopping phase (radians)

    Returns:
        np.array: chopping signal oscillating between 0 and 1
    """
    return (signal.square(2 * jnp.pi * f_chop * t_v + phi_chop) + 1) / 2

def I_source(t_v, I0=I0, source_type=source_type, f_chop=f_chop, phi_chop=phi_chop):
    """
    Generate sky source intensity with different modulation components.

    Parameters:
        I0 (float): base intensity
        t_v (jnp.array): time(s)
        f_chop (float): chopping frequency (Hz)
        phi_chop (float): chopping phase (radians)
        source_type (str): string controlling signal composition, with flags:
            'i' : include intensity I0
            'c' : include chopping modulation
            'v1': include low-frequency variation (variations_omega1)
            'v2': include high-frequency variation (variations_omega2)
            'd' : add drone amplitude offset (thermal emission)

    Returns:
        jnp.array: modulated source intensity 
    """
    # Initialize signal: 0 if 'd' only, else 1
    signal_ = jnp.where(source_type == 'd', 0.0, 1.0)

    if 'i' in source_type:
        signal_ *= I0
    if 'c' in source_type:
        signal_ *= chopper(t_v, f_chop, phi_chop)
    if 'v1' in source_type:
        signal_ *= (1.0 + jnp.cos(2.0 * jnp.pi * variations_omega1 * t_v) * delta_variations1)
    if 'v2' in source_type:
        signal_ *= (1.0 + jnp.cos(2.0 * jnp.pi * variations_omega2 * t_v) * delta_variations2)

    # Add drone amplitude offset
    if 'd' in source_type:
        signal_ += drone_amplitude

    return signal_

def drone_intensity(t_v, source_type=source_type, I0=I0, f_chop=f_chop, phi_chop=phi_chop):
    """
    Build the operator representing the drone source intensity mapping.

    Parameters:
        t_v (jnp array): time vector
        source_type (str): string controlling signal composition
        I0 (float): base intensity
        f_chop (float): chopping frequency (Hz)
        phi_chop (float): chopping phase (radians)

    Returns:
        HomothetyOperator: operator applying source intensity factor
    """
    intensity_factor = I_source(t_v, I0=I0, source_type=source_type,\
                                f_chop=f_chop, phi_chop=phi_chop)
    return HomothetyOperator(value=intensity_factor,\
                             _in_structure=Stokes.class_for('IQUV').structure_for((n_detectors, len(t_v))))

def drone_operator(t_v, alpha_drone=alpha_drone, source_type=source_type, I0=I0, f_chop=f_chop, phi_chop=phi_chop):
    """
    Calculate source signal (I, Q, U) including beam intensity and polarization effects.

    Parameters: 
        t_v (jnp array): time vector
        alpha_drone (float): drone polarization angle (radians)
        source_type (str): string controlling signal composition
        amplitude (float): base intensity
        f_chop (float): chopping frequency (Hz)
        phi_chop (float): chopping phase (radians)

    Returns:
        Operator: operator representing the drone source mapping
    """
    polarized_source = source_type.replace('d', '')
    intensity_op = drone_intensity(t_v, source_type=polarized_source, I0=I0,\
                                  f_chop=f_chop, phi_chop=phi_chop)
    
    polarization_op = ActualLinearPolarizerOperator.create(shape = (n_detectors,len(t_v)),\
                                                           stokes = 'IQUV',\
                                                           angles = jnp.array([alpha_drone]))

    drone_op = polarization_op @ intensity_op
    if 'd' in source_type:
        thermal_op = drone_intensity(t_v, source_type='d', I0=I0,\
                                     f_chop=f_chop, phi_chop=phi_chop)
        drone_op = thermal_op + drone_op
    
    return drone_op

#%% VPM (Variable Phase Modulator) and HWP (Half-Wave Plate)

def z_computation(t_v):
    """
    Compute VPM modulation displacement z as function of time.

    Parameters:
        t_v (jnp float array): time (s)

    Returns:
        jnp float array: displacement (m)
    """
    return z_0 * jnp.cos(2 * jnp.pi * f_vpm * t_v)

def delta_computation(t_v):
    """
    Compute VPM modulation phase delta as function of time.

    Parameters:
        t_v (jnp float array): time (s)

    Returns:
        jnp float array: phase shift (radians)
    """
    nu = 40e9 # frequency 40Ghz
    c = 3e8
    incidence_theta = 0.0
    z = z_computation(t_v)
    return 4 * jnp.pi * nu / c * z * jnp.cos(incidence_theta)

def wave_plate(t_v, hwp=hwp, vpm=vpm):
    """
    Create wave plate operator (HWP or VPM).

    Parameters:
        t_v (jnp float array): time vector
        hwp (bool): whether to include HWP
        vpm (bool): whether to include VPM

    Returns:
        WavePlateOperator: wave plate operator
    """
    if hwp and vpm:
        print('Choose either HWP or VPM modulation, not both.')
        sys.exit()
    
    if hwp:
        phi = jnp.ones_like(t_v) * jnp.pi # HWP -> phi = pi
        angles = 2*jnp.pi*f_hwp*t_v # Rotation of the plate
    elif vpm:
        phi = delta_computation(t_v)
        angles = None # No rotation
    
    return WPOperator.create(shape=(n_detectors, len(t_v)),
                              phi=phi, angles=angles,
                              stokes='IQUV')

#%% Operator combination

def telescope_op(t_v, alpha_det=true_det_angle, hwp=hwp, vpm=vpm):
    """
    Build telescope detection operator including HWP or VPM.

    Parameters:
        t_v (jnp array): time vector
        alpha_det (float): detector angle (radians)
        hwp (bool): whether to include HWP
        vpm (bool): whether to include VPM
    Returns:
        Operator: telescope detection operator
    """

    wp_op = wave_plate(t_v, hwp=hwp, vpm=vpm)

    pol = LinearPolarizerOperator.create(shape=(n_detectors, len(t_v)), 
                                         angles=alpha_det, stokes='IQUV')

    return pol @ wp_op

def pipeline(t_v, landscape, alpha_det=true_det_angle, scan_type=scan_type,\
          amp_scan=amp_scan, hwp=True, vpm=False,\
          alpha_drone=alpha_drone, source_type=source_type, I0=I0,\
          f_chop=f_chop, phi_chop=phi_chop):
    """
    Build full pipeline operator from sky to time-ordered data (TOD).
    
    Parameters:
        landscape (HealpixLandscape): sky landscape
        t_v (jnp array): time vector
        alpha_det (float): detector angle (radians)
        scan_type (int): type of scan (0: fixed, 1: scanning)
        amp_scan (float): scanning amplitude (radians)
        hwp (bool): whether to include HWP
        vpm (bool): whether to include VPM
        alpha_drone (float): drone polarization angle (radians)
        source_type (str): string controlling signal composition
        I0 (float): base intensity
        f_chop (float): chopping frequency (Hz)
        phi_chop (float): chopping phase (radians)

    Returns:
        Operator: full pipeline operator
    """
    
    # Telescope pointing
    pointing = telescope_pointing(landscape, t_v, scan_type, amp_scan)

    # Drone modulation (= source intensity and polarization)
    M_drone = drone_operator(t_v,alpha_drone,source_type,I0,f_chop,phi_chop)

    # Detector
    telescope_detection = telescope_op(t_v, alpha_det, hwp, vpm)
    
    return (telescope_detection @ M_drone @ pointing).reduce()

#%% Data output

def add_noise(detector_res, noise_level=noise_level, key=0):
    """
    Add Gaussian noise to detector results.

    Parameters:
        detector_res (jnp.array): detector time-ordered data
        noise_level (float): standard deviation of Gaussian noise

    Returns:
        jnp.array: noisy detector data
    """
    key = jax.random.PRNGKey(key)
    noise_values = jax.random.normal(key, shape=detector_res.shape) * noise_level # X ~ N(0, noise_level)
    return detector_res + noise_values

def detectors_output(t_v=t_v, alpha_det=true_det_angle, scan_type=scan_type,\
          amp_scan=amp_scan, hwp=hwp, vpm=vpm, noise_level=noise_level,\
          alpha_drone=alpha_drone, source_type=source_type, I0=I0,\
          f_chop=f_chop, phi_chop=phi_chop, beam_fwhm=beam_fwhm,\
          Nside=Nside, ipix_source_input=ipix_source_input):
    """
    Simulate detector time-ordered data (TOD) using the full pipeline.
    Parameters:
        t_v (jnp array): time vector
        alpha_det (float): detector angle (radians)
        scan_type (int): type of scan (0: fixed, 1: scanning)
        amp_scan (float): scanning amplitude (radians)
        hwp (bool): whether to include HWP
        vpm (bool): whether to include VPM
        noise_level (float): standard deviation of Gaussian noise
        alpha_drone (float): drone polarization angle (radians)
        source_type (str): string controlling signal composition
        I0 (float): base intensity
        f_chop (float): chopping frequency (Hz)
        phi_chop (float): chopping phase (radians)
        beam_fwhm (float): beam full width half max in arcminutes
        Nside (int): Healpix Nside parameter
        ipix_source_input (int): pixel index of source location
    
    Returns:
        pipeline_op (Operator): full pipeline operator
        fixed_conv_source (StokesIQUV): convolved source map
        detectors_res (jnp array): simulated detector TOD
    """
    landscape = HealpixLandscape(nside=Nside, stokes='IQUV')
    
    pipeline_op = pipeline(t_v, landscape, alpha_det, scan_type,\
                            amp_scan, hwp, vpm,\
                            alpha_drone, source_type, I0,\
                            f_chop, phi_chop)
    
    fixed_conv_source = conv_source(beam_fwhm, Nside, ipix_source_input)
    detectors_res = pipeline_op.mv(fixed_conv_source)

    # Add noise if specified
    if 'n' in source_type:
        detectors_res = add_noise(detectors_res, noise_level)
    
    jnp.save('detectors_res.npy', detectors_res)

    return pipeline_op, fixed_conv_source, detectors_res