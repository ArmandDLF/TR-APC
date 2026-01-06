#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author : armand, éléonore

Angle estimation from detector TOD,
furax integration of Anouk's code "Testing Spectral Likelihood for Drone Data Analysis"
"""

#%% Imports
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from furax.core import HWPDemodOperator, ChopperDemodOperator

import tod_simulation as ts

#%% Global parameters

arcmin2rad = ts.arcmin2rad
Nside = ts.Nside

I0 = ts.I0
noise_level = ts.noise_level

scan_type = ts.scan_type
azel_source_input = ts.azel_source_input

# Drone amplitude variations parameters
drone_amplitude = ts.drone_amplitude
delta_variations1 = ts.delta_variations1
delta_variations2 = ts.delta_variations2
variations_omega1 = ts.variations_omega1
variations_omega2 = ts.variations_omega2

ipix_source_input = ts.ipix_source_input

# Drone emission
beam_fwhm = ts.beam_fwhm
alpha_drone = ts.alpha_drone
true_det_angle = ts.true_det_angle
source_type = ts.source_type

# HWP and chopper parameters
f_chop = ts.f_chop
phi_chop = ts.phi_chop
z_0 = ts.z_0
f_vpm = ts.f_vpm
f_hwp = ts.f_hwp
demod_mode = ts.demod_mode
hwp, vpm = ts.hwp, ts.vpm

# Telescope parameters 
amp_scan = ts.amp_scan
freq_scan = ts.freq_scan
n_detectors = ts.n_detectors

# Time
k = ts.k
sample_rate = ts.sample_rate
t_v = ts.t_v

#%% Demodulation

def demod_tod_double(tod, t_v=t_v, source_type=source_type, demod_mode=demod_mode, \
                     f_chop=f_chop, phi_chop=phi_chop, det_angle=true_det_angle):
    """
    Demodulate time-ordered data (TOD) including chopper modulation and HWP angle effects.
    
    Parameters:
    -----------
    tod : array_like (1D)
        signal to demodulate in time domain -> should be TOD and not its FFT
    t_v : array_like
        Time vector for the TOD.
    source_type : str, optional
        Type of source simulated or observed (e.g., 'i', 'c' indicating presence of chopper).
    demod_mode : int, optional
        Demodulation mode; typically corresponds to harmonics of HWP rotation frequency.
    freq_chopper : float, optional
        Frequency of the chopper modulation (Hz).
    chopper_phi : float, optional
        Phase offset of the chopper modulation (radians).
    det_angle : float
        True detector angle (radians) used in demodulation phase correction.
    
    Returns:
    --------
    dsT : ndarray
        Demodulated and filtered intensity (I) signal.
    demodQ : ndarray
        Demodulated and filtered Q Stokes parameter.
    demodU : ndarray
        Demodulated and filtered U Stokes parameter.
    
    Notes:
    ------
    This function performs double demodulation:
    - Demodulation with respect to chopper frequency using a square wave approximation. (if 'c' in source_type)
    - Demodulation with respect to half-wave plate (HWP) rotation frequency.
    Finally applies a low-pass filter in frequency space and returns the I, Q, U components.
    """
    
    in_struct = jax.ShapeDtypeStruct(tod.shape, tod.dtype)

    # --- temporal operators ---

    phase_func = lambda t: demod_mode * 2 * jnp.pi * f_hwp * t - 2.0 * det_angle
    op_hwp = HWPDemodOperator(
        phase_func = phase_func,
        in_structure = in_struct,
        sample_rate = sample_rate,
    )
    
    # --- full operator ---

    phasor_chopper = ts.chopper(t_v, f_chop, phi_chop) if 'c' in source_type else 1.0

    demodQU = op_hwp.mv(tod * phasor_chopper)
    demodI = tod * phasor_chopper

    #low-pass filtering at 2 Hz
    freq = jnp.fft.fftfreq(len(t_v), t_v[1] - t_v[0])
    filtre = jnp.abs(freq) < 2
    filterfft1 = jnp.array(jnp.fft.fft(demodQU))
    filterfft2 = jnp.array(jnp.fft.fft(demodI))
    filterfft1 = filterfft1.at[~filtre].set(0.0)
    filterfft2 = filterfft2.at[~filtre].set(0.0)

    # Transform filtered signals back to time domain
    redemod = jnp.fft.ifft(filterfft1)
    redemod_I = jnp.fft.ifft(filterfft2)
    
    dsT = 2 * jnp.real(redemod_I)      #x2 car renormalisation dûe au chopper 0/1 et pas -1/1
    demodQ = 4 * jnp.real(redemod)     #x4 pour renormalisation chopper (x2) et moyenne de cos^2 = 1/2 (x2)
    demodU = 4 * jnp.imag(redemod)     #idem 

    return dsT, demodQ, demodU


def estimangle(tod, t_v=t_v, det_angle=true_det_angle, source_type=source_type,\
                demod_mode=demod_mode, f_chop=f_chop, phi_chop=phi_chop, beam_fwhm = beam_fwhm):
    """
    Estimate the polarization angle from simulated or observed TOD data.
    
    Parameters:
    -----------
    tod : array_like (1D)
        Time-ordered data (TOD) from the detector.
    t_v : array_like
        Time vector for the TOD.
    alpha_drone : float
        Theoretical polarization angle of the source (in radians).
    det_angle : float
        Detector angle correction (in radians).
    source_type : str, optional
        Type of source simulated or observed (e.g., 'i', 'c' indicating presence of chopper).
    scan_type : str
        Type of telescope scan used.
    demod_mode : int, optional
        Demodulation mode; typically corresponds to harmonics of HWP rotation frequency.
    f_chop : float, optional
        Frequency of the chopper modulation (Hz).
    phi_chop : float, optional
        Phase offset of the chopper modulation (radians).

    Returns:
    --------
    alpha_reconstruit : float or tuple
        Estimated polarization angle in degrees    
    """
    _, Q, U = demod_tod_double(tod, t_v, source_type, demod_mode, f_chop, phi_chop, det_angle)    
    
    if scan_type == 1:
        # 1. Calculer la position du télescope pendant le scan
        azs = jnp.pi/2 + amp_scan * jnp.cos(2*jnp.pi*freq_scan*t_v)
        
        # 2. Calculer l'offset par rapport à la source
        offset_rad = azs - azel_source_input[0]  # différence en azimut
        
        # 3. Calculer la réponse du beam (gaussien)
        beam_fwhm_rad = beam_fwhm * arcmin2rad
        # Pour un faisceau gaussien : I(θ) = I₀ × exp(-4ln(2) × (θ/FWHM)²)
        beam_response = jnp.exp(-4 * jnp.log(2) * (offset_rad / beam_fwhm_rad)**2)
        
        # 4. Corriger Q et U en divisant par la réponse du beam
        # (on compense la perte de signal quand on s'éloigne du centre)
        Q_corrected = Q / beam_response
        U_corrected = U / beam_response
        
        # 5. Calculer l'angle avec les valeurs corrigées
        alpha_reconstruit = 0.5 * jnp.arctan(jnp.mean(U_corrected) / jnp.mean(Q_corrected))
        
    else:
        # Pas de correction nécessaire pour scan_type = 0
        Q_corrected = Q
        U_corrected = U
        alpha_reconstruit = 0.5 * jnp.arctan(jnp.mean(U) / jnp.mean(Q))

    import matplotlib.pyplot as plt
    # Plot Q,U variations and angle reconstruction over time
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_v, Q, label='Q(t)')
    plt.plot(t_v, U, label='U(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Stokes parameters')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(t_v, jnp.degrees(0.5 * jnp.arctan2(U, Q)), label='Reconstructed angle (deg)')
    plt.axhline(y=jnp.degrees(alpha_reconstruit), color='r', linestyle='--', label='Reconstructed mean angle (deg)')
    plt.xlabel('Time (s)')
    plt.ylabel('Polarization angle (deg)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("QUalpha_TOD.png", dpi=150)
    plt.close()

    return jnp.degrees(alpha_reconstruit)