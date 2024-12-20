import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy
import glob
from astropy.table import Table
import os
import matplotlib.colors as mcolors


# Point Spread Function Size in pixels (to be played with)
PSF_Size_Left = 20
PSF_Size_Right = 8

# wave_range = 'nir'
wave_range = 'vis'
# wave_range = 'uvb'

obs = '59153'
star = 'RMC_140'

plot = True
write = False

# Angstroem or nm (10 / 1):
WaveUnitFactor = 10.
#WaveUnitFactor = 1.

# ------------------- RMC140 -----------

# fill in the path to your downloaded fits file
# path per downloaded arm (UVB, VIS, NIR)

# 59153
PATH_TO_OBS = '/Users/tomer/Desktop/Work/Supervision/Amsterdam/2023/Freek/WC-LMC-thesis-Freek-files/2DIMAGE/HD32228/59150/ADP.2020-11-13T13_19_45.607.fits'
PATH_TO_OBS = f'../RawData/archive/ADP.2020-11-13T12:33:07.692.fits'

# -----------------------

# image_data = fits.getdata(PATH_TO_OBS, ext=0)

def Plot2DImage_for_cleaning(image_data, wavelengths, band, bottom_spacial, top_spacial, abs_include_start, abs_include_end, title='', ValMin=None, ValMax=None, norm=False, ax=None):
    """
    Plots a 2D image of flux vs. wavelength and spatial coordinate, with internal cropping.
    Negative indices for bottom_spacial and top_spacial are interpreted from the end.
    After cropping, the vertical axis starts at 0 at the new bottom line and goes up
    to (top_spacial - bottom_spacial - 1).

    Parameters:
        image_data: 2D numpy array (full image)
        wavelengths: 1D array of wavelengths corresponding to columns of image_data
        band: str, the observation band (e.g. 'NIR', 'UVB')
        bottom_spacial: int, bottom spatial index (can be negative)
        top_spacial: int, top spatial index (can be negative)
        title: str, title for the plot
        ValMin, ValMax: float, optional intensity limits
        norm: bool, if True, use log normalization
        ax: matplotlib axis object, if None, create a new figure and axis

    Returns:
        tuple: (wavelengths or None, summed_flux_over_spatial)
    """

    height = image_data.shape[0]

    # Crop the image internally based on the provided spatial indices
    cropped_image = image_data[bottom_spacial:top_spacial, :]
    
    # Convert negative indices to positive
    if bottom_spacial < 0:
        bottom_spacial = height + bottom_spacial
    if top_spacial < 0:
        top_spacial = height + top_spacial

    # Ensure top_spacial > bottom_spacial
    if top_spacial <= bottom_spacial:
        top_spacial = bottom_spacial + 1


    # print(f'Spatial axis has {cropped_image.shape[0]} items')

    # Filter out non-positive values if needed for LogNorm
    positive_data = cropped_image[cropped_image > 0]

    # Determine extent for imshow
    if wavelengths is not None:
        x_min, x_max = wavelengths[0], wavelengths[-1]
    else:
        x_min, x_max = 0, cropped_image.shape[1] - 1

    # After cropping, define the y-axis from 0 to height_of_cropped_region-1
    # so that the bottom line is 0 and the top line is top_spacial - bottom_spacial.
    cropped_height = cropped_image.shape[0]
    y_min, y_max = bottom_spacial, top_spacial

    # If ax is not provided, create a new figure and axis
    created_figure = False
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        created_figure = True

    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Pixel coordinate")
    ax.set_ylabel('Spatial coordinate')

    # Determine normalization or linear scale
    if norm:
        if positive_data.size > 0:
            if ValMin is None:
                ValMin = np.amin(positive_data)
            if ValMax is None:
                ValMax = np.amax(positive_data)
        else:
            # If no positive data, fall back to entire cropped_image
            if ValMin is None:
                ValMin = np.amin(cropped_image)
            if ValMax is None:
                ValMax = np.amax(cropped_image)
        norm_scale = mcolors.LogNorm(vmin=ValMin, vmax=ValMax)
        im = ax.imshow(cropped_image, norm=norm_scale, aspect='auto', extent=(x_min, x_max, y_min, y_max), origin='lower')
    else:
        ValMin = np.amin(cropped_image) if ValMin is None else ValMin
        ValMax = np.amax(cropped_image) if ValMax is None else ValMax
        # ax.axhline(abs_include_start - bottom_spacial, color='red', linestyle='--')
        # ax.axhline(abs_include_end - bottom_spacial, color='red', linestyle='--')
        ax.axhline(abs_include_start, color='red', linestyle='--')
        ax.axhline(abs_include_end, color='red', linestyle='--')
        im = ax.imshow(cropped_image, vmin=ValMin, vmax=ValMax, aspect='auto', extent=(x_min, x_max, y_min, y_max), origin='lower')

    if created_figure:
        plt.colorbar(im, ax=ax, orientation='vertical', label='Flux (counts)')
        plt.tight_layout()
        plt.show()

    # Return wavelengths and summed flux
    if wavelengths is not None:
        return wavelengths, np.sum(cropped_image, axis=0)
    else:
        return None, np.sum(cropped_image, axis=0)

def Plot2DImage_old(image_data, wavelengths, band, title='', ValMin=None, ValMax=None, norm=False, see_all=False, ax=None):
    if not see_all:
        if band == 'NIR':
            image_data = image_data[-52:-24, :]
        else:
            image_data = image_data[-68:-30, :]

    print(f'spacial axis has {len(image_data[:, 1])} items')
    # Filter out non-positive values for LogNorm
    positive_data = image_data[image_data > 0]

    # Determine extent for imshow
    if wavelengths is not None:
        x_min, x_max = wavelengths[0], wavelengths[-1]
    else:
        # If no wavelengths given, just use pixel coordinates
        x_min, x_max = 0, image_data.shape[1] - 1
    y_min, y_max = 0, image_data.shape[0] - 1

    # If ax is not provided, create a new figure and axis
    created_figure = False
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        created_figure = True

    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Pixel coordinate")
    ax.set_ylabel('Spatial coordinate')

    # Determine normalization or linear scale
    if norm:
        ValMin = np.amin(positive_data) if ValMin is None else ValMin
        ValMax = np.amax(positive_data) if ValMax is None else ValMax
        norm = mcolors.LogNorm(vmin=ValMin, vmax=ValMax)
        im = ax.imshow(image_data, norm=norm, aspect='auto', extent=(x_min, x_max, y_min, y_max), origin='lower')
    else:
        ValMin = np.amin(image_data) if ValMin is None else ValMin
        ValMax = np.amax(image_data) if ValMax is None else ValMax
        im = ax.imshow(image_data, vmin=ValMin, vmax=ValMax, aspect='auto', extent=(x_min, x_max, y_min, y_max), origin='lower')

    # Add colorbar only if we created the figure (to avoid duplication if caller handles it)
    if created_figure:
        plt.colorbar(im, ax=ax, orientation='vertical', label='Flux (counts)')
        plt.tight_layout()
        plt.show()

    # If needed, reconstruct the wavelength solution (though this logic may be out of place if PATH_TO_OBS isn't defined here)
    # Without the path or header info, we can't reliably reconstruct the wavelength grid.
    # Returning the sums if needed:
    if wavelengths is not None:
        return wavelengths, np.sum(image_data, axis=0)
    else:
        return None, np.sum(image_data, axis=0)


def Plot2DImage_older(image_data,wavelengths,band, title='', ValMin=None, ValMax=None,norm = False,see_all = False):
    if not see_all:
        if band == 'NIR':
            image_data = image_data[-52:-24,:]
        else:
            image_data = image_data[-68:-30,:]

    print(f'spacial axis has {len(image_data[:,1])} items')
    # Filter out non-positive values for LogNorm
    positive_data = image_data[image_data > 0]

    # Determine extent for imshow
    x_min, x_max = wavelengths[0], wavelengths[-1]
    y_min, y_max = 0, image_data.shape[0] - 1

    # Set default values for ValMin and ValMax if not provided, using only positive values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    # ax.set_xlabel('Wavelength coordinate')
    ax.set_xlabel("Wavelength nm")
    ax.set_ylabel('Spatial coordinate')

    # Create LogNorm with the validated ValMin and ValMax
    if norm:
        ValMin = np.amin(positive_data) if ValMin is None else ValMin
        ValMax = np.amax(positive_data) if ValMax is None else ValMax
        norm = mcolors.LogNorm(vmin=ValMin, vmax=ValMax)
        plt.imshow(image_data,norm = norm, aspect='auto', extent=(x_min, x_max, y_min, y_max))
    else:
        ValMin = np.amin(image_data) if ValMin is None else ValMin
        ValMax = np.amax(image_data) if ValMax is None else ValMax
        plt.imshow(image_data, vmin=ValMin, vmax=ValMax, aspect='auto', extent=(x_min, x_max, y_min, y_max))


    # Apply the LogNorm to imshow
    plt.colorbar(orientation='vertical', label='Flux (counts)')
    fig.tight_layout()
    plt.show()

# Construct wavelength grid
    crval = get_key_from_header(PATH_TO_OBS, 'CRVAL1')
    pix = get_key_from_header(PATH_TO_OBS, 'CRPIX1')
    cdelt = get_key_from_header(PATH_TO_OBS, 'CRVAL1')
    Nwave = len(image_data[0,:])
    wave = crval - cdelt*(pix - 1) + np.arange(Nwave)*cdelt 
# Convert to Angstroem potentially
    return wave*WaveUnitFactor, np.sum(image_data, axis=0)

# Retrieves header key (e.g.: get_key_from_header(PATH_TO_OBS, 'MJD-OBS') )
def get_key_from_header(infile, key, ext=0):
    header = fits.getheader(infile, ext)
    info = header[key]
    return info

# ------------------------ PLOTS --------------------------------

# Plot image and extract wavelength grid and total flux
# WaveGrid, TotFlux = Plot2DImage(image_data)


# # Compute "collapsed flux vs. spatial coordinate", e.g. to spot the Wolf-Rayet coordinates:
# FluxSumArray = np.sum(image_data, axis=1)

# # Can be manual too!
# StarPosition1 = np.argmax(FluxSumArray)


# plt.plot(FluxSumArray, label='collapsed flux')
# plt.axvline(x=StarPosition1-PSF_Size_Left, color='red', linestyle='--', linewidth=1)
# plt.axvline(x=StarPosition1+PSF_Size_Right, color='red', linestyle='--', linewidth=1)


# plt.legend()
# plt.show()


# # Compute flux in selected range
# Star1_Flux = np.sum(image_data[StarPosition1 - PSF_Size_Left : StarPosition1 + PSF_Size_Right], axis=0)

# plt.plot(WaveGrid, TotFlux, label='total flux')
# plt.plot(WaveGrid, Star1_Flux, label='extracted flux star1')

# plt.legend()
# plt.show()

# Repeat for additional "peaks" you wish to extract...
# Save file ideally in fits format (e.g., by replacing the y-axis of the reduced data with the extracted data, making a copy of course....). 


