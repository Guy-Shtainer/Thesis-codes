from astropy.io import fits

# Path to the uploaded FITS file
file_path = '../RawData/lco_data-20250104-56/cptnrs03-fa13-20250103-0033-e92-2d.fits'

# Open the FITS file
with fits.open(file_path) as hdul:
    # Print information about the FITS file
    hdul.info()
    
    # Access the primary HDU (Header Data Unit)
    primary_hdu = hdul[0]
    print("\nPrimary HDU Header:")
    print(repr(primary_hdu.header))
    
    # Access data if present
    if primary_hdu.data is not None:
        print("\nPrimary HDU Data:")
        print(primary_hdu.data)
    
    # Loop through extensions if present
    for i, hdu in enumerate(hdul[1:], start=1):
        print(f"\nExtension {i}:")
        print(repr(hdu.header))
        if hdu.data is not None:
            print("Data present in this extension.")
