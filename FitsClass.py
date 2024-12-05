from astropy.io import fits
import numpy as np

class FITSFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.primary_data = None
        self.raw_data = None
        self.data = None
        self.dataheader = None
        self.header = None

    def load_data(self):
        """Load data and header from the FITS file."""
        try:
            with fits.open(self.filepath) as hdulist:
                self.primary_data = hdulist[0].data
                self.raw_data = hdulist[1]
                self.data = hdulist[1].data
                self.dataheader = hdulist[1].header
                self.header = hdulist[0].header
            print(f"Data loaded from {self.filepath}")
        except Exception as e:
            print(f"Error loading {self.filepath}: {e}")

    def print_file_info(self):
        """Prints all available information in the FITS file."""
        try:
            with fits.open(self.filepath) as hdulist:
                print(f"Opened FITS file: {self.filepath}")
                print(f"Number of HDUs: {len(hdulist)}")
                print("=" * 60)
                
                for idx, hdu in enumerate(hdulist):
                    print(f"HDU {idx}: {type(hdu).__name__}")
                    print("-" * 60)
                    
                    # Print header keywords and values
                    print("Header:")
                    for key, value in hdu.header.items():
                        print(f"{key} = {value}")
                    print("-" * 60)
                    
                    # Check the type of HDU and print relevant info
                    if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                        # Image data
                        data_shape = hdu.data.shape if hdu.data is not None else None
                        data_type = hdu.data.dtype if hdu.data is not None else None
                        print(f"Image Data Shape: {data_shape}")
                        print(f"Image Data Type: {data_type}")
                    elif isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                        # Table data
                        print("Table Columns:")
                        for col in hdu.columns:
                            print(f"Name: {col.name}, Format: {col.format}, Unit: {col.unit}")
                        print("Column Names:")
                        print(hdu.columns.names)
                    else:
                        print("HDU data type not recognized or not supported.")
                    
                    print("=" * 60)
        except Exception as e:
            print(f"Error opening FITS file: {e}")
        

    def process_data(self, func=None):
        """Process the data using a provided function."""
        if self.data is None:
            self.load_data()
            if self.data is None:
                return
        if func:
            self.data = func(self.data)
            print(f"Custom processing applied to {self.filepath}")
        else:
            # Default processing (e.g., normalization)
            self.data = self.data / np.max(self.data)
            print(f"Data normalized for {self.filepath}")

    # Additional methods as needed (e.g., save processed data)
