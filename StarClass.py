import os
import shutil
import glob
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import utils as ut
from IPython.display import display
from astropy.io import fits
import specs
from FitsClass import FITSFile as myfits 
from CCF import CCFclass
import plot as p
from threading import Thread
import multiprocess
from itertools import product
from functools import partial

#SIMBAD
import requests
from bs4 import BeautifulSoup
import re

class Star:
    def __init__(self, star_name, data_dir, backup_dir):
        """
        Initialize the Star with an identifier and a dictionary of file paths.
        file_paths: A nested dictionary organized by epochs, sub-exposures, and bands.
        """
        self.star_name = star_name
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        self.observation_dict = specs.obs_file_names[star_name]
        self.observations = {}  # Cache of FITSFile instances
        self.normalized_spectra = {}  # Structure: {epoch: {sub_exp: {band: {'wavelength':..., 'flux':..., 'fitted_continuum':...}}}}
        self.BAT_ID = self.get_bat_id()

        self.normalized_wavelength = []
        self.normalized_flux = []
        self.included_wave = []
        self.included_flux = []
        self.sensitivities = []

########################################                 File Handleing                      ########################################
    
    def get_file_path(self, epoch_number, band=None):
        """
        Retrieves the full path for a specific observation.

        Parameters:
        - epoch_number (int): The epoch (e.g., '1').
        - band (str, optional): The band (e.g., 'UVB', 'VIS', 'NIR').
                                If not specified, assumes the combined FITS file.

        Returns:
        - full_file_path (str): The full path of the FITS file.
        - None: If the requested file is not found.
        """
        # Check if the specified epoch exists for the star
        if f'epoch{epoch_number}' not in self.observation_dict:
            print(f"Epoch '{epoch_number}' not found for star '{self.star_name}'")
            return None

        epoch_data = self.observation_dict[f'epoch{epoch_number}']
        # If a band is specified, return the file for that band
        if band:
            if band in epoch_data:
                filename = epoch_data[band]
                return os.path.join(self.data_dir, self.star_name, f'epoch{epoch_number}', band, filename)
            else:
                print(f"Band '{band}' not found in epoch '{epoch_number}")
                return None
        else:
            # Assume we're looking for a combined file if no band is specified
            if 'combined' in epoch_data:
                filename = epoch_data['combined']
                return os.path.join(self.data_dir, self.star_name, f'epoch{epoch_number}', 'combined', filename)
            else:
                print(f"Combined FITS file not found for epoch '{epoch_number}")
                return None

########################################                                     ########################################
    
    def delete_files(self, epoch_numbers=None, bands=None, backup_flag=True,
                     property_to_delete='', delete_all=False, delete_all_in_folder=False):
        """
        Deletes files or folders for this star for the specified epochs and bands.

        Parameters:
            epoch_numbers (int or list): Epoch number(s). Default is None.
            bands (str or list): Band name(s). Default is None.
            backup_flag (bool): Whether to backup files before deletion.
            property_to_delete (str): Name of the property/file or folder to delete.
            delete_all (bool): If True, allows deletion when parameters are None.
            delete_all_in_folder (bool): If True, deletes all files in the folder without prompting.
        """
        all_bands = False
        # Handle default parameters and delete_all flag
        if epoch_numbers is None or bands is None:
            if delete_all:
                if epoch_numbers is None:
                    epoch_numbers = self.get_all_epoch_numbers()
                if bands is None:
                    all_bands = True
            else:
                missing = []
                if epoch_numbers is None:
                    missing.append('epoch_numbers')
                if bands is None:
                    missing.append('bands')
                missing_str = ', '.join(missing)
                raise ValueError(f"Error: {missing_str} parameter(s) are None for star '{self.star_name}'. To delete all options, set delete_all=True.")

        # Ensure epoch_numbers and bands are lists
        if not isinstance(epoch_numbers, list):
            epoch_numbers = [epoch_numbers]
        if not isinstance(bands, list):
            bands = [bands]

        # For each combination
        for epoch_num in epoch_numbers:
            epoch_key = f'epoch{epoch_num}'
            if all_bands:
                bands = self.observation_dict[epoch_key]
            for band in bands:
                # Construct the base path to the 'output' directory
                output_dir = os.path.join(self.data_dir, self.star_name, epoch_key, band, 'output')
                
                # If property_to_delete is empty, prompt for input
                if not property_to_delete:
                    print(f"Error: 'property_to_delete' is empty. Please specify a file or folder name.")
                    continue

                # Construct the full path
                property_path = os.path.join(output_dir, property_to_delete + '.npz')
                print(f'property_path is {property_path}')

                # Check if exact match exists
                if os.path.isfile(property_path):
                    self._delete_file(property_path, backup_flag)
                elif os.path.isdir(property_path):
                    self._handle_folder_deletion(property_path, backup_flag, delete_all_in_folder)
                else:
                    # Use glob to find files starting with property_to_delete
                    pattern = os.path.join(output_dir, property_to_delete + '*')
                    matching_files = glob.glob(pattern)

                    if matching_files:
                        self._handle_matching_files(matching_files, backup_flag, delete_all_in_folder)
                    else:
                        print(f"Error: No file or folder matching '{property_to_delete}' found in '{output_dir}' for star '{self.star_name}', epoch '{epoch_num}', band '{band}'.")

########################################                                     ########################################
    
    def _delete_file(self, file_path, backup_flag):
        """Helper method to delete a single file."""
        # If backup_flag is True, backup before deletion
        if backup_flag:
            self.backup_property(file_path, overwrite=False)
        # Delete the file
        try:
            os.remove(file_path)
            print(f"Deleted file: '{file_path}'")
        except Exception as e:
            print(f"Error deleting file '{file_path}': {e}")

########################################                                     ########################################
    
    def _handle_matching_files(self, matching_files, backup_flag, delete_all_in_folder):
        """Helper method to handle deletion of matching files."""
        if not matching_files:
            print("No matching files to delete.")
            return
        
        # Display files and number them
        print("\nFound the following matching files:")
        for idx, file_path in enumerate(matching_files, start=1):
            print(f"{idx}. {os.path.basename(file_path)}")
        
        # Prompt user for input
        while True:
            user_input = input("\nEnter the numbers of the files to delete (comma-separated), or 'all' to delete all: ")
            if user_input.strip().lower() == 'all':
                selected_indices = list(range(1, len(matching_files) + 1))
                break
            else:
                try:
                    selected_indices = [int(num.strip()) for num in user_input.split(',') if num.strip()]
                    if all(1 <= idx <= len(matching_files) for idx in selected_indices):
                        break
                    else:
                        print("Invalid input. Please enter valid file numbers.")
                except ValueError:
                    print("Invalid input. Please enter numbers separated by commas or 'all'.")
        
        # Delete selected files
        for idx in selected_indices:
            file_path = matching_files[idx - 1]
            if os.path.isdir(file_path):
                self._handle_folder_deletion(file_path, file_path, delete_all_in_folder)
            else:
                self._delete_file(file_path, backup_flag)
            
########################################                                     ########################################
    
    def _handle_folder_deletion(self, folder_path, backup_flag, delete_all_in_folder):
        """Helper method to handle deletion of files within a folder."""
        # List files in the folder
        files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if not files_in_folder:
            print(f"The folder '{folder_path}' is empty.")
            return

        # Display files and number them
        print(f"\nFolder '{folder_path}' contains the following files:")
        for idx, filename in enumerate(files_in_folder, start=1):
            print(f"{idx}. {filename}")

        # Determine which files to delete
        if delete_all_in_folder:
            selected_indices = list(range(1, len(files_in_folder) + 1))
        else:
            # Prompt user for input
            while True:
                user_input = input("\nEnter the numbers of the files to delete (comma-separated), or 'all' to delete all: ")
                if user_input.strip().lower() == 'all':
                    selected_indices = list(range(1, len(files_in_folder) + 1))
                    break
                else:
                    try:
                        selected_indices = [int(num.strip()) for num in user_input.split(',') if num.strip()]
                        if all(1 <= idx <= len(files_in_folder) for idx in selected_indices):
                            break
                        else:
                            print("Invalid input. Please enter valid file numbers.")
                    except ValueError:
                        print("Invalid input. Please enter numbers separated by commas or 'all'.")

        # Delete selected files
        for idx in selected_indices:
            file_name = files_in_folder[idx - 1]
            file_path = os.path.join(folder_path, file_name)
            self._delete_file(file_path, backup_flag)

        # Optionally delete the folder if empty
        if not os.listdir(folder_path):
            try:
                os.rmdir(folder_path)
                print(f"Deleted empty folder: '{folder_path}'")
            except Exception as e:
                print(f"Error deleting folder '{folder_path}': {e}")

########################################                                     ########################################
    
    def clean(self):
        """
        Cleans up empty folders within the 'output' directories for every band and epoch.

        Deletes any empty folders found within 'output' directories.

        Returns:
            None
        """
        star_path = os.path.join(self.data_dir, self.star_name)
        if not os.path.exists(star_path):
            print(f"No data found for star '{self.star_name}' in '{self.data_dir}'.")
            return

        # Iterate over epochs
        for epoch_dir in sorted(os.listdir(star_path)):
            print(epoch_dir)
            epoch_path = os.path.join(star_path, epoch_dir)
            if os.path.isdir(epoch_path) and epoch_dir.startswith('epoch'):
                epoch_num = epoch_dir.replace('epoch', '')

                # Iterate over bands
                for band in sorted(os.listdir(epoch_path)):
                    band_path = os.path.join(epoch_path, band)
                    print(band_path)
                    if os.path.isdir(band_path):
                        # Path to the 'output' directory
                        output_dir = os.path.join(band_path, 'output')
                        if os.path.exists(output_dir):
                            # Check for empty folders within the 'output' directory
                            for item in os.listdir(output_dir):
                                print(item)
                                item_path = os.path.join(output_dir, item)
                                if os.path.isdir(item_path):
                                    # Check if the directory is empty or only contains '.ipynb_checkpoints'
                                    print(item_path)
                                    items = os.listdir(item_path)
                                    if not items or (len(items) == 1 and items[0] == '.ipynb_checkpoints'):
                                        # Directory is empty or only contains '.ipynb_checkpoints'; delete it
                                        try:
                                            if '.ipynb_checkpoints' in items:
                                                checkpoint_file = os.path.join(item_path, '.ipynb_checkpoints')
                                                try:
                                                    # Attempt to change file permissions to allow deletion
                                                    os.chmod(checkpoint_file, 777)
                                                    os.remove(checkpoint_file)
                                                    print(f"Deleted file: '{checkpoint_file}'")
                                                except PermissionError:
                                                    print(f"Permission denied when trying to delete file: '{checkpoint_file}'")
                                                except Exception as e:
                                                    print(f"Error deleting file '{checkpoint_file}': {e}")
                                            os.rmdir(item_path)
                                            print(f"Deleted empty folder: '{item_path}'")
                                        except Exception as e:
                                            print(f"Error deleting folder '{item_path}': {e}")
                        else:
                            # 'output' directory does not exist
                            pass  # Nothing to do if output directory doesn't exist

########################################                                     ########################################
    
    def _load_file(self, file_path):
        """
        Helper method to load data from a file.
    
        Parameters:
            file_path (str): The path to the file to load.
    
        Returns:
            dict or Any: The data loaded from the file.
                         Returns None if an error occurs.
        """
        try:
            # Assuming the files are saved as .npz files
            if file_path.endswith('.npz'):
                with np.load(file_path, allow_pickle=True) as data:
                    # Check the keys in the saved file
                    keys = data.files
                    if len(keys) == 1 and keys[0] == 'data':
                        # The file contains a single array or list
                        return data['data']
                    else:
                        # The file contains multiple arrays or variables
                        return dict(data)
            else:
                print(f"Unsupported file format: '{file_path}'")
                return None
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")
            return None

########################################                                     ########################################

    def _generate_output_file_path(self, method_name, params, multiple_params):
        """
        Generates the output file path based on the method name and parameters.

        Parameters:
            method_name (str): The name of the method.
            params (dict): The parameters used in the method.
            multiple_params (bool): Indicates if multiple parameter sets are being processed.

        Returns:
            str: The full path to the output file.
        """
        epoch_num = params['epoch_num']
        band = params['band']
        # Base output directory
        base_output_dir = os.path.join(self.data_dir, self.star_name,f'epoch{epoch_num}',band,'output')

        if multiple_params:
            # Create a subfolder with the method name
            output_dir = os.path.join(base_output_dir, method_name)
            os.makedirs(output_dir, exist_ok=True)

            # Generate a filename based on parameters
            param_str = '_'.join(f"{k}{v}" for k, v in sorted(params.items()) if k not in ['epoch_num', 'band'])
            # Remove invalid filename characters
            param_str = param_str.replace('/', '_').replace('\\', '_').replace(':', '_')
            filename = f"{param_str}.npz"
            return os.path.join(output_dir, filename)
        else:
            # Only one set of parameters; save directly in the output folder with the method name
            os.makedirs(base_output_dir, exist_ok=True)
            filename = f"{method_name}.npz"
            return os.path.join(base_output_dir, filename)

########################################                                     ########################################
    

    def _save_result(self, output_file_path, result, params):
        """
        Saves the result and parameters to the specified output file path.

        Parameters:
            output_file_path (str): The path to save the result.
            result (Any): The result data to save.
            params (dict): The parameters used to generate the result.
        """
        # Include parameters in the saved file for traceability
        np.savez_compressed(output_file_path, result=result, params=params)

########################################                                     ########################################

    def save_property(self, property_name, property_data, epoch_number, band, overwrite=False, backup=True):

        # Get the path of the observation file
        file_path = self.get_file_path(epoch_number, band)
        # Determine the directory where the file is located
        directory = os.path.dirname(file_path)
        # Create an 'output' folder in the same directory if it doesn't exist
        output_dir = os.path.join(directory, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the property data to an npz file in the 'output' folder
        output_path = os.path.join(output_dir, f"{property_name}.npz")
        
        if os.path.exists(output_path):
            if not overwrite:
                raise FileExistsError(f"Cannot save property, file already exists: {output_path}. Set overwrite=True to proceed.")
            if backup:
                print(f"File exists. Creating a backup before overwriting: {output_path}")
                self.backup_property(output_path, overwrite)
        
        # Check the type of property_data and save accordingly
        if isinstance(property_data, dict):
            # Save dictionary using np.savez
            np.savez(output_path, **property_data)
        else:
            # Save array or list using np.savez with a default variable name
            np.savez(output_path, data=property_data)
        print(f"Property saved at {output_path}")


########################################                                     ########################################

    def backup_property(self, output_path, overwrite):
        # Get the path of the observation file
        # output_file_path = self._generate_output_file_path(method.__name__, params, multiple_params)
        # file_path = self.get_file_path(epoch_number, band)
        # # Determine the directory where the file is located
        directory = os.path.dirname(os.path.dirname(output_path))
        # # Locate the 'output' folder in the same directory
        # output_dir = os.path.join(directory, "output")
        # output_path = os.path.join(output_dir, f"{file_name}.npz")
        property_name = os.path.basename(output_path).split('.')[0]
        
        if os.path.exists(output_path):
            # Determine backup directory based on 'overwrite' parameter
            backup_type = 'overwritten' if overwrite else 'deleted'
    
            # Create backup directory structure in 'Backups' parallel to 'Data'
            backup_base_path = "Backups"
            # Build the backup directory path
            backup_dir = os.path.join(backup_base_path, backup_type, *directory.split(os.sep)[1:])
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup file name with date and time
            timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
            backup_file_name = f"{property_name}_backup_{timestamp}.npz"
            backup_path = os.path.join(backup_dir, backup_file_name)
            
            # Move the existing file to the backup location
            os.rename(output_path, backup_path)
            print(f"Backup created at {backup_path}")
        else:
            raise FileNotFoundError(f"Cannot create backup, original file not found: {output_path}")

########################################                                     ########################################


    def get_all_epoch_numbers(self):
        """
        Returns a list of all available epoch numbers for this star.

        Returns:
            list: A list of epoch numbers as strings.
        """
        epoch_numbers = [int(epoch[-1]) for epoch in self.observation_dict.keys()]
        return epoch_numbers

########################################                                     ########################################


    def get_all_bands(self):
        """
        Returns a list of all available bands for this star.

        Returns:
            list: A list of band names.
        """
        bands = ['NIR','VIS','UVB','COMBINED']
        bands = ['NIR','VIS','UVB','COMBINED2']
        return list(bands)

    
########################################              Data Handeling                  ########################################
    
    def list_available_properties(self):
        """
        Lists all available properties (files and folders) in the 'output' directories
        within each band and epoch for this star. If a property is a folder, it shows
        the number of files inside the folder.
    
        Returns:
            None
        """
        star_path = os.path.join(self.data_dir, self.star_name)
        if not os.path.exists(star_path):
            print(f"No data found for star '{self.star_name}' in '{self.data_dir}'.")
            return
    
        # Collect data for the table
        table_data = []
    
        # Iterate over epochs
        for epoch_dir in sorted(os.listdir(star_path)):
            epoch_path = os.path.join(star_path, epoch_dir)
            if os.path.isdir(epoch_path) and epoch_dir.startswith('epoch'):
                epoch_num = epoch_dir.replace('epoch', '')
    
                # Iterate over bands
                for band in sorted(os.listdir(epoch_path)):
                    band_path = os.path.join(epoch_path, band)
                    if os.path.isdir(band_path):
                        # Path to the 'output' directory
                        output_dir = os.path.join(band_path, 'output')
                        if os.path.exists(output_dir):
                            # List all files and folders in the 'output' directory
                            properties = os.listdir(output_dir)
                            for prop in properties:
                                prop_path = os.path.join(output_dir, prop)
                                if os.path.isdir(prop_path):
                                    prop_type = 'Folder'
                                    # Count the number of files in the folder
                                    num_files = len([f for f in os.listdir(prop_path) if os.path.isfile(os.path.join(prop_path, f))])
                                    details = f"{num_files} files"
                                else:
                                    prop_type = 'File'
                                    details = ''
                                table_data.append({
                                    'Epoch': epoch_num,
                                    'Band': band,
                                    'Property': prop,
                                    'Type': prop_type,
                                    'Details': details
                                })
                        else:
                            # 'output' directory does not exist
                            table_data.append({
                                'Epoch': epoch_num,
                                'Band': band,
                                'Property': '(No output directory)',
                                'Type': '',
                                'Details': ''
                            })
    
        # Check if any data was collected
        if not table_data:
            print(f"No properties found for star '{self.star_name}'.")
            return
    
        # Print the table header
        print(f"\nAvailable properties for star '{self.star_name}':\n")
        header = "{:<10} {:<10} {:<40} {:<10} {:<15}".format('Epoch', 'Band', 'Property', 'Type', 'Details')
        print(header)
        print('-' * len(header))
    
        # Print each row
        for row in table_data:
            print("{:<10} {:<10} {:<40} {:<10} {:<15}".format(
                row['Epoch'], row['Band'], row['Property'], row['Type'], row['Details']
            ))
    
        print('\n')
    
########################################                                     ########################################
    
    def load_property(self, property_name, epoch_number, band):
        """
        Loads and returns the data stored in the specified property for a given epoch and band.

        Parameters:
            property_name (str): The name of the property to load.
            epoch_number (int or str): The epoch number.
            band (str): The band name.

        Returns:
            dict or Any: The data loaded from the property file.
                         Returns None if the property is not found or an error occurs.

        """
        # Construct the path to the output directory for the specified epoch and band
        epoch_str = f'epoch{epoch_number}'
        output_dir = os.path.join(self.data_dir, self.star_name, epoch_str, band, 'output')

        if not os.path.exists(output_dir):
            print(f"No output directory found for star '{self.star_name}', epoch '{epoch_number}', band '{band}'.")
            return None

        # Construct the full path to the property
        property_path = os.path.join(output_dir, property_name + '.npz')

        if os.path.isfile(property_path):
            # Property is a file; load and return the data
            return self._load_file(property_path)
        elif os.path.isdir(property_path):
            # Property is a folder; list files and ask user which one to load
            files_in_folder = [f for f in os.listdir(property_path) if os.path.isfile(os.path.join(property_path, f))]
            if not files_in_folder:
                print(f"The folder '{property_path}' is empty.")
                return None

            # Display files and number them
            print(f"\nFolder '{property_path}' contains the following files:")
            for idx, filename in enumerate(files_in_folder, start=1):
                print(f"{idx}. {filename}")

            # Prompt user for input
            while True:
                user_input = input("\nEnter the number of the file to load: ")
                try:
                    selected_index = int(user_input.strip())
                    if 1 <= selected_index <= len(files_in_folder):
                        break
                    else:
                        print("Invalid input. Please enter a valid file number.")
                except ValueError:
                    print("Invalid input. Please enter a number corresponding to the file.")

            # Load and return the selected file
            selected_file = files_in_folder[selected_index - 1]
            selected_file_path = os.path.join(property_path, selected_file)
            return self._load_file(selected_file_path)
        else:
            # Property does not exist
            print(f"No file or folder named '{property_name}' found in '{output_dir}'.")
            return None


    
########################################                                     ########################################
    
    def load_observation(self, epoch_number, band=None):
        """
        Loads the FITS file for a specific observation.

        Parameters:
        - epoch_number (int): The epoch number to retrieve the observation from.
        - band (str, optional): The band (e.g., 'UVB', 'VIS', 'NIR'). If not specified, assumes combined.

        Returns:
        - FITSFile object: The loaded FITS file object.
        - None: If the file is not found or cannot be loaded.
        """
        # Get the full path of the FITS file
        if band == None:
            band = 'combined'
        file_path = self.get_file_path(epoch_number, band)
        print(file_path)

        if not file_path:
            print("Error: Could not find the requested file.")
            return None

        # Load the FITS file using FITSFile class from FitsClass
        try:
            fits_file = myfits(file_path)
            fits_file.load_data()
            return fits_file
        except Exception as e:
            print(f"Error loading FITS file: {e} error here")
            return None

########################################                                     ########################################
    
    def create_observation_table(self, epoch_list=None, band_list=None, attributes_list=None, print_table=True):
        """
        Scans the star's folder for FITS files and extracts 'DISPELEM' and 'TMID'.
        Can optionally filter by epoch number, sub-exposure number, and band.
        Can also extract additional attributes if specified.

        Parameters:
        - epoch_list (list, optional): List of epoch numbers to filter by.
        - band_list (list, optional): List of bands to filter by.
        - attributes_list (list, optional): List of additional FITS file attributes to extract.
        - print_table (bool, optional): Whether to print the table (default is True).

        Returns:
        - pd.DataFrame: A Pandas DataFrame containing the extracted data.
        """
        table_data = []
        
        try:
            # Loop through all epochs
            for epoch in self.observation_dict.keys():
                # Convert epoch to its numerical form ('epoch1' -> 1) for filtering
                epoch_num = int(epoch.replace('epoch', ''))

                # Filter by epoch if epoch_list is provided
                if epoch_list and epoch_num not in epoch_list:
                    continue

                # Loop through bands or combined FITS files
                for band_or_combined, filename in self.observation_dict[epoch].items():
                    # Filter by band if band_list is provided
                    if band_list and band_or_combined not in band_list: # and band_or_combined != 'combined':
                        continue

                    # Get the full file path
                    file_path = os.path.join(self.data_dir, self.star_name, epoch, band_or_combined, filename)
                    
                    # Try to open the FITS file and extract the data
                    try:
                        with fits.open(file_path) as hdul:

                            # Initialize a row with the basic information
                            row = {
                                'File Name': filename
                            }

                            # If additional attributes are requested, extract them
                            if attributes_list:
                                for attr in attributes_list:
                                    attr_name = str(attr)
                                    if attr_name == 'DISPELEM':
                                        attr_name = 'Band'
                                    elif attr_name == 'TMID':
                                        attr_name = 'Mid-Exposure Date (TMID)'
                                    row[attr_name] = hdul[0].header.get(attr, 'Unknown')
                                    if row[attr_name] == 'Unknown':
                                        row[attr_name] = hdul[1].header.get(attr, 'Unknown')

                                    
                            else:
                                # Get the band (DISPELEM) from the primary header (0th extension)
                                band = hdul[0].header.get('DISPELEM', 'Unknown')

                                # Get the mid-exposure date (TMID) from the first extension (1st HDU)
                                tmid = hdul[1].header.get('TMID', 'Unknown')

                                row = {
                                'File Name': filename,
                                'Band': band,
                                'Mid-Exposure Date (TMID)': tmid
                                }
                            
                            
                            # Add the row to the table data
                            table_data.append(row)

                    except Exception as e:
                        print(f"Error reading FITS file {file_path}: {e}")
        except KeyError as ke:
            print(f"Error: Invalid epoch or sub-exposure number provided. Details: {ke}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        # Create a Pandas DataFrame from the collected data
        df = pd.DataFrame(table_data)
        
        # Print the table if print_table is True
        if print_table:
            display(df)
        
        return df


########################################                 Plots                      ########################################
    
    def plot_spectra(self, epoch_nums=None, bands=None, save=False, linewidth = 1.5, scatter = False):
        """
        Plots flux vs. wavelength from the FITS files for the specified epochs and bands.

        Parameters:
            epoch_nums (list or None): List of epoch numbers to plot. Defaults to None (all epochs).
            bands (list or None): List of bands to plot. Defaults to None (all bands).
            save (bool): If True, saves the figure. Defaults to False.

        Returns:
            None
        """
        # Handle default epoch_nums and bands
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, list):
            epoch_nums = [epoch_nums]

        if bands is None:
            bands = self.get_all_bands()
        elif not isinstance(bands, list):
            bands = [bands]

        

        # Generate combinations of epoch_numbers and bands
        combinations = list(product(epoch_nums, bands))

        # Check if only one combination
        single_plot = len(combinations) == 1

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        # Loop over combinations and plot
        for epoch_num, band in combinations:
            epoch_str = f'epoch{epoch_num}'
            try:
                fits_file = self.load_observation(epoch_num, band=band)
                data = fits_file.data
                wavelength = data['WAVE'][0]
                flux = data['FLUX'][0]

                # Plot the spectrum
                if scatter:
                    plt.scatter(wavelength, flux, label=f'Epoch {epoch_num}, Band {band}', linewidth = linewidth)
                else:
                    plt.plot(wavelength, flux, label=f'Epoch {epoch_num}, Band {band}', linewidth = linewidth)

            except Exception as e:
                print(f"Error reading FITS file '{fits_file}': {e}")
                continue

        # Customize the plot
        plt.xlabel('Wavelength [nm]')
        plt.ylabel(r'Flux [$\frac{erg}{cm^{2}*s^{1}*angstrom^{1}}$]')
        plt.title(f'Spectrum of {self.star_name}')
        plt.legend()
        plt.grid(True)

        # Adjust layout
        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if single_plot:
                # Only one epoch and band
                epoch_num, band = combinations[0]
                output_dir = os.path.join(self.data_dir, self.star_name, f'epoch{epoch_num}', band, 'output', 'Figures')
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{self.star_name}_{timestamp}.png"
                save_path = os.path.join(output_dir, filename)
            else:
                # Multiple epochs and bands
                output_dir = os.path.join(self.data_dir, self.star_name, 'output', 'Figures')
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{self.star_name}_{timestamp}.png"
                save_path = os.path.join(output_dir, filename)

            plt.savefig(save_path)
            print(f"Figure saved to '{save_path}'")

        # Show the plot
        plt.show()

########################################                                       ########################################
    
    def plot_spectra_errors(self, epoch_nums=None, bands=None, save=False):
        """
        Plots the error of the flux vs. wavelength from the FITS files for the specified epochs and bands.
    
        Parameters:
            epoch_nums (list or None): List of epoch numbers to plot. Defaults to None (all epochs).
            bands (list or None): List of bands to plot. Defaults to None (all bands).
            save (bool): If True, saves the figure. Defaults to False.
    
        Returns:
            None
        """
        # Handle default epoch_nums and bands
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, list):
            epoch_nums = [epoch_nums]
    
        if bands is None:
            bands = self.get_all_bands()
        elif not isinstance(bands, list):
            bands = [bands]
    
        # Generate combinations of epoch_numbers and bands
        combinations = list(product(epoch_nums, bands))
    
        # Check if only one combination
        single_plot = len(combinations) == 1
    
        # Initialize the plot
        plt.figure(figsize=(10, 6))
    
        # Loop over combinations and plot
        for epoch_num, band in combinations:
            try:
                fits_file = self.load_observation(epoch_num, band=band)
                data = fits_file.data
                wavelength = data['WAVE'][0]
                error = data['ERR'][0]  # Assuming the error data is stored under 'ERR'
    
                # Plot the error spectrum
                plt.plot(wavelength, error, label=f'Epoch {epoch_num}, Band {band}')
    
            except Exception as e:
                print(f"Error reading FITS file for Epoch {epoch_num}, Band {band}: {e}")
                continue
    
        # Customize the plot
        plt.xlabel('Wavelength [nm]')
        plt.ylabel(r'Error [$\frac{erg}{cm^{2}\cdot s\cdot \AA}$]')
        plt.title(f'Error Spectrum of {self.star_name}')
        plt.legend()
        plt.grid(True)
    
        # Adjust layout
        plt.tight_layout()
    
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if single_plot:
                # Only one epoch and band
                epoch_num, band = combinations[0]
                output_dir = os.path.join(self.data_dir, self.star_name, f'epoch{epoch_num}', band, 'output', 'Figures')
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{self.star_name}_error_{timestamp}.png"
                save_path = os.path.join(output_dir, filename)
            else:
                # Multiple epochs and bands
                output_dir = os.path.join(self.data_dir, self.star_name, 'output', 'Figures')
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{self.star_name}_error_{timestamp}.png"
                save_path = os.path.join(output_dir, filename)
    
            plt.savefig(save_path)
            print(f"Figure saved to '{save_path}'")
    
        # Show the plot
        plt.show()

########################################                                       ########################################
    
    def plot_normalized_spectrum(self, epoch, band):
        """
        Plots the normalized spectrum for a specific epoch, sub-exposure, and band.
    
        Parameters:
        - epoch (int): Epoch identifier (e.g., 1 for 'epoch1').
        - band (str): Band identifier (e.g., 'UVB', 'VIS').
        """
        # Convert numerical epoch and sub-exposure to string keys
        epoch_key = f"epoch{epoch}"
    
        try:
            fits_file = self.load_observation(epoch,band)
            norm_data = self.load_property('norm_anchors_results',epoch,band)
            # spectrum = self.normalized_spectra[epoch_key][band]
            wavelength = fits_file.data['WAVE'][0]
            normalized_flux = norm_data['normalized_flux']
            # fitted_continuum = spectrum['fitted_continuum']
    
            plt.figure(figsize=(12, 6))
            plt.plot(wavelength, normalized_flux, label='Normalized Flux', color='blue')
            plt.plot(wavelength, normalized_flux / normalized_flux, '--', label='Fitted Continuum (Normalized)', color='red')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Normalized Flux')
            plt.title(f'Normalized Spectrum for {self.star_name} - {epoch_key}  - {band}')
            plt.legend()
            plt.grid(True)
            plt.show()
        except KeyError:
            print(f"No normalized spectrum found for {epoch_key}  -> {band}.")
        except Exception as e:
            print(f"Error plotting spectrum: {e}")

########################################                 Method Executer                      ########################################
    
    def execute_method(self, method, params={}, epoch_numbers=None, bands=None, overwrite=False, backup=True, save = True, parallel=False, max_workers=None):
        """
        Executes a given method with provided parameters, handling overwrite, backup, and parallel execution.

        Parameters:
            method (callable): The method from the Star class to execute.
            params (dict): A dictionary of additional parameters to pass to the method.
                           Values in params can be single values or lists.
            epoch_numbers (list, optional): A list of epoch numbers to process. If None, all available epochs are used.
            bands (list, optional): A list of bands to process. If None, all available bands are used.
            overwrite (bool): Whether to overwrite existing outputs. Defaults to False.
            backup (bool): Whether to backup existing outputs before overwriting. Defaults to True.
            parallel (bool): Whether to execute methods in parallel. Defaults to False.
            max_workers (int, optional): Maximum number of worker processes for parallel execution.

        Returns:
            list: A list of results from each method execution. None for skipped executions.
        """
        # Handle default epoch_numbers and bands
        if epoch_numbers is None:
            epoch_numbers = self.get_all_epoch_numbers()
        elif not isinstance(epoch_numbers, list):
            epoch_numbers = [epoch_numbers]

        if bands is None:
            bands = self.get_all_bands()
        elif not isinstance(bands, list):
            bands = [bands]

        # Ensure epoch_numbers and bands are lists
        epoch_numbers = list(epoch_numbers)
        bands = list(bands)

        # Convert params values to lists if they are not already
        param_keys = list(params.keys())
        param_values = []
        for value in params.values():
            if isinstance(value, list):
                param_values.append(value)
            else:
                param_values.append([value])

        # Generate all combinations
        all_values = [epoch_numbers, bands] + param_values
        combinations = list(product(*all_values))
        # print(combinations)

        # Generate params_list
        params_list = []
        for combination in combinations:
            # combination is a tuple: (epoch_num, band, param_value1, param_value2, ...)
            param_dict = {}
            param_dict['epoch_num'] = combination[0]
            param_dict['band'] = combination[1]
            for key, value in zip(param_keys, combination[2:]):
                param_dict[key] = value
            params_list.append(param_dict)
        
        # Determine if there are multiple sets of parameters
        multiple_params = len(params_list) > 1
        print(params_list)

        if parallel:
            if max_workers is None:
                max_workers = multiprocess.cpu_count() - 1
            # Use partial to fix the method argument
            method_wrapper_partial = partial(self._method_wrapper, method, overwrite=overwrite, backup=backup, multiple_params=multiple_params, save=save)
            with multiprocess.Pool(processes=max_workers) as pool:
                results = pool.map(method_wrapper_partial, params_list)
        else:
            results = []
            for params in params_list:
                result = self._method_wrapper(method, params, overwrite, backup, multiple_params, save=save)
                results.append(result)
        return results

########################################                                       ########################################

    def _method_wrapper(self, method, params, overwrite, backup, multiple_params,save):
        """
        Wrapper to execute the method with given parameters, handling overwrite and backup.

        Parameters:
            method (callable): The method to execute.
            params (dict): A dictionary of parameters to pass to the method.
            overwrite (bool): Whether to overwrite existing outputs.
            backup (bool): Whether to backup existing outputs before overwriting.
            multiple_params (bool): Indicates if multiple parameter sets are being processed.

        Returns:
            Any: The result of the method execution. None if the execution was skipped.
        """
        output_file_path = self._generate_output_file_path(method.__name__, params, multiple_params)

        try:
            if os.path.exists(output_file_path):
                if not overwrite:
                    print(f"Output '{output_file_path}' already exists. Skipping execution.")
                    return None
                else:
                    if backup:
                        self.backup_property(output_file_path, overwrite=True)
                    print(f"Overwriting existing output '{output_file_path}'.")

            # Execute the method with provided parameters
            result = method(**params)

            # Save the result
            if save:
                self._save_result(output_file_path, result, params)
    
                print(f"Saved result to '{output_file_path}'.")
            else:
                print(f'Didnt save due to save flag = False, but returned the results anyway')
            return result

        except Exception as e:
            print(f"Error executing method '{method.__name__}' with params {params}: {e}")
            return None


########################################                SIMABD                   ########################################

    def get_bat_id(self):
        """
        Fetches the BAT99 identifier for a given star from SIMBAD.
    
        Parameters:
            star_name (str): The name of the star to search for.
    
        Returns:
            str or None: The BAT99 identifier if found, else None.
        """
        # Construct the search URL
        base_url = 'https://simbad.u-strasbg.fr/simbad/sim-basic'
        params = {
            'Ident': self.star_name,
            'submit': 'SIMBAD search'
        }
    
        try:
            # Send a GET request to the SIMBAD server
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Check for HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching data: {e}")
            return None
    
        # Get the HTML content
        html_content = response.text
        index = html_content.find('BAT99')
        index = html_content.find('BAT99',index+1)
        index = html_content.find('BAT99',index+1)
        index = html_content.find('BAT99',index+1)
        index2 = html_content.find('\n',index+5+5)
        BAT_num =  html_content[index+5+5:index2]
    
        try:
            BAT_num_int = int(BAT_num)
            return BAT_num
        except:
            print(f"BAT99 identifier not found. The indexes were: {index} and {index2}. It found {BAT_num}")
            return None

########################################                Combined Spectra                   ########################################

    def combine_fits_files(self, epoch_num=None,band = None):
        try:
            # Determine which epochs to process
            # if epoch_num is None:
            #     epoch_num = [int(num[-1]) for num in self.observation_dict.keys()]
            #     print(epoch_num)
            #     # epoch_nums = [int(d.split('_')[1]) for d in epoch_dirs]
            # elif isinstance(epoch_num, int):
            #     epoch_num = [epoch_num]
    
            # for epoch_num in epoch_nums:
            try:
                # Load observations
                nir_fits = self.load_observation(epoch_num, 'NIR')
                vis_fits = self.load_observation(epoch_num, 'VIS')
                uvb_fits = self.load_observation(epoch_num, 'UVB')

                # Check if the FITS files are loaded properly
                if nir_fits is None or vis_fits is None or uvb_fits is None:
                    raise FileNotFoundError("One or more FITS files for the specified epoch do not exist.")

                # Create the COMBINED folder if it doesn't exist
                # combined_folder = os.path.join(os.path.dirname(os.path.dirname(self.get_file_path(epoch_num, band='NIR'))), 'COMBINED')
                combined_folder = os.path.join(os.path.dirname(os.path.dirname(self.get_file_path(epoch_num, band='NIR'))), 'COMBINED2')
                os.makedirs(combined_folder, exist_ok=True)

                # Extract WAVE, FLUX, and SNR data
                nir_wave = nir_fits.data['WAVE'][0]
                nir_flux = nir_fits.data['FLUX'][0]
                nir_snr = nir_fits.data['SNR'][0]
                nir_red_flux = nir_fits.data['FLUX_REDUCED'][0]
                vis_wave = vis_fits.data['WAVE'][0]
                vis_flux = vis_fits.data['FLUX'][0]
                vis_snr = vis_fits.data['SNR'][0]
                vis_red_flux = vis_fits.data['FLUX_REDUCED'][0]
                uvb_wave = uvb_fits.data['WAVE'][0]
                uvb_flux = uvb_fits.data['FLUX'][0]
                uvb_snr = uvb_fits.data['SNR'][0]
                uvb_red_flux = uvb_fits.data['FLUX_REDUCED'][0]

                # # Combine the data
                # combined_wave, combined_flux, combined_snr, combined_flux_reduced = self._combine_spectra(
                #     [uvb_wave, vis_wave, nir_wave],
                #     [uvb_flux, vis_flux, nir_flux],
                #     [uvb_snr, vis_snr, nir_snr],
                #     [nir_red_flux,vis_red_flux,uvb_red_flux]
                # )

                combined_wave, combined_flux, combined_snr, combined_flux_reduced = self._combine_spectra_tmp(
                    [uvb_wave, vis_wave, nir_wave],
                    [uvb_flux, vis_flux, nir_flux],
                    [uvb_snr, vis_snr, nir_snr],
                    [uvb_red_flux,vis_red_flux,nir_red_flux]
                )

                combined_data = {
                    'WAVE': combined_wave,
                    'FLUX': combined_flux,
                    # 'ERR': combined_err,
                    # 'QUAL': combined_qual,
                    'SNR': combined_snr,  # Update SNR calculation as needed
                    'FLUX_REDUCED': combined_flux_reduced,
                    # 'ERR_REDUCED': combined_err_reduced
                }

                # Create and save the new FITS file
                combined_fits_path = os.path.join(combined_folder, f'combined_bands.fits')
                self.create_combined_fits(nir_fits, combined_data, combined_fits_path)

                print(f'Combined FITS file saved at: {combined_fits_path}')

            except FileNotFoundError as e:
                print(f'Error: {e}')
            except Exception as e:
                print(f'An unexpected error occurred1: {e}')
        except Exception as e:
            print(f'An unexpected error occurred2: {e}')

########################################                                   ########################################

    def _combine_spectra_tmp(self, wave_list, flux_list, snr_list, flux_reduced_list):
        """
        Combine multiple spectra into a single spectrum by handling overlaps with different sampling,
        aligning mean fluxes before combination, and combining fluxes using the weighted SNR method.
        The alignment factor is applied to the entire spectrum, and the combined flux is used for subsequent overlaps.
    
        Parameters:
        wave_list : list of numpy arrays
            List containing wavelength arrays from different spectra.
        flux_list : list of numpy arrays
            List containing flux arrays corresponding to the wavelengths.
        snr_list : list of numpy arrays
            List containing SNR arrays corresponding to the wavelengths.
        flux_reduced_list : list of numpy arrays
            List containing FLUX_REDUCED arrays corresponding to the wavelengths.
    
        Returns:
        combined_wave : numpy array
            Combined wavelength array.
        combined_flux : numpy array
            Combined flux array.
        combined_snr : numpy array
            Combined SNR array.
        combined_flux_reduced: numpy array
            Combined FLUX_REDUCED array.
        """
        import numpy as np
    
        # Initialize the combined spectrum with the first spectrum
        combined_wave = wave_list[0]
        combined_flux = flux_list[0]
        combined_snr = snr_list[0]
        combined_flux_reduced = flux_reduced_list[0]
    
        # Loop over the rest of the spectra
        for idx in range(1, len(wave_list)):
            wave_current = wave_list[idx]
            flux_current = flux_list[idx]
            snr_current = snr_list[idx]
            flux_reduced_current = flux_reduced_list[idx]
    
            # Find overlap between combined_wave and wave_current
            overlap_start = max(combined_wave[0], wave_current[0])
            overlap_end = min(combined_wave[-1], wave_current[-1])
    
            if overlap_start < overlap_end:
                # There is an overlap
                # Get indices for the overlapping region in both spectra
                combined_overlap_indices = np.where((combined_wave >= overlap_start) & (combined_wave <= overlap_end))[0]
                print(f'combined_overlap_indices is {combined_overlap_indices}')
                current_overlap_indices = np.where((wave_current >= overlap_start) & (wave_current <= overlap_end))[0]

                mean_flux_combined = ut.robust_mean(combined_flux[combined_overlap_indices],1)
                mean_flux_current = ut.robust_mean(flux_current[current_overlap_indices],1)
                print(f'first the mean_flux_combined was {mean_flux_combined} and mean_flux_current is {mean_flux_current}')

                # plt.plot(combined_wave,combined_flux, label = f'idx = {idx}')

                if mean_flux_combined >= mean_flux_current:
                    print(f'entered case where mean_flux_combined >= mean_flux_current')
                    alignment_factor = mean_flux_combined / mean_flux_current
                    flux_current *= alignment_factor
                    snr_current *= alignment_factor
                    flux_reduced_current *= alignment_factor
                else:
                    print(f'entered case where mean_flux_combined < mean_flux_current')
                    alignment_factor = mean_flux_current / mean_flux_combined
                    combined_flux *= alignment_factor
                    combined_snr *= alignment_factor
                    combined_flux_reduced *= alignment_factor



                # Determine which spectrum has finer sampling in the overlap
                delta_combined = np.mean(np.diff(combined_wave[combined_overlap_indices]))
                delta_current = np.mean(np.diff(wave_current[current_overlap_indices]))
    
                if delta_combined <= delta_current:
                    # Combined spectrum has finer sampling
                    finer_wave = combined_wave[combined_overlap_indices]
                    coarser_wave = wave_current[current_overlap_indices]
                    flux_finer = combined_flux[combined_overlap_indices]
                    flux_coarser = flux_current[current_overlap_indices]
                    snr_finer = combined_snr[combined_overlap_indices]
                    snr_coarser = snr_current[current_overlap_indices]
                    flux_reduced_finer = combined_flux_reduced[combined_overlap_indices]
                    flux_reduced_coarser = flux_reduced_current[current_overlap_indices]
                    is_finer_combined =  True
                else:
                    # Current spectrum has finer sampling
                    finer_wave = wave_current[current_overlap_indices]
                    coarser_wave = combined_wave[combined_overlap_indices]
                    flux_finer = flux_current[current_overlap_indices]
                    flux_coarser = combined_flux[combined_overlap_indices]
                    snr_finer = snr_current[current_overlap_indices]
                    snr_coarser = combined_snr[combined_overlap_indices]
                    flux_reduced_finer = flux_reduced_current[current_overlap_indices]
                    flux_reduced_coarser = combined_flux_reduced[combined_overlap_indices]
                    is_finer_combined =  False
    
                # Interpolate coarser spectrum onto finer_wave
                interp_flux_coarser = np.interp(finer_wave, coarser_wave, flux_coarser)
                interp_snr_coarser = np.interp(finer_wave, coarser_wave, snr_coarser)
                interp_flux_reduced_coarser = np.interp(finer_wave, coarser_wave, flux_reduced_coarser)
    
                # Calculate mean fluxes and standard deviations in the overlap
                mean_flux_finer = np.mean(flux_finer)
                mean_flux_coarser = np.mean(interp_flux_coarser)
                std_flux_finer = np.std(flux_finer)
                std_flux_coarser = np.std(interp_flux_coarser)

                print(f'mean_flux_finer is {mean_flux_finer} and mean_flux_coarser is {mean_flux_coarser}')
                alignment_score = abs(mean_flux_combined - mean_flux_current) / np.sqrt(std_flux_finer**2 + std_flux_coarser**2)
    
                # # Calculate alignment factor and alignment score
                # if mean_flux_finer > mean_flux_coarser: # determines aligment factor, but which is finer now?
                #     alignment_factor = mean_flux_finer / mean_flux_coarser
                #     print(f'entered case where mean_flux_finer > mean_flux_coarser but is_finer_combined = {is_finer_combined}')
                #     if is_finer_combined:
                #         # Adjust current spectrum
                #         flux_current *= alignment_factor
                #         flux_reduced_current *= alignment_factor
                #         snr_current *= alignment_factor

                #     else: # coarser is the combined so 
                #         # Adjust combined spectrum
                #         combined_flux *= alignment_factor
                #         combined_flux_reduced *= alignment_factor
                #         combined_snr *= alignment_factor
                        
                #     # Recalculate interpolated fluxes after alignment
                #     interp_flux_coarser = interp_flux_coarser * alignment_factor
                #     interp_snr_coarser = interp_snr_coarser * alignment_factor
                #     interp_flux_reduced_coarser = interp_flux_reduced_coarser * alignment_factor
                # else:
                #     alignment_factor = mean_flux_coarser / mean_flux_finer
                #     print(f'entered case where mean_flux_finer <= mean_flux_coarser but is_finer_combined = {is_finer_combined}')
                #     if is_finer_combined:
                #         # Adjust combined spectrum
                #         combined_flux *= alignment_factor
                #         combined_flux_reduced *= alignment_factor
                #         combined_snr *= alignment_factor

                #         flux_finer *= alignment_factor
                #         flux_reduced_finer *= alignment_factor
                #         snr_finer *= alignment_factor

                #     else:
                #         # Adjust current spectrum
                #         flux_current *= alignment_factor
                #         flux_reduced_current *= alignment_factor
                #         snr_current *= alignment_factor
                    
                    # # Recalculate interpolated fluxes after alignment
                    # interp_flux_coarser = interp_flux_coarser * alignment_factor
                    # interp_snr_coarser = interp_snr_coarser * alignment_factor
                    # interp_flux_reduced_coarser = interp_flux_reduced_coarser * alignment_factor
                    
                
                        
                # Apply alignment factor to the entire coarser spectrum
                # if delta_combined <= delta_current:
                # else:
    
                
    
                # Combine fluxes using weighted SNR method
                weights_finer = snr_finer ** 2
                weights_coarser = interp_snr_coarser ** 2
                total_weights = weights_finer + weights_coarser
    
                combined_flux_overlap = (flux_finer * weights_finer + interp_flux_coarser * weights_coarser) / total_weights
                combined_snr_overlap = np.sqrt(total_weights)
                combined_flux_reduced_overlap = (flux_reduced_finer * weights_finer + interp_flux_reduced_coarser * weights_coarser) / total_weights
    
                # Update combined spectrum
                if delta_combined <= delta_current:
                    # Replace overlapping region in combined spectrum
                    combined_flux[combined_overlap_indices] = combined_flux_overlap
                    combined_snr[combined_overlap_indices] = combined_snr_overlap
                    combined_flux_reduced[combined_overlap_indices] = combined_flux_reduced_overlap
    
                    # Append non-overlapping part of current spectrum
                    non_overlap_indices_current = np.where(wave_current > overlap_end)[0]
                    if non_overlap_indices_current.size > 0:
                        combined_wave = np.concatenate((combined_wave, wave_current[non_overlap_indices_current]))
                        combined_flux = np.concatenate((combined_flux, flux_current[non_overlap_indices_current]))
                        combined_snr = np.concatenate((combined_snr, snr_current[non_overlap_indices_current]))
                        combined_flux_reduced = np.concatenate((combined_flux_reduced, flux_reduced_current[non_overlap_indices_current]))
                else:
                    # Replace overlapping region in current spectrum
                    flux_current[current_overlap_indices] = combined_flux_overlap
                    snr_current[current_overlap_indices] = combined_snr_overlap
                    flux_reduced_current[current_overlap_indices] = combined_flux_reduced_overlap
    
                    # Append non-overlapping part of combined spectrum
                    non_overlap_indices_combined = np.where(combined_wave < overlap_start)[0]
                    if non_overlap_indices_combined.size > 0:
                        wave_current = np.concatenate((combined_wave[non_overlap_indices_combined], wave_current))
                        flux_current = np.concatenate((combined_flux[non_overlap_indices_combined], flux_current))
                        snr_current = np.concatenate((combined_snr[non_overlap_indices_combined], snr_current))
                        flux_reduced_current = np.concatenate((combined_flux_reduced[non_overlap_indices_combined], flux_reduced_current))
    
                    # Set combined arrays to current
                    combined_wave = wave_current
                    combined_flux = flux_current
                    combined_snr = snr_current
                    combined_flux_reduced = flux_reduced_current

                # Sort the combined arrays
                sorted_indices = np.argsort(combined_wave)
                combined_wave = combined_wave[sorted_indices]
                combined_flux = combined_flux[sorted_indices]
                combined_snr = combined_snr[sorted_indices]
                combined_flux_reduced = combined_flux_reduced[sorted_indices]
    
                # Print alignment information
                print(f"Aligned spectra in overlap between {overlap_start:.2f} and {overlap_end:.2f} Å.")
                print(f"Alignment factor: {alignment_factor:.4f}, Alignment score: {alignment_score:.4f}")
    
            else:
                # No overlap; simply concatenate
                combined_wave = np.concatenate((combined_wave, wave_current))
                combined_flux = np.concatenate((combined_flux, flux_current))
                combined_snr = np.concatenate((combined_snr, snr_current))
                combined_flux_reduced = np.concatenate((combined_flux_reduced, flux_reduced_current))
    
                # Sort the combined arrays
                sorted_indices = np.argsort(combined_wave)
                combined_wave = combined_wave[sorted_indices]
                combined_flux = combined_flux[sorted_indices]
                combined_snr = combined_snr[sorted_indices]
                combined_flux_reduced = combined_flux_reduced[sorted_indices]

        # plt.legend()
    
        return combined_wave, combined_flux, combined_snr, combined_flux_reduced



########################################                                   ########################################
    def _combine_spectra(self, wave_list, flux_list, snr_list, flux_reduced_list):
        """
        Combine multiple spectra into a single spectrum by averaging overlapping wavelengths.
    
        Parameters:
        wave_list : list of numpy arrays
            List containing wavelength arrays from different spectra.
        flux_list : list of numpy arrays
            List containing flux arrays corresponding to the wavelengths.
        snr_list : list of numpy arrays
            List containing SNR arrays corresponding to the wavelengths.
        flux_reduced_list : list of numpy arrays
            List containing FLUX_REDUCED arrays corresponding to the wavelengths.
    
        Returns:
        combined_wave : numpy array
            Combined wavelength array.
        combined_flux : numpy array
            Combined flux array.
        combined_snr : numpy array
            Combined SNR array.
        combined_flux_reduced: numpy array
            Combined FLUX_REDUCED array.
        """
    
        # Concatenate all wavelengths, fluxes, and SNRs, and sort them by wavelength
        combined_wave = np.concatenate(wave_list)
        combined_flux = np.concatenate(flux_list)
        combined_snr = np.concatenate(snr_list)
        combined_flux_reduced = np.concatenate(flux_reduced_list)
    
        sorted_indices = np.argsort(combined_wave)
        combined_wave_sorted = combined_wave[sorted_indices]
        combined_flux_sorted = combined_flux[sorted_indices]
        combined_snr_sorted = combined_snr[sorted_indices]
        combined_flux_reduced_sorted = combined_flux_reduced[sorted_indices]
    
        # Initialize lists for the final combined data
        final_wave = []
        final_flux = []
        final_snr = []
        final_flux_reduced = []
    
        n = len(combined_wave_sorted)
        i = 0
        while i < n:
            # Start of a new group (cluster of overlapping wavelengths)
            cluster_waves = [np.round(combined_wave_sorted[i],2)]
            cluster_fluxes = [combined_flux_sorted[i]]
            cluster_snrs = [combined_snr_sorted[i]]
            cluster_flux_reduced = [combined_flux_reduced_sorted[i]]
            current_wave = np.round(combined_wave_sorted[i],2)
    
            i += 1
            # Collect all wavelengths within 0.02 nm of the current wavelength
            while i < n-1 and np.round(combined_wave_sorted[i]- current_wave,2)  < 0.02:
                cluster_waves.append(np.round(combined_wave_sorted[i],2))
                cluster_fluxes.append(combined_flux_sorted[i])
                cluster_snrs.append(combined_snr_sorted[i])
                cluster_flux_reduced.append(combined_flux_reduced_sorted[i])
                i += 1
                break
    
            # Calculate the mean wavelength of the cluster
            mean_wave = np.mean(cluster_waves)
    
            # Compute combined flux and SNR for the cluster
            if len(cluster_fluxes) == 1:
                combined_flux = cluster_fluxes[0]
                combined_snr = cluster_snrs[0]
                combined_flux_reduced = cluster_flux_reduced[0]
            else:
                # Weights are proportional to SNR squared
                weights = np.array(cluster_snrs) ** 2
                total_weight = np.sum(weights)
                combined_flux = np.sum(np.array(cluster_fluxes) * weights) / total_weight
                combined_snr = np.sqrt(np.sum(weights))
                combined_flux_reduced = np.sum(np.array(cluster_flux_reduced) * weights) / total_weight
    
            # Append the combined data to the final lists
            final_wave.append(mean_wave)
            final_flux.append(combined_flux)
            final_snr.append(combined_snr)
            final_flux_reduced.append(combined_flux_reduced)
    
        # Convert lists to numpy arrays
        combined_wave_array = np.array(final_wave)
        combined_flux_array = np.array(final_flux)
        combined_snr_array = np.array(final_snr)
        combined_flux_reduced_array = np.array(final_flux_reduced)
    
        return combined_wave_array, combined_flux_array, combined_snr_array, combined_flux_reduced_array

########################################                                   ########################################

    def create_combined_fits(self, nir_fits, combined_data, combined_fits_path):
        # Extract all columns from NIR FITS file
        nir_data = nir_fits.data
        col_names = nir_data.columns.names
    
        # Prepare fits columns using combined data
        columns = []
        nelem = len(combined_data['WAVE'])
    
        for col_name in col_names:
            # Get the unit from the original NIR FITS
            col_unit = nir_data.columns[col_name].unit
    
            # Determine the data type and format character
            original_format = nir_data.columns[col_name].format
    
            if 'D' in original_format:
                format_char = 'D'  # Double-precision float
                fill_value = np.nan
            elif 'E' in original_format:
                format_char = 'E'  # Single-precision float
                fill_value = np.nan
            elif 'J' in original_format:
                format_char = 'J'  # 32-bit integer
                fill_value = 0
            else:
                # Handle other data types as needed
                format_char = 'D'  # Default to double-precision float
                fill_value = np.nan
    
            # Set the format to include the correct repeat count
            col_format = f'{nelem}{format_char}'
    
            # Get the combined data array
            if col_name in combined_data:
                # Ensure data_array is a list containing the array
                data_array = [combined_data[col_name]]
            else:
                # Fill with appropriate fill value
                data_array = [np.full(nelem, fill_value)]
    
            # Create the fits.Column without the 'dim' parameter
            column = fits.Column(
                name=col_name,
                format=col_format,
                unit=col_unit,
                array=data_array
            )
            columns.append(column)

        # Create a new BinTableHDU from the columns
        hdu = fits.BinTableHDU.from_columns(columns)

        # Copy and update header
        
        hdu.header['DISPELEM'] = 'COMBINED'
        hdu.header['NELEM'] = nelem
        
        # Write the new HDU to a FITS file
        hdu.writeto(combined_fits_path, overwrite=True)

########################################               CCF                 ########################################

    def combined_normalized_template(self,epoch_num,band = None):
        fits_file= self.load_observation(1,'COMBINED')
        initial_template = fits_file.data['FLUX'][0]
        template_wave = fits_file.data['WAVE'][0]

########################################                                ########################################

    # def CCF(self,epoch_num,band = None):
    #     if epoch_num == 1:
    #         return 0
    #     fits_file= self.load_observation(1,'COMBINED')
    #     initial_template = fits_file.data['FLUX'][0]
    #     template_wave = fits_file.data['WAVE'][0]
    #     CCF = CCFclass(


########################################                                ########################################

    

    



