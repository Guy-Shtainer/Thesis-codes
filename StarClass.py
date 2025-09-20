import os
import shutil
import glob
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import utils as ut
from IPython.display import display
from astropy.io import fits
from threading import Thread
import multiprocess
from itertools import product
from functools import partial
from tabulate import tabulate
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia

# Tomers tools
from CCF import CCFclass
import plot as p
import TwoDImage as p2D

# My tools
import specs
from FitsClass import FITSFile as myfits
# import sys

# Assume catalogs.py is available and contains dictionaries named after catalogs
import catalogs

# SIMBAD
import requests
from bs4 import BeautifulSoup
import re


class Star:
    def __init__(self, star_name, data_dir, backup_dir, to_print=True):
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
        self.spectral_type = self.get_spectral_type()

        self.normalized_wavelength = []
        self.normalized_flux = []
        self.included_wave = []
        self.included_flux = []
        self.sensitivities = []
        self.to_print = to_print

    ########################################                 Printing                       ########################################

    def print(self, text, to_print=True):
        if to_print and self.to_print:
            print(text)

    ########################################                 File Handleing                      ########################################

    def get_file_path(self, epoch_number, band=None, D2=False):
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
        if band == None:
            band = 'COMBINED'
        if band in epoch_data:
            filename = epoch_data[band]
            if not D2:
                return os.path.join(self.data_dir, self.star_name, f'epoch{epoch_number}', band, filename)
            else:
                num = int(filename.split('.')[-2])
                num += 1
                filename = filename[:-15] + ':' + filename[-14:-12] + ':' + filename[-11:-8] + f'{num}'.zfill(
                    3) + '.fits'
                fits_directory = os.path.join(self.data_dir, self.star_name, f'epoch{epoch_number}', band, '2D image')
                print(fits_directory)
                fits_file = glob.glob(os.path.join(fits_directory, '**', '*.fits'), recursive=True)
                print(fits_file)
                return fits_file[0]
                return os.path.join(self.data_dir, self.star_name, f'epoch{epoch_number}', band, '2D image', filename)
        else:
            print(f"Band '{band}' not found in epoch '{epoch_number}")
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
                raise ValueError(
                    f"Error: {missing_str} parameter(s) are None for star '{self.star_name}'. To delete all options, set delete_all=True.")

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
                        print(
                            f"Error: No file or folder matching '{property_to_delete}' found in '{output_dir}' for star '{self.star_name}', epoch '{epoch_num}', band '{band}'.")

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
                user_input = input(
                    "\nEnter the numbers of the files to delete (comma-separated), or 'all' to delete all: ")
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
                                                    print(
                                                        f"Permission denied when trying to delete file: '{checkpoint_file}'")
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
        base_output_dir = os.path.join(self.data_dir, self.star_name, f'epoch{epoch_num}', band, 'output')

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
                raise FileExistsError(
                    f"Cannot save property, file already exists: {output_path}. Set overwrite=True to proceed.")
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
        bands = ['NIR', 'VIS', 'UVB', 'COMBINED']
        # bands = ['NIR','VIS','UVB','COMBINED2']
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
                                    num_files = len([f for f in os.listdir(prop_path) if
                                                     os.path.isfile(os.path.join(prop_path, f))])
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

    def load_property(self, property_name, epoch_num, band, from_backup=False):
        """
        Loads and returns the data stored in the specified property for a given epoch and band.
        Can also load from backup files if requested.

        Parameters:
            property_name (str): The name of the property to load.
            epoch_num (int or str): The epoch number.
            band (str): The band name.
            to_print (bool): Whether to print messages.
            from_backup (bool): Whether to check and offer loading from backup files.

        Returns:
            dict or Any: The data loaded from the property file.
                         Returns None if the property is not found or an error occurs.
        """
        # Construct paths
        epoch_str = f'epoch{epoch_num}'
        output_dir = os.path.join(self.data_dir, self.star_name, epoch_str, band, 'output')
        backup_dir = os.path.join(self.backup_dir, self.star_name, epoch_str, band)

        if not os.path.exists(output_dir):
            if self.to_print:
                print(f"No output directory found for epoch {epoch_num}, band {band}")
            if not from_backup:
                return None

        property_path = os.path.join(output_dir, property_name + '.npz')
        backup_pattern = os.path.join(backup_dir, f"{property_name}*.npz")

        # If from_backup is True or property doesn't exist, check backups
        if from_backup:
            # Check both deleted and overwritten folders
            self.print(os.getcwd())
            backup_base = "Backups"
            deleted_pattern = os.path.join(backup_base, "deleted", *backup_dir.split(os.sep)[1:], f"{property_name}*.npz")
            overwritten_pattern = os.path.join(backup_base, "overwritten", *backup_dir.split(os.sep)[1:], f"{property_name}*.npz")

            deleted_files = sorted(glob.glob(deleted_pattern))
            overwritten_files = sorted(glob.glob(overwritten_pattern))

            if deleted_files and not overwritten_files:
                self.print("\nFound backup files in 'deleted' folder:")
                for idx, bfile in enumerate(deleted_files, 1):
                    timestamp = os.path.basename(bfile).split('_')[-1].split('.')[0]
                    self.print(f"{idx}. {os.path.basename(bfile)} (saved on {timestamp})")
                self.print("\nEnter the number of the backup file to load (or 0 to cancel)")

            elif overwritten_files and not deleted_files:
                self.print("\nFound backup files in 'overwritten' folder:")
                for idx, bfile in enumerate(overwritten_files, 1):
                    timestamp = os.path.basename(bfile).split('_')[-1].split('.')[0]
                    self.print(f"{idx}. {os.path.basename(bfile)}")
                self.print("\nEnter the number of the backup file to load (or 0 to cancel)")

            elif deleted_files and overwritten_files:
                self.print("\nAvailable backup files:")
                self.print("\nFrom 'deleted' folder (use 'd#'):")
                for idx, bfile in enumerate(deleted_files, 1):
                    timestamp = os.path.basename(bfile).split('_')[-1].split('.')[0]
                    self.print(f"{idx}. {os.path.basename(bfile)}")

                self.print("\nFrom 'overwritten' folder (use 'o#'):")
                for idx, bfile in enumerate(overwritten_files, 1):
                    timestamp = os.path.basename(bfile).split('_')[-1].split('.')[0]
                    self.print(f"{idx}. {os.path.basename(bfile)} (saved on {timestamp})")
                self.print("\nEnter 'd#' for deleted or 'o#' for overwritten (e.g., 'd3' or 'o2'), or 0 to cancel")

            else:
                if self.to_print:
                    print(f"No backup files found for {property_name}")
                return None

            while True:
                try:
                    choice = input().strip()
                    if choice == '0':
                        return None

                    if deleted_files and not overwritten_files:
                        choice = int(choice)
                        if 1 <= choice <= len(deleted_files):
                            selected_file = deleted_files[choice - 1]
                            break

                    elif overwritten_files and not deleted_files:
                        choice = int(choice)
                        if 1 <= choice <= len(overwritten_files):
                            selected_file = overwritten_files[choice - 1]
                            break

                    else:  # Both folders have files
                        if choice.startswith('d'):
                            idx = int(choice[1:])
                            if 1 <= idx <= len(deleted_files):
                                selected_file = deleted_files[idx - 1]
                                break
                        elif choice.startswith('o'):
                            idx = int(choice[1:])
                            if 1 <= idx <= len(overwritten_files):
                                selected_file = overwritten_files[idx - 1]
                                break

                    self.print("Invalid choice. Please try again.")
                except ValueError:
                    self.print("Please enter a valid number or format (d#/o#).")

            try:
                data = np.load(selected_file, allow_pickle=True)
                if self.to_print:
                    print(f"Loaded backup file: {os.path.basename(selected_file)}")
                return data
            except Exception as e:
                self.print(f"Error loading backup file: {e}")
                return None

        # Load original file if it exists and from_backup is False
        if os.path.isfile(property_path):
            try:
                # data = np.load(property_path, allow_pickle=True)
                data = self._load_file(property_path)
                if self.to_print:
                    self.print(f"Loaded property from: {property_path}",to_print=self.to_print)
                return data
            except Exception as e:
                self.print(f"Error loading property: {e}")
                return None
        elif os.path.isdir(property_path):
            if self.to_print:
                print(f"Property '{property_name}' is a directory")
            return property_path
        else:
            if self.to_print:
                print(f"Property '{property_name}' not found")
            return None

    ########################################                                     ########################################

    def load_observation(self, epoch_num, band=None):
        """
        Loads the FITS file for a specific observation.

        Parameters:
        - epoch_num (int): The epoch number to retrieve the observation from.
        - band (str, optional): The band (e.g., 'UVB', 'VIS', 'NIR'). If not specified, assumes combined.

        Returns:
        - FITSFile object: The loaded FITS file object.
        - None: If the file is not found or cannot be loaded.
        """
        # Get the full path of the FITS file
        if band == None:
            band = 'combined'
        file_path = self.get_file_path(epoch_num, band)
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

    def load_2D_observation(self, epoch_num, band=None):
        """
        Loads the FITS file for a specific observation.

        Parameters:
        - epoch_num (int): The epoch number to retrieve the observation from.
        - band (str, optional): The band (e.g., 'UVB', 'VIS', 'NIR'). If not specified, assumes combined.

        Returns:
        - FITSFile object: The loaded FITS file object.
        - None: If the file is not found or cannot be loaded.
        """
        # Get the full path of the FITS file
        if band == None:
            band = 'COMBINED'
        file_path = self.get_file_path(epoch_num, band, D2=True)
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
                    if band_list and band_or_combined not in band_list:  # and band_or_combined != 'combined':
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

    def plot_spectra_old(self, epoch_nums=None, bands=None, save=False, linewidth=1.5, scatter=False):
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
            try:
                fits_file = self.load_observation(epoch_num, band=band)
                data = fits_file.data
                wavelength = data['WAVE'][0]
                flux = data['FLUX'][0]

                if band == 'UVB':
                    color = 'blue'
                elif band == 'VIS':
                    color = 'green'
                elif band == 'NIR':
                    color = 'red'

                # Plot the spectrum
                if scatter:
                    plt.scatter(wavelength, flux, label=f'Epoch {epoch_num}, Band {band}', linewidth=linewidth)
                else:
                    plt.plot(wavelength, flux, label=f'Epoch {epoch_num}, Band {band}', linewidth=linewidth,
                             color=color)

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

    def plot_spectra(
            self,
            epoch_nums=None,
            bands=None,
            save=False,
            linewidth=1.5,
            scatter=False,
            normalize=False,
            Rest_frame=False,
    ):
        """
        Plot flux vs. wavelength for selected epochs and bands.

        Flags:
          - normalize: if True, use 'norm_anchor_wavelengths' from COMBINED to build a
            piecewise-linear pseudo-continuum and divide the spectrum by it.
          - Rest_frame: if True, add each epoch's RV to the legend. If normalize=True,
            also shift the anchor points into a common velocity frame before interpolation.

        Notes:
          - Anchors and RVs are loaded via self.load_property(..., ep, 'COMBINED').
          - If RVs contain multiple lines, the mean full_RV across available lines is used.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime
        from itertools import product

        c_kms = 299792.458  # km/s

        # ---- helpers -----------------------------------------------------------
        def _coerce_anchors(obj):
            """Accept array-like or dict-like and return a clean, sorted 1D array (nm)."""
            if obj is None:
                return None
            cand = None
            if isinstance(obj, dict):
                for k in ("wavelengths", "anchors", "norm_anchor_wavelengths", "lambda", "lam"):
                    if k in obj:
                        cand = obj[k]
                        break
            else:
                cand = obj
            if cand is None:
                return None
            arr = np.asarray(cand, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            arr = np.unique(arr)
            return arr if arr.size >= 2 else None

        def _extract_full_rvs_from_dict(rv_dict):
            """From a dict of {line: structure}, pull full_RV values and average them."""
            vals = []
            for v in rv_dict.values():
                try:
                    v = v.item()
                except Exception:
                    pass
                if isinstance(v, dict) and "full_RV" in v:
                    try:
                        vals.append(float(v["full_RV"]))
                    except Exception:
                        pass
                else:
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
            if vals:
                return float(np.nanmean(vals))
            return None

        def _get_epoch_rv(ep):
            """Load COMBINED RVs for an epoch and return a single representative RV (km/s)."""
            try:
                rvs_prop = self.load_property("RVs", ep, "COMBINED")
            except Exception:
                rvs_prop = None
            if rvs_prop is None:
                return None
            if isinstance(rvs_prop, dict):
                for k in ("global", "combined", "CCF", "median"):
                    if k in rvs_prop:
                        try:
                            val = rvs_prop[k]
                            try:
                                val = val.item()
                            except Exception:
                                pass
                            if isinstance(val, dict) and "full_RV" in val:
                                return float(val["full_RV"])
                            return float(val)
                        except Exception:
                            pass
                return _extract_full_rvs_from_dict(rvs_prop)
            try:
                return float(rvs_prop)
            except Exception:
                return None

        def _get_epoch_anchors(ep):
            """Load COMBINED norm_anchor_wavelengths (nm) for an epoch."""
            try:
                anchors_prop = self.load_property("norm_anchor_wavelengths", ep, "COMBINED")
            except Exception:
                anchors_prop = None
            return _coerce_anchors(anchors_prop)

        # ---- select epochs & bands --------------------------------------------
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, list):
            epoch_nums = [epoch_nums]

        if bands is None:
            bands = self.get_all_bands()
        elif not isinstance(bands, list):
            bands = [bands]

        combinations = list(product(epoch_nums, bands))
        single_plot = len(combinations) == 1

        # ---- prefetch RVs (for legend and, if normalize, for rest-frame anchors) ----
        rv_by_epoch = {}
        rv_ref = None
        if Rest_frame:
            for ep in epoch_nums:
                rv_by_epoch[ep] = self.load_property("RVs", ep, "COMBINED")['C IV 5808-5812'].item()['full_RV']
            rv_vals = [v for v in rv_by_epoch.values() if v is not None and np.isfinite(v)]
            rv_ref = float(np.mean(rv_vals)) if rv_vals else 0.0

        # ---- prefetch anchors if needed ----------------------------------------
        anchors_by_epoch = {}
        if normalize:
            for ep in epoch_nums:
                anchors_by_epoch[ep] = _get_epoch_anchors(ep)

        # ---- prepare colors: distinct colors for COMBINED per epoch ------------
        combined_epochs = sorted({ep for ep, b in combinations if b == "COMBINED"})
        # obtain matplotlib's default color cycle
        prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
        palette = []
        if prop_cycle is not None:
            try:
                palette = prop_cycle.by_key().get("color", [])
            except Exception:
                palette = []
        if not palette:
            palette = [f"C{i}" for i in range(10)]
        combined_color_map = {ep: palette[i % len(palette)] for i, ep in enumerate(combined_epochs)}

        # ---- plot --------------------------------------------------------------
        plt.figure(figsize=(10, 6))

        for epoch_num, band in combinations:
            fits_file = None
            try:
                # Load the raw spectra from FITS
                fits_file = self.load_observation(epoch_num, band=band)
                data = fits_file.data
                wavelength = np.asarray(data["WAVE"][0], dtype=float)  # nm
                flux = np.asarray(data["FLUX"][0], dtype=float)

                # --- normalization using COMBINED anchors -----------------------
                if normalize:
                    anchors = anchors_by_epoch.get(epoch_num)
                    if anchors is None:
                        # optional fallback: try anchors saved with this band
                        try:
                            alt_prop = self.load_property("norm_anchor_wavelengths", epoch_num, band)
                        except Exception:
                            alt_prop = None
                        anchors = _coerce_anchors(alt_prop)

                    if anchors is not None and anchors.size >= 2:
                        # Shift anchors to a common velocity frame if requested
                        use_anchors = anchors
                        if Rest_frame:
                            rv_ep = rv_by_epoch.get(epoch_num, None)
                            if rv_ep is not None and np.isfinite(rv_ep):
                                rest_anchors = anchors / (1.0 + rv_ref / c_kms)
                                use_anchors = rest_anchors * (1.0 + rv_ep / c_kms)

                        # Keep only anchors inside the spectral coverage
                        in_band = (use_anchors >= np.nanmin(wavelength)) & (use_anchors <= np.nanmax(wavelength))
                        use_anchors = np.unique(use_anchors[in_band])

                        if use_anchors.size >= 2:
                            # Evaluate flux at anchor wavelengths
                            flux_at_anchors = np.interp(use_anchors, wavelength, flux)
                            # Linear continuum across the whole wavelength grid
                            continuum = np.interp(wavelength, use_anchors, flux_at_anchors)
                            good = np.isfinite(continuum) & (continuum > 0)
                            if np.any(good):
                                tmp = np.full_like(flux, np.nan, dtype=float)
                                tmp[good] = flux[good] / continuum[good]
                                flux = tmp
                            else:
                                print(
                                    f"[warn] Epoch {epoch_num}, Band {band}: continuum invalid; skipping normalization.")
                        else:
                            print(
                                f"[warn] Epoch {epoch_num}, Band {band}: <2 anchors within coverage; skipping normalization.")
                    else:
                        print(
                            f"[warn] Epoch {epoch_num}, Band {band}: no COMBINED 'norm_anchor_wavelengths'; skipping normalization.")

                # --- choose color ------------------------------------------------
                if band == "COMBINED":
                    color = combined_color_map.get(epoch_num, None)  # distinct per epoch
                elif band == "UVB":
                    color = "blue"
                elif band == "VIS":
                    color = "green"
                elif band == "NIR":
                    color = "red"
                else:
                    color = None  # let mpl choose

                # --- build label (include RV if requested) ----------------------
                rv_suffix = ""
                if Rest_frame:
                    rv_val = rv_by_epoch.get(epoch_num, None)
                    if rv_val is None or not np.isfinite(rv_val):
                        rv_suffix = " (RV=NA)"
                    else:
                        rv_suffix = f" (RV={rv_val:+.1f} km/s)"
                label = f"Epoch {epoch_num}, Band {band}{rv_suffix}"

                # --- plot -------------------------------------------------------
                if scatter:
                    plt.scatter(wavelength, flux, label=label, linewidth=linewidth,
                                s=max(1.0, 10.0 - 2.0 * linewidth), color=color)
                else:
                    plt.plot(wavelength, flux, label=label, linewidth=linewidth, color=color)

            except Exception as e:
                print(f"Error reading FITS file '{fits_file}': {e}")
                continue

        # ---- cosmetics & save --------------------------------------------------
        plt.xlabel("Wavelength [nm]")
        if normalize:
            plt.ylabel("Normalized flux (F/Fc)")
        else:
            plt.ylabel(r"Flux [$\mathrm{erg}\ \mathrm{cm}^{-2}\ \mathrm{s}^{-1}\ \mathrm{\AA}^{-1}$]")
        title = f"Spectrum of {self.star_name}"
        if Rest_frame and rv_ref is not None:
            title += f"  (Ref RV = {rv_ref:+.1f} km/s)"
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if single_plot:
                ep0, b0 = combinations[0]
                outdir = os.path.join(self.data_dir, self.star_name, f"epoch{ep0}", b0, "output", "Figures")
            else:
                outdir = os.path.join(self.data_dir, self.star_name, "output", "Figures")
            os.makedirs(outdir, exist_ok=True)
            fname = f"{self.star_name}_{timestamp}.png"
            path = os.path.join(outdir, fname)
            plt.savefig(path, dpi=200)
            print(f"Figure saved to '{path}'")

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

    def plot_normalized_spectra(self,
                                epoch_nums=None,
                                bands=None,
                                save=False,
                                separate=True,
                                separation=10,
                                bin_window=10,
                                clean=True,
                                compare=False,
                                bary_correction=False):
        """
        Plots normalized spectra for the specified epoch(s) and band(s).

        If `separate` is True, you get a slider to adjust the vertical gap
        between epochs (default gap=10).  If `compare` is True and a cleaned
        spectrum exists for a given epoch+band, its original counterpart is
        plotted just above it—and you get a second slider to tweak that “pair offset.”

        Parameters
        ----------
        epoch_nums : int or list of int, optional
            Epoch number(s). If None, uses all from get_all_epoch_numbers().
        bands : str or list of str, optional
            Which band(s) to plot (e.g. 'UVB' or ['UVB','VIS']). If None, uses
            self.get_all_bands() (you’ll need to implement that or substitute
            your own list).
        save : bool, optional
            If True, saves the figure to data_dir/star_name/output/Figures.
        separate : bool, optional
            If True, enables the “epoch spacing” slider.
        separation : float, optional
            Default vertical gap between different epochs. Default=10.
        bin_window : int, optional
            If >1, bin the flux over this many pixels using ut.robust_mean.
        clean : bool, optional
            If True, prefer 'clean_normalized_flux' over 'normalized_flux'.
        compare : bool, optional
            If True and a cleaned trace is drawn for an epoch+band, also plots
            the original just above it, with its own "(orig)" legend.
        """
        c_kms = 299792.458
        # ─── Prepare epochs ─────────────────────────────────────────
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, (list, tuple)):
            epoch_nums = [epoch_nums]
        if not epoch_nums:
            print("No epochs to plot.");
            return

        # ─── Prepare bands ──────────────────────────────────────────
        if bands is None:
            # replace this with your own method if needed
            bands = self.get_all_bands()
        elif isinstance(bands, str):
            bands = [bands]

        # ─── Binning helper ─────────────────────────────────────────
        def bin_data(wl, fl, wsize):
            n = len(wl)
            return (
                np.array([wl[i:i + wsize].mean() for i in range(0, n, wsize)]),
                np.array([ut.robust_mean(fl[i:i + wsize], 3) for i in range(0, n, wsize)])
            )

        # ─── Load & bin everything up front ─────────────────────────
        data_list = []
        for ep in epoch_nums:
            for band in bands:
                # — try cleaned first —
                d_clean = (self.load_property('cleaned_normalized_flux', ep, band)
                           if clean else None)
                if d_clean is not None:
                    wl = np.array(d_clean['wavelengths'])
                    fl = np.array(d_clean['normalized_flux'])
                    if bary_correction:
                        fits_file = self.load_observation(ep, 'VIS')
                        bary = fits_file.header['ESO QC VRAD BARYCOR']
                        wl = wl * (1 - bary / c_kms)  # correct to barycentric frame

                    wl_b, fl_b = (wl, fl) if bin_window <= 1 else bin_data(wl, fl, bin_window)
                    data_list.append({'epoch': ep, 'band': band,
                                      'wl': wl_b, 'fl': fl_b, 'type': 'clean'})
                    # optionally add original right above
                    if compare:
                        d_orig = self.load_property('normalized_flux', ep, band)
                        if d_orig is not None:
                            wl2 = np.array(d_orig['wavelengths'])
                            fl2 = np.array(d_orig['normalized_flux'])
                            wl2_b, fl2_b = (wl2, fl2) if bin_window <= 1 else bin_data(wl2, fl2, bin_window)
                            data_list.append({'epoch': ep, 'band': band,
                                              'wl': wl2_b, 'fl': fl2_b, 'type': 'orig'})
                    continue

                # — fallback to original only —
                d_orig = self.load_property('normalized_flux', ep, band)
                if d_orig is None:
                    print(f"▶ Missing data: epoch {ep}, band {band}.")
                    continue
                wl = np.array(d_orig['wavelengths'])
                fl = np.array(d_orig['normalized_flux'])
                if bary_correction:
                    fits_file = self.load_observation(ep, 'VIS')
                    bary = fits_file.header['ESO QC VRAD BARYCOR']
                    wl = wl * (1 + bary / c_kms)  # correct to barycentric frame
                wl_b, fl_b = (wl, fl) if bin_window <= 1 else bin_data(wl, fl, bin_window)
                data_list.append({'epoch': ep, 'band': band,
                                  'wl': wl_b, 'fl': fl_b, 'type': 'orig'})

        if not data_list:
            print("No spectra loaded.");
            return

        # ─── Figure + sliders setup ────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        if separate or compare:
            plt.subplots_adjust(bottom=0.25)

        # initial slider values
        sep_val = separation
        pair_val = separation / 4 if compare else 0

        # draw function
        def _draw(sep, pair):
            ax.clear()
            for entry in data_list:
                # all cleaned for this epoch sit at epoch_index*sep
                idx = epoch_nums.index(entry['epoch'])
                base = idx * sep
                y_off = base + (pair if entry['type'] == 'orig' else 0)
                lbl = f"Ep {entry['epoch']}, {entry['band']} ({entry['type']})"
                ax.plot(entry['wl'], entry['fl'] + y_off, label=lbl)
                ax.hlines(y_off + 1,
                          entry['wl'].min(), entry['wl'].max(),
                          linestyles='--', colors='gray', linewidth=0.8)

            ax.set_xlabel('Wavelength [nm]')
            ax.set_ylabel('Normalized Flux')
            ax.set_title(f"{self.star_name} — Bands: {', '.join(bands)}", fontsize=10)
            ax.legend(fontsize='small', ncol=2)
            ax.grid(True)
            fig.canvas.draw_idle()

        # initial draw
        _draw(sep_val, pair_val)

        # epoch‐spacing slider
        if separate and len(epoch_nums) > 1:
            ax_sep = fig.add_axes([0.1, 0.15, 0.35, 0.03])
            slider_sep = Slider(ax_sep, 'Separation',
                                0, separation * len(epoch_nums),
                                valinit=sep_val)
            slider_sep.on_changed(lambda v: _draw(v, slider_pair.val if compare else pair_val))

        # pair‐offset slider
        if compare:
            ax_pair = fig.add_axes([0.55, 0.15, 0.35, 0.03])
            slider_pair = Slider(ax_pair, 'Pair offset',
                                 0, separation,
                                 valinit=pair_val)
            slider_pair.on_changed(lambda v: _draw(slider_sep.val if separate else sep_val, v))

        # save if requested
        if save:
            outdir = os.path.join(self.data_dir, self.star_name, 'output', 'Figures')
            os.makedirs(outdir, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"{self.star_name}_normspec_{stamp}.png"
            plt.savefig(os.path.join(outdir, fname))
            print(f"Saved figure to: {outdir}/{fname}")

        plt.show()

    ########################################                                       ########################################

    def plot_extreme_rv_spectra(self,
                                emission_line,
                                emission_lines={},
                                band='COMBINED',
                                save=False,
                                linewidth=1.5,
                                scatter=False,
                                ts=None,
                                to_plot=False,
                                models=None,
                                correct_lmc=False,
                                lmc_rv_kms=262.2,
                                auto_annotate=True,  # turn on automatic labeling
                                species_filter=("H I", "He I", "He II", "C III", "C IV", "O V", "N III"),
                                match_to_peaks=False,  # only label lines that align with detected peaks
                                peak_prominence=0.03,  # tweak for your S/N; used if match_to_peaks=True
                                max_label_per_100A=6,  # de-clutter throttle
                                even_emission_line = False,
                                line_list=None,
                                label_y=0.94,          # y position in axes fraction (0..1); lower it if you want
                                label_stack_dy=0.06,   # how far the 2nd row drops when x-labels are too close
                                ):
        """
        ...
        correct_lmc (bool): If True, shift wavelengths to the LMC rest frame by
            removing the LMC systemic heliocentric radial velocity (default 262.2 km/s).
        lmc_rv_kms (float): LMC systemic radial velocity in km/s. Defaults to 262.2.
        """

        import importlib
        from math import sqrt

        def _get_default_wr_lines():
            # Curated (Å, rest) — lean list for WC + H/He work; expand as you wish
            return [
                ("He II 4200", 4199.83), ("Hγ", 4340.47), ("He II 4542", 4541.59),
                ("C III 4650", 4650.0), ("He II 4686", 4685.68), ("Hβ", 4861.33),
                ("He I 4922", 4921.93), ("O V 5590", 5590.0), ("C III 5696", 5696.0),
                ("C IV 5801", 5801.33), ("C IV 5812", 5812.02), ("He I 5876", 5875.62),
                ("N III 4640", 4640.64), ("Hα", 6562.80)
            ]

        def _get_default_hot_star_lines():
            # Fallback (lean) set if no list is provided
            return [
                ("Hγ", 4340.47), ("He I 4471", 4471.48), ("He II 4542", 4541.59),
                ("He II 4686", 4685.68), ("Hβ", 4861.33), ("He I 4922", 4921.93),
                ("He II 5411", 5411.52), ("O III 5592", 5592.26),
                ("C IV 5801", 5801.33), ("C IV 5812", 5811.98),
                ("He I 5876", 5875.62), ("Hα", 6562.80),
                # a few key NIR lines so labels still work out-of-the-box
                ("He I 1.0830 μm", 10830.25), ("Paβ 1.2818 μm", 12818.1),
                ("He II 1.693 μm", 16930.0), ("Brγ 2.166 μm", 21655.0),
                ("He II 2.1885 μm", 21885.0),
            ]

            # --- SINGLE, ROBUST line-list gatherer (uses your line_list if given) ---

        def _gather_line_list(species_filter):
            # Prefer the user-supplied list; otherwise provide a lean fallback
            if line_list is not None and len(line_list) > 0:
                lines = list(line_list)  # list of (label, lambda_A)
            else:
                lines = [
                    # lean fallback you already had
                    ("He II 4200", 4199.83),
                    ("Hγ", 4340.47),
                    ("He II 4542", 4541.59),
                    ("C III 4650", 4650.00),
                    ("He II 4686", 4685.68),
                    ("Hβ", 4861.33),
                    ("He I 4922", 4921.93),
                    ("O III 5592", 5592.26),
                    ("C IV 5801.33", 5801.33),
                    ("C IV 5811.98", 5811.98),
                    ("He I 5876", 5875.62),
                    ("Hα", 6562.80),
                    # a few NIR stalwarts so you still get labels without a custom list
                    ("He I 1.0830 μm", 10830.25),
                    ("Paβ 1.2818 μm", 12818.10),
                    ("He II 1.693 μm", 16930.00),
                    ("Brγ 2.166 μm", 21655.00),
                    ("He II 2.1885 μm", 21885.00),
                ]

            # Apply species filter by substring on the label (case-insensitive).
            # This keeps "Paβ ..." when species_filter contains "Pa", etc.
            if species_filter:
                tokens = tuple(s.upper() for s in species_filter)
                filt = [
                    (name, lam)
                    for (name, lam) in lines
                    if any(tok in name.upper() for tok in tokens)
                ]
                if filt:
                    lines = filt  # only apply if it actually matched something

            # Sort and de-duplicate by wavelength (0.3 Å tolerance)
            lines.sort(key=lambda x: x[1])
            dedup = []
            last_lam = None
            for name, lam in lines:
                if (last_lam is None) or (abs(lam - last_lam) > 0.3):
                    dedup.append((name, float(lam)))
                    last_lam = lam
            return dedup

        def _observed_factor(correct_lmc, lmc_rv_kms):
            c = 299_792.458
            return 1.0 if correct_lmc else (1.0 + lmc_rv_kms / c)

        def _find_emission_peaks(wave, flux, prom):
            # very light-weight peak picker without scipy dependency
            f = np.asarray(flux)
            w = np.asarray(wave)
            # local maxima
            m = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])
            idx = np.where(m)[0] + 1
            if prom is not None and prom > 0:
                baseline = np.median(f)
                idx = idx[(f[idx] - baseline) >= prom]
            return w[idx]

        def _throttle_labels(lines_sorted, per_100A):
            if per_100A <= 0:
                return lines_sorted
            keep = []
            last_bin_start = None
            count_in_bin = 0
            for name, lam in lines_sorted:
                bin_start = int(lam // 100) * 100
                if bin_start != last_bin_start:
                    last_bin_start = bin_start
                    count_in_bin = 0
                if count_in_bin < per_100A:
                    keep.append((name, lam))
                    count_in_bin += 1
            return keep

        def _annotate_lines_auto(ax, lines_rest, wl_min, wl_max,
                                 correct_lmc, lmc_rv_kms,
                                 match_to_peaks, peak_prominence,
                                 max_label_per_100A):
            # shift to plotting frame
            factor = _observed_factor(correct_lmc, lmc_rv_kms)
            lines_obs = [(name, lam * factor) for name, lam in lines_rest]

            # crop to the current x-range from the data
            xmin = float(np.min([wl_min.min(), wl_max.min()]))
            xmax = float(np.max([wl_min.max(), wl_max.max()]))
            in_range = [(n, l) for (n, l) in lines_obs if xmin <= l <= xmax]

            # Optionally require a nearby peak (helps avoid clutter if many lines)
            if match_to_peaks:
                # use the more-featured spectrum (max epoch) to find peaks
                peaks = _find_emission_peaks(wl_max, fl_max, prom=peak_prominence)
                c = 299_792.458
                max_dv = 200.0  # km/s tolerance for a match
                keep = []
                for name, lam in in_range:
                    # convert to dv from nearest peak
                    if peaks.size == 0:
                        continue
                    dv = np.min(np.abs((peaks - lam) / lam) * c)
                    if dv <= max_dv:
                        keep.append((name, lam))
                in_range = keep

            # throttle density
            in_range = _throttle_labels(sorted(in_range, key=lambda x: x[1]),
                                        per_100A=max_label_per_100A)

            # draw with minimal overlap: stack small y-offsets when too close
            ytop = ax.get_ylim()[1]
            used_x = []
            trans = ax.get_xaxis_transform()  # x in data, y in axes fraction
            for name, lam in in_range:
                ax.axvline(lam, linestyle=":", linewidth=0.9, alpha=0.75, zorder=1)
                y_frac = label_y
                if any(abs(lam - x0) < 6.0 for x0 in used_x):
                    y_frac = max(0.05, label_y - label_stack_dy)  # 2nd row if crowded
                ax.text(
                    lam,
                    y_frac,
                    name,
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=8,
                    transform=trans,
                    clip_on=True,
                    zorder=5,
                )
                used_x.append(lam)

        # ─── Get all epochs ─────────────────────────────────────────
        epoch_nums = self.get_all_epoch_numbers()

        # ─── Gather RVs ─────────────────────────────────────────────
        rvs = {}
        for ep in epoch_nums:
            try:
                RVs = self.load_property('RVs', ep, band)
                rv = RVs[emission_line].item()['full_RV']
            except Exception:
                rv = None
            if rv is not None:
                rvs[ep] = rv

        if not rvs:
            print(f"No radial velocities found for band '{band}'.")
            return

        # ─── Find extremes ───────────────────────────────────────────
        min_ep = min(rvs, key=rvs.get)
        max_ep = max(rvs, key=rvs.get)
        min_rv, max_rv = rvs[min_ep], rvs[max_ep]

        # ─── Load spectra ────────────────────────────────────────────
        def _load(ep):
            norm_flux = self.load_property('cleaned_normalized_flux', ep, band)
            if norm_flux is not None:
                tag = 'cleaned'
            else:
                norm_flux = self.load_property('normalized_flux', ep, band)
                tag = 'orig'
            wave_nm = norm_flux['wavelengths']  # in nm
            flux = norm_flux['normalized_flux']
            return wave_nm * 10.0, flux, tag  # convert to Å, return tag

        wl_min, fl_min, tag_min = _load(min_ep)
        wl_max, fl_max, tag_max = _load(max_ep)

        # ─── Optional: compute default scaling (blue→red) ──────────
        def _default_scale_factor(wl_a, fl_a, wl_b, fl_b, line_window=None):
            """
            Scale fl_a to fl_b: returns factor f for y' = 1 + (y-1)*f.
            Uses the max above continuum ~1.0 in the chosen window.
            Falls back to robust 95th-percentile if needed.
            """
            # overlap/window mask
            if line_window is not None:
                x0, x1 = line_window
                mask_a = (wl_a >= x0) & (wl_a <= x1)
                mask_b = (wl_b >= x0) & (wl_b <= x1)
            else:
                x0 = max(wl_a.min(), wl_b.min())*10
                x1 = min(wl_a.max(), wl_b.max())*10
                mask_a = (wl_a >= x0) & (wl_a <= x1)
                mask_b = (wl_b >= x0) & (wl_b <= x1)

            if mask_a.sum() < 10 or mask_b.sum() < 10:
                return 1.0

            seg_a = fl_a[mask_a]
            seg_b = fl_b[mask_b]

            # focus on emission above continuum ~1
            elev_a = seg_a - 1.0
            elev_b = seg_b - 1.0

            # use robust peak estimate
            try:
                peak_a = np.nanmax(elev_a)
                peak_b = np.nanmax(elev_b)
                if (peak_a <= 0) or (peak_b <= 0):
                    raise ValueError
            except Exception:
                qa = np.nanpercentile(elev_a, 95)
                qb = np.nanpercentile(elev_b, 95)
                peak_a = max(qa, 1e-3)
                peak_b = max(qb, 1e-3)

            print(f"Default scale factor (blue→red) near {emission_line}: {peak_b/peak_a:.3f} ")
            return float(peak_b / peak_a)

        # Decide the window: use emission_lines dict if available
        line_window = None
        if emission_lines and (emission_line in emission_lines):
            line_window = emission_lines[emission_line]

        # Compute default factor to scale blue (min epoch) to red (max epoch)
        # default_scale = 1.0
        # if even_emission_line:
        #     default_scale = _default_scale_factor(wl_min, fl_min, wl_max, fl_max, line_window)
        #     print(f"Applying default scale factor {default_scale:.3f} to blue (min epoch) spectrum.")
        #     # apply once so the first draw looks right
        #     fl_min = 1.0 + (fl_min - 1.0) * default_scale
        default_scale = 1.0
        if even_emission_line:
            default_scale = _default_scale_factor(wl_min, fl_min, wl_max, fl_max, line_window)
            print(f"Applying default scale factor {default_scale:.3f} to blue (min epoch) spectrum.")
            fl_min_base = fl_min.copy()  # Keep unmodified copy for slider
            fl_min = 1.0 + (fl_min - 1.0) * default_scale  # Apply initial scaling
        else:
            fl_min_base = fl_min.copy()

        # ─── Optional: correct for LMC systemic Doppler shift ────────
        if correct_lmc:
            c_kms = 299_792.458  # speed of light in km/s
            doppler_factor = 1.0 + (lmc_rv_kms / c_kms)  # non-relativistic λ_obs = λ_rest*(1+v/c)
            wl_min = wl_min / doppler_factor
            wl_max = wl_max / doppler_factor

        # ─── Plot ───────────────────────────────────────────────────
        fig, ax_main = plt.subplots(figsize=(10, 6))

        if scatter:
            blue = ax_main.scatter(wl_min, fl_min,
                                label=f'Epoch {min_ep} (RV={min_rv:.1f} km/s, {tag_min})',
                                linewidth=linewidth)
            red  = ax_main.scatter(wl_max, fl_max,
                                label=f'Epoch {max_ep} (RV={max_rv:.1f} km/s, {tag_max})',
                                linewidth=linewidth)
        else:
            blue, = ax_main.plot(wl_min, fl_min,
                                label=f'Epoch {min_ep} (RV={min_rv:.1f} km/s, {tag_min})',
                                linewidth=linewidth, color='blue')
            red,  = ax_main.plot(wl_max, fl_max,
                                label=f'Epoch {max_ep} (RV={max_rv:.1f} km/s, {tag_max})',
                                linewidth=linewidth, color='red')

        # Models
        if models:
            for tpl in models:
                try:
                    tpl = f'Data/Models_for_Guy/' + tpl
                    model_wave, model_flux = np.loadtxt(tpl, unpack=True)
                    ax_main.plot(model_wave, model_flux, label=os.path.basename(tpl),
                                linestyle='--', linewidth=1.0)
                except Exception as e:
                    print(f"Warning: could not load template '{tpl}': {e}")

        ax_main.set_xlabel('Wavelength [Å]' + (' (LMC rest frame)' if correct_lmc else ''))
        ax_main.set_ylabel('Normalized Flux')
        ax_main.set_title(f'{self.star_name}: Extreme RV Spectra ({emission_line})')
        ax_main.legend(fontsize='small', loc='best')
        ax_main.grid(True)
        fig.tight_layout()

        # keep references for your slider code
        blue_artist = blue
        red_artist  = red

        if scatter:
            # PathCollection; update with set_offsets
            blue_artist = ax_main.collections[0]   # first scatter call above
            red_artist  = ax_main.collections[1]   # second scatter call above
        else:
            # Line2D; update with set_ydata
            blue_artist = ax_main.lines[0]         # first plot call above (blue)
            red_artist  = ax_main.lines[1]         # second plot call above (red)

        # Create slider axis below the plot
        plt.subplots_adjust(bottom=0.18)  # give room for slider & button
        ax_slider = plt.axes([0.13, 0.06, 0.70, 0.03])  # [left, bottom, width, height]
        ax_reset  = plt.axes([0.85, 0.055, 0.10, 0.04])

        # Slider range: ±3× around default, but clamp to sensible bounds
        lo = max(0.1, default_scale / 10.0)
        hi = min(10.0, default_scale * 10.0)
        scale_slider = Slider(ax=ax_slider, label="Scale blue→red", valmin=lo, valmax=hi,
                              valinit=default_scale, valstep=0.001)

        reset_btn = Button(ax_reset, "Reset")

        # Keep pristine copies to avoid cumulative rounding
        # fl_min_base = fl_min.copy() if even_emission_line else (1.0 + (fl_min - 1.0) / default_scale).copy()
        # Note: if even_emission_line=False, fl_min is unscaled now; we still allow interactive scaling from 1.0

        def _apply_scale(factor):
            y_scaled = 1.0 + (fl_min_base - 1.0) * factor
            if scatter:
                # update blue scatter offsets
                blue_artist.set_offsets(np.column_stack((wl_min, y_scaled)))
            else:
                blue_artist.set_ydata(y_scaled)
            # Update legend text to reflect current factor (optional)
            leg = ax_main.get_legend()
            if leg is not None:
                # keep labels but append factor to blue
                labels = [t.get_text() for t in leg.get_texts()]
                if len(labels) >= 1:
                    # replace or append "(×f=...)" to first label
                    base = labels[0].split("×f=")[0].strip()
                    labels[0] = f"{base} ×f={factor:.3f}"
                    for txt, new in zip(leg.texts, labels):
                        txt.set_text(new)
            ax_main.figure.canvas.draw_idle()

        def _on_slider(val):
            _apply_scale(scale_slider.val)

        def _on_reset(event):
            scale_slider.reset()

        scale_slider.on_changed(_on_slider)
        reset_btn.on_clicked(_on_reset)

        # If we pre-scaled (even_emission_line), ensure legend shows the initial factor
        if even_emission_line:
            _apply_scale(default_scale)

        if auto_annotate:
            lines_rest = _gather_line_list(species_filter)
            _annotate_lines_auto(ax_main, lines_rest,
                                wl_min, wl_max,
                                correct_lmc, lmc_rv_kms,
                                match_to_peaks, peak_prominence,
                                max_label_per_100A)

        # ─── Zoom on line + robust Y limits on THE SAME AXES ─────────────
        if save and emission_lines:
            xmin_nm, xmax_nm = emission_lines[emission_line]
            x_low = float(min(xmin_nm, xmax_nm) * 10.0)
            x_high = float(max(xmin_nm, xmax_nm) * 10.0)
            width = x_high - x_low
            ax_main.set_xlim(x_low - 0.10 * width, x_high + 0.10 * width)

            # collect flux in-window from what you actually drew
            m1 = (wl_min >= x_low) & (wl_min <= x_high)
            m2 = (wl_max >= x_low) & (wl_max <= x_high)
            flux_window = np.concatenate([fl_min[m1], fl_max[m2]])
            flux_window = flux_window[np.isfinite(flux_window)]

            if flux_window.size:
                q1, q99 = np.nanpercentile(flux_window, [1, 99])
                if not np.isfinite(q1) or not np.isfinite(q99) or (q99 - q1) < 1e-3:
                    med = np.nanmedian(flux_window) if np.isfinite(flux_window).any() else 1.0
                    span = max(np.nanstd(flux_window), 5e-3)
                    y_lo, y_hi = med - 3 * span, med + 3 * span
                else:
                    pad = 0.10 * (q99 - q1)
                    y_lo, y_hi = q1 - pad, q99 + pad
            else:
                # fallback if masks are empty
                ax_main.relim()
                ax_main.autoscale_view(scalex=False, scaley=True)
                y_lo, y_hi = ax_main.get_ylim()

            # lock the y-limits on the real axes and force a draw before save
            ax_main.set_autoscale_on(False)  # prevent later artists from changing it
            ax_main.set_ylim(y_lo, y_hi)
            fig.canvas.draw()  # <- ensures limits are committed

        # ─── Save / Show / Close (no notebook rendering unless to_plot=True) ─
        if save:
            if ts is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = os.path.join(
                "../output",
                re.sub(r"[^A-Za-z0-9_-]", "_", self.star_name),
                "CCF",
                ts,
                emission_line,
            )
            os.makedirs(outdir, exist_ok=True)
            if even_emission_line:
                fname = f"{self.star_name}_{emission_line}_extremeRV_scaled.png"
            else:
                fname = f"{self.star_name}_{emission_line}_extremeRV_unscaled.png"
            path = os.path.join(outdir, fname)

            with plt.ioff():  # suppress notebook display
                fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved {emission_line} plot to {path}")

        if to_plot:
            plt.show()
        else:
            plt.close(fig)

    ########################################                                       ########################################

    def plot_2D_image(self, epoch_num, band, title='', ValMin=None, ValMax=None, norm=False, see_all=False):
        """
        Plots a 2D spectral image for a given observation epoch and band.

        Parameters:
        ----------
        epoch_num : int
            The observation epoch number to load the data from.
        band : str
            The spectral band for which the image is plotted (e.g., 'VIS', 'NIR').
        title : str, optional
            Title of the plot. If not provided, an automatic title is generated.
        ValMin : float, optional
            Minimum value for the image display scale. Defaults to None.
        ValMax : float, optional
            Maximum value for the image display scale. Defaults to None.
        norm : bool, optional
            If True, normalizes the image data. Defaults to False.
        see_all : bool, optional
            If True, displays the full image view. Defaults to False.

        Description:
        ------------
        This method retrieves the 2D observation data and wavelength information
        for the given epoch and band. It then generates a 2D image plot using the
        custom plotting utility. The title and display range can be customized.
        """
        # Load observation data
        fits_file_1D = self.load_observation(epoch_num, band)
        wavelengths = fits_file_1D.data['WAVE'][0]
        fits_file_2D = self.load_2D_observation(epoch_num, band)
        image_data = fits_file_2D.primary_data

        # Automatically generate title if not provided
        if not title:
            title = f"2D Image Plot for {self.star_name} (Epoch {epoch_num}, Band {band})"

        # Plot the 2D image
        p2D.Plot2DImage(
            image_data,
            wavelengths,
            band,
            title=title,
            ValMin=ValMin,
            ValMax=ValMax,
            norm=norm,
            see_all=see_all
        )

    ########################################                 Method Executer                      ########################################

    def execute_method(self, method, params={}, epoch_numbers=None, bands=None, overwrite=False, backup=True, save=True,
                       parallel=False, max_workers=None):
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
            method_wrapper_partial = partial(self._method_wrapper, method, overwrite=overwrite, backup=backup,
                                             multiple_params=multiple_params, save=save)
            with multiprocess.Pool(processes=max_workers) as pool:
                results = pool.map(method_wrapper_partial, params_list)
        else:
            results = []
            for params in params_list:
                result = self._method_wrapper(method, params, overwrite, backup, multiple_params, save=save)
                results.append(result)
        return results

    ########################################                                       ########################################

    def _method_wrapper(self, method, params, overwrite, backup, multiple_params, save):
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
        index = html_content.find('BAT99', index + 1)
        index = html_content.find('BAT99', index + 1)
        index = html_content.find('BAT99', index + 1)
        index2 = html_content.find('\n', index + 5 + 5)
        BAT_num = html_content[index + 5 + 5:index2]

        try:
            BAT_num_int = int(BAT_num)
            return BAT_num
        except:
            print(f"BAT99 identifier not found. The indexes were: {index} and {index2}. It found {BAT_num}")
            return None

    ########################################                SIMABD & Others                   ########################################

    def get_spectral_type(self):
        """
        Fetches the spectral type for a given star from SIMBAD.

        Parameters:
            None (uses self.star_name).

        Returns:
            str or None: The spectral type if found, else None.
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

        # Locate "Spectral type" in the HTML and extract the value
        spectral_type_label = 'Spectral type:        </SPAN>'
        spectral_type_start = html_content.find(spectral_type_label)
        if spectral_type_start == -1:
            print("Spectral type not found in the SIMBAD response.")
            return None

        # Parse the spectral type
        spectral_type_start = html_content.find('<TT>', spectral_type_start) + len('<TT>')
        spectral_type_end = html_content.find('</TT>', spectral_type_start)
        spectral_type = html_content[spectral_type_start:spectral_type_end].strip()

        if spectral_type:
            return spectral_type
        else:
            print("Spectral type could not be parsed.")
            return None

    ########################################                                   ########################################

    def get_catalogs_data(self, queries, catalogs_list=None):
        # Normalize queries input
        if isinstance(queries, str):
            queries = [queries]

        # Use the catalogs list from catalogs.py if no custom list is provided
        if catalogs_list is None:
            selected_catalogs = catalogs.catalogs[:]  # make a copy of the list
        else:
            if isinstance(catalogs_list, str):
                catalogs_list = [catalogs_list]
            # Filter to ensure we only keep the catalogs that exist in catalogs.py
            selected_catalogs = [c for c in catalogs_list if c in catalogs.catalogs]
            if len(selected_catalogs) < len(catalogs_list):
                missing = set(catalogs_list) - set(selected_catalogs)
                print(f"The following catalogs were not found: {missing}")
                print("Available catalogs are:")
                for cat in catalogs.catalogs:
                    print(f" - {cat}")

        results = []  # Will store (Catalog, Query_Key, Query_Value)

        # Handle SIMBAD queries
        if "SIMBAD" in selected_catalogs:
            Simbad.reset_votable_fields()
            simbad_fields = catalogs.SIMBAD.keys()  # Get available fields from catalogs.py
            for q in queries:
                if q in simbad_fields:
                    Simbad.add_votable_fields(q)
                else:
                    print(f"Query '{q}' is not recognized by SIMBAD. Available fields:")
                    for i, field in enumerate(simbad_fields, start=1):
                        print(f"{i}. {field}")
                    print("0. Skip this query")
                    user_in = input("Enter the number of the query to use, or 0 to skip: ")
                    try:
                        choice = int(user_in)
                    except ValueError:
                        choice = 0
                    if choice == 0:
                        print(f"Skipping query '{q}' for SIMBAD.")
                    elif 1 <= choice <= len(simbad_fields):
                        chosen_field = list(simbad_fields)[choice - 1]
                        Simbad.add_votable_fields(chosen_field)
                        queries[queries.index(q)] = chosen_field  # Update the query to the chosen field
                    else:
                        print("Invalid choice. Skipping this query for SIMBAD.")

            try:
                sim_res = Simbad.query_object(self.star_name)
                if sim_res is None or len(sim_res) == 0:
                    print(f"Star {self.star_name} not found in SIMBAD.")
                else:
                    for q in queries:
                        if q in sim_res.colnames:
                            value = sim_res[q][0]
                            results.append(("SIMBAD", q, value))
                        else:
                            print(f"Query '{q}' not found in SIMBAD response.")
            except Exception as e:
                print(f"Error querying SIMBAD: {e}")

            # Remove SIMBAD from the selected catalogs after processing
            selected_catalogs.remove("SIMBAD")

        # Handle Vizier queries
        viz = Vizier(row_limit=-1)
        for cat in selected_catalogs:
            try:
                viz_res = viz.query_object(self.star_name, catalog=cat)
                if viz_res is None or len(viz_res) == 0:
                    print(f"No results found for {self.star_name} in {cat}.")
                    continue

                table = viz_res[0]
                for q in queries:
                    if q in table.colnames:
                        value = table[q][0] if len(table) > 0 else None
                        results.append((cat, q, value))
                    else:
                        print(f"Query '{q}' not found in {cat}. Available fields:")
                        for i, field in enumerate(table.colnames, start=1):
                            print(f"{i}. {field}")
                        print("0. Skip this query")
                        user_in = input("Enter the number of the query to use, or 0 to skip: ")
                        try:
                            choice = int(user_in)
                        except ValueError:
                            choice = 0
                        if choice == 0:
                            print(f"Skipping this query in {cat}.")
                        elif 1 <= choice <= len(table.colnames):
                            chosen_query = table.colnames[choice - 1]
                            value = table[chosen_query][0] if len(table) > 0 else None
                            results.append((cat, chosen_query, value))
                        else:
                            print("Invalid choice. Skipping this query in {cat}.")
            except Exception as e:
                print(f"Error querying VizieR catalog {cat}: {e}")

        # Handle Gaia queries (I_355_GAIADR3)
        if "I_355_GAIADR3" in selected_catalogs:
            gaia_fields = catalogs.I_355_GAIADR3.keys()  # Get available fields from catalogs.py
            valid_queries = []
            for q in queries:
                if q in gaia_fields:
                    valid_queries.append(q)
                else:
                    print(f"Query '{q}' is not recognized by GAIA. Available fields:")
                    for i, field in enumerate(gaia_fields, start=1):
                        print(f"{i}. {field}")
                    print("0. Skip this query")
                    user_in = input("Enter the number of the query to use, or 0 to skip: ")
                    try:
                        choice = int(user_in)
                    except ValueError:
                        choice = 0
                    if choice == 0:
                        print(f"Skipping query '{q}' for GAIA.")
                    elif 1 <= choice <= len(gaia_fields):
                        chosen_field = list(gaia_fields)[choice - 1]
                        valid_queries.append(chosen_field)
                    else:
                        print("Invalid choice. Skipping this query for GAIA.")

            if valid_queries:
                try:
                    gaia_query = f"SELECT {', '.join(valid_queries)} FROM gaiadr3.gaia_source WHERE source_id = {self.star_name}"
                    gaia_job = Gaia.launch_job(gaia_query)
                    gaia_res = gaia_job.get_results()
                    if gaia_res is None or len(gaia_res) == 0:
                        print(f"No results found for {self.star_name} in Gaia.")
                    else:
                        for q in valid_queries:
                            if q in gaia_res.colnames:
                                value = gaia_res[q][0]
                                results.append(("I_355_GAIADR3", q, value))
                except Exception as e:
                    print(f"Error querying Gaia: {e}")

        # Now we have a list of (Catalog, Query_Key, Query_Value)
        # Print a table with star name, catalog, query key, and value
        if results:
            df = pd.DataFrame(results, columns=["Catalog", "Query Key", "Query Value"])
            df.insert(0, "Star Name", self.star_name)
            print(tabulate(df, headers="keys", tablefmt="pretty"))
        else:
            print("No queries were resolved.")

    def get_catalogs_data2(self, queries, catalogs_list=None):
        # Normalize queries input
        if isinstance(queries, str):
            queries = [queries]

        # Use the catalogs list from catalogs.py if no custom list is provided
        if catalogs_list is None:
            selected_catalogs = catalogs.catalogs[:]  # make a copy of the list
        else:
            if isinstance(catalogs_list, str):
                catalogs_list = [catalogs_list]
            # Filter to ensure we only keep the catalogs that exist in catalogs.py
            selected_catalogs = [c for c in catalogs_list if c in catalogs.catalogs]
            if len(selected_catalogs) < len(catalogs_list):
                missing = set(catalogs_list) - set(selected_catalogs)
                print(f"The following catalogs were not found: {missing}")
                print("Available catalogs are:")
                for cat in catalogs.catalogs:
                    print(f" - {cat}")

        results = []  # Will store (Catalog, Query_Key, Query_Value)

        # Handle SIMBAD queries
        if "SIMBAD" in selected_catalogs:
            # Create a custom Simbad instance
            custom_simbad = Simbad()
            simbad_fields = catalogs.SIMBAD.keys()  # Get available fields from catalogs.py

            # Add fields from queries
            valid_queries = []
            for q in queries:
                if q.lower() == "sptype":  # Special handling for spectral type
                    custom_simbad.add_votable_fields("sptype")
                    valid_queries.append("SP_TYPE")
                elif q in simbad_fields:
                    custom_simbad.add_votable_fields(q)
                    valid_queries.append(q)
                else:
                    print(f"Query '{q}' is not recognized by SIMBAD. Available fields:")
                    for i, field in enumerate(simbad_fields, start=1):
                        print(f"{i}. {field}")
                    print("0. Skip this query")
                    user_in = input("Enter the number of the query to use, or 0 to skip: ")
                    try:
                        choice = int(user_in)
                    except ValueError:
                        choice = 0
                    if choice == 0:
                        print(f"Skipping query '{q}' for SIMBAD.")
                    elif 1 <= choice <= len(simbad_fields):
                        chosen_field = list(simbad_fields)[choice - 1]
                        custom_simbad.add_votable_fields(chosen_field)
                        valid_queries.append(chosen_field)
                    else:
                        print("Invalid choice. Skipping this query for SIMBAD.")

            # Query SIMBAD
            try:
                sim_res = custom_simbad.query_object(self.star_name)
                if sim_res is None or len(sim_res) == 0:
                    print(f"Star {self.star_name} not found in SIMBAD.")
                else:
                    for q in valid_queries:
                        if q in sim_res.colnames:
                            value = sim_res[q][0]
                            results.append(("SIMBAD", q, value))
                        else:
                            print(f"Query '{q}' not found in SIMBAD response.")
            except Exception as e:
                print(f"Error querying SIMBAD: {e}")

            # Remove SIMBAD from the selected catalogs after processing
            selected_catalogs.remove("SIMBAD")

        # Handle Vizier queries
        viz = Vizier(row_limit=-1)
        for cat in selected_catalogs:
            try:
                viz_res = viz.query_object(self.star_name, catalog=cat)
                if viz_res is None or len(viz_res) == 0:
                    print(f"No results found for {self.star_name} in {cat}.")
                    continue

                table = viz_res[0]
                for q in queries:
                    if q in table.colnames:
                        value = table[q][0] if len(table) > 0 else None
                        results.append((cat, q, value))
                    else:
                        print(f"Query '{q}' not found in {cat}. Available fields:")
                        for i, field in enumerate(table.colnames, start=1):
                            print(f"{i}. {field}")
                        print("0. Skip this query")
                        user_in = input("Enter the number of the query to use, or 0 to skip: ")
                        try:
                            choice = int(user_in)
                        except ValueError:
                            choice = 0
                        if choice == 0:
                            print(f"Skipping this query in {cat}.")
                        elif 1 <= choice <= len(table.colnames):
                            chosen_query = table.colnames[choice - 1]
                            value = table[chosen_query][0] if len(table) > 0 else None
                            results.append((cat, chosen_query, value))
                        else:
                            print("Invalid choice. Skipping this query in {cat}.")
            except Exception as e:
                print(f"Error querying VizieR catalog {cat}: {e}")

        # Handle Gaia queries
        if "I_355_GAIADR3" in selected_catalogs:
            gaia_fields = catalogs.I_355_GAIADR3.keys()  # Get available fields from catalogs.py
            valid_queries = []
            for q in queries:
                if q in gaia_fields:
                    valid_queries.append(q)
                else:
                    print(f"Query '{q}' is not recognized by GAIA. Available fields:")
                    for i, field in enumerate(gaia_fields, start=1):
                        print(f"{i}. {field}")
                    print("0. Skip this query")
                    user_in = input("Enter the number of the query to use, or 0 to skip: ")
                    try:
                        choice = int(user_in)
                    except ValueError:
                        choice = 0
                    if choice == 0:
                        print(f"Skipping query '{q}' for GAIA.")
                    elif 1 <= choice <= len(gaia_fields):
                        chosen_field = list(gaia_fields)[choice - 1]
                        valid_queries.append(chosen_field)
                    else:
                        print("Invalid choice. Skipping this query for GAIA.")

            if valid_queries:
                try:
                    gaia_query = f"SELECT {', '.join(valid_queries)} FROM gaiadr3.gaia_source WHERE source_id = {self.star_name}"
                    gaia_job = Gaia.launch_job(gaia_query)
                    gaia_res = gaia_job.get_results()
                    if gaia_res is None or len(gaia_res) == 0:
                        print(f"No results found for {self.star_name} in Gaia.")
                    else:
                        for q in valid_queries:
                            if q in gaia_res.colnames:
                                value = gaia_res[q][0]
                                results.append(("I_355_GAIADR3", q, value))
                except Exception as e:
                    print(f"Error querying Gaia: {e}")

        # Now we have a list of (Catalog, Query_Key, Query_Value)
        # Print a table with star name, catalog, query key, and value
        if results:
            df = pd.DataFrame(results, columns=["Catalog", "Query Key", "Query Value"])
            df.insert(0, "Star Name", self.star_name)
            print(tabulate(df, headers="keys", tablefmt="pretty"))
        else:
            print("No queries were resolved.")

    ########################################                Combined Spectra                   ########################################

    def combine_fits_files(self, epoch_num=None, band=None, overwrite=False, backup=False, save=False):
        try:
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
                combined_folder = os.path.join(
                    os.path.dirname(os.path.dirname(self.get_file_path(epoch_num, band='NIR'))), 'COMBINED')
                # combined_folder = os.path.join(os.path.dirname(os.path.dirname(self.get_file_path(epoch_num, band='NIR'))), 'COMBINED2')
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

                combined_wave, combined_flux, combined_snr, combined_flux_reduced, aligment_data = self._combine_spectra(
                    [uvb_wave, vis_wave, nir_wave],
                    [uvb_flux, vis_flux, nir_flux],
                    [uvb_snr, vis_snr, nir_snr],
                    [uvb_red_flux, vis_red_flux, nir_red_flux]
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

                if save:
                    # Create and save the new FITS file
                    combined_fits_path = os.path.join(combined_folder, f'combined_bands.fits')
                    self.create_combined_fits(nir_fits, combined_data, combined_fits_path)
                    self.save_property('aligment_data', aligment_data, epoch_number=epoch_num, band='COMBINED',
                                       overwrite=True, backup=True)

                print(f'Combined FITS file saved at: {combined_fits_path}')

            except FileNotFoundError as e:
                print(f'Error: {e}')
            except Exception as e:
                print(f'An unexpected error occurred1: {e}')
        except Exception as e:
            print(f'An unexpected error occurred2: {e}')

    ########################################                                   ########################################

    def _combine_spectra(self, wave_list, flux_list, snr_list, flux_reduced_list, align=True):
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

        aligment_data = {}

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
                combined_overlap_indices = np.where((combined_wave >= overlap_start) & (combined_wave <= overlap_end))[
                    0]
                print(f'combined_overlap_indices is {combined_overlap_indices}')
                current_overlap_indices = np.where((wave_current >= overlap_start) & (wave_current <= overlap_end))[0]

                # Determine which spectrum has finer sampling in the overlap
                delta_combined = np.mean(np.diff(combined_wave[combined_overlap_indices]))
                delta_current = np.mean(np.diff(wave_current[current_overlap_indices]))

                mean_flux_combined = ut.robust_mean(
                    combined_flux[combined_overlap_indices[int(len(combined_overlap_indices) * 0.95):]], 1)
                std_flux_combined = ut.robust_std(
                    combined_flux[combined_overlap_indices[int(len(combined_overlap_indices) * 0.95):]], 1)
                mean_flux_current = ut.robust_mean(flux_current[np.concatenate((current_overlap_indices[int(len(
                    current_overlap_indices) * 0.96):], np.arange(current_overlap_indices[-1] + 1,
                                                                  current_overlap_indices[-1] + 1 + int(
                                                                      len(current_overlap_indices) * 0.01), 1)))], 1)
                std_flux_current = ut.robust_std(flux_current[np.concatenate((current_overlap_indices[int(len(
                    current_overlap_indices) * 0.96):], np.arange(current_overlap_indices[-1] + 1,
                                                                  current_overlap_indices[-1] + 1 + int(
                                                                      len(current_overlap_indices) * 0.01), 1)))], 1)
                print(
                    f'first the mean_flux_combined was {mean_flux_combined} and mean_flux_current is {mean_flux_current}')

                # plt.plot(combined_wave,combined_flux, label = f'idx = {idx}')

                if mean_flux_combined >= mean_flux_current:
                    print(f'entered case where mean_flux_combined >= mean_flux_current')
                    if align:
                        alignment_factor = mean_flux_combined / mean_flux_current
                    else:
                        alignment_factor = 1.0
                    flux_current *= alignment_factor
                    snr_current *= alignment_factor
                    flux_reduced_current *= alignment_factor
                else:
                    print(f'entered case where mean_flux_combined < mean_flux_current')
                    if align:
                        alignment_factor = mean_flux_current / mean_flux_combined
                    else:
                        alignment_factor = 1.0
                    combined_flux *= alignment_factor
                    combined_snr *= alignment_factor
                    combined_flux_reduced *= alignment_factor

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
                    is_finer_combined = True
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
                    is_finer_combined = False

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
                alignment_score = abs(mean_flux_combined - mean_flux_current) / np.sqrt(
                    std_flux_combined ** 2 + std_flux_current ** 2)
                alignment_score_interp = abs(mean_flux_finer - mean_flux_coarser) / np.sqrt(
                    std_flux_finer ** 2 + std_flux_coarser ** 2)

                # Combine fluxes using weighted SNR method
                weights_finer = snr_finer ** 2
                weights_coarser = interp_snr_coarser ** 2
                total_weights = weights_finer + weights_coarser

                combined_flux_overlap = (
                                                    flux_finer * weights_finer + interp_flux_coarser * weights_coarser) / total_weights
                combined_snr_overlap = np.sqrt(total_weights)
                combined_flux_reduced_overlap = (
                                                            flux_reduced_finer * weights_finer + interp_flux_reduced_coarser * weights_coarser) / total_weights

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
                        combined_flux_reduced = np.concatenate(
                            (combined_flux_reduced, flux_reduced_current[non_overlap_indices_current]))
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
                        flux_reduced_current = np.concatenate(
                            (combined_flux_reduced[non_overlap_indices_combined], flux_reduced_current))

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
                print(
                    f"Alignment factor: {alignment_factor:.4f}, Alignment score: {alignment_score:.4f} and Alignment scoreafter interpolation: {alignment_score_interp:.4f}")

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

            aligment_data_tmp = {'Alignment_factor': alignment_factor, 'Initial_Alignment_score': alignment_score,
                                 'End_Alignment_score': alignment_score_interp,
                                 'Desc': 'Alignment_factor is the factor one side of the spectra was multiplied by to make it consistent. Initial_Alignment_score is the score before the fixes. below 1 means it was already consistent, above means it wasnt. End_Alignment_score is the score after the fix, should be way closer to 0'}
            aligment_data[f'overlap_{idx}'] = aligment_data_tmp

        return combined_wave, combined_flux, combined_snr, combined_flux_reduced, aligment_data

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

    ########################################                                   ########################################

    def _local_snr(self,
                   flux: np.ndarray,
                   idx: np.ndarray,
                   half_win: int = 20,
                   ) -> np.ndarray:
        """
        Per-pixel SNR = 1 / std(window)  **only for the indices in `idx`.**
        Pixels not in `idx` should never be passed in.
        """
        snr = np.empty(idx.size, dtype=float)
        n = len(flux)
        h = half_win
        for k, i in enumerate(idx):
            lo = max(0, i - h)
            hi = min(n, i + h + 1)
            sigma = np.std(flux[lo:hi])
            snr[k] = 1.0 / sigma if sigma > 0 else np.nan
        return snr

    # ------------------------------------------------------------------
    # public method
    # ------------------------------------------------------------------

    def combine_cleaned_bands(
            self,
            wl_list: List[np.ndarray],
            flux_list: List[np.ndarray],
            trash,
            edge_frac=0.05,
            window_size=40
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        wl_list, flux_list : list of 1-D np.ndarrays, each strictly increasing.

        Returns
        -------
        wave, flux, snr : 1-D np.ndarrays
                          (snr = NaN everywhere except stitched overlaps)
        """
        if len(wl_list) != len(flux_list):
            raise ValueError("wl_list and flux_list must be the same length")

        # ---------- seed with first band ---------------------------------
        wave = wl_list[0].copy()
        flux = flux_list[0].copy()
        half_win = int(window_size / 2)
        snr = np.full_like(flux, np.nan)  # no overlap yet ⇒ NaN

        order = np.argsort(wave)
        wave, flux, snr = wave[order], flux[order], snr[order]

        # ---------- iterate over remaining bands -------------------------
        for wl, fl in zip(wl_list[1:], flux_list[1:]):
            wl = wl.copy()
            fl = fl.copy()
            sn = np.full_like(fl, np.nan)

            # overlap limits
            left, right = max(wave[0], wl[0]), min(wave[-1], wl[-1])

            # --------------- no overlap → simple append ------------------
            if left >= right:
                wave = np.concatenate((wave, wl))
                flux = np.concatenate((flux, fl))
                snr = np.concatenate((snr, sn))
                order = np.argsort(wave)
                wave, flux, snr = wave[order], flux[order], snr[order]
                continue

            # indices inside overlap
            idx_c = np.where((wave >= left) & (wave <= right))[0]
            idx_n = np.where((wl >= left) & (wl <= right))[0]

            # ---------- continuum alignment (right-edge slice) ----------
            tail_c = idx_c[-max(1, int(len(idx_c) * edge_frac)):]
            tail_n = idx_n[-max(1, int(len(idx_n) * edge_frac)):]
            scale = np.median(flux[tail_c]) / np.median(fl[tail_n])
            fl *= scale

            # ---------- SNR **only** in the overlap ---------------------
            snr_c = self._local_snr(flux, idx_c, half_win)
            snr_n = self._local_snr(fl, idx_n, half_win)
            plt.plot(wave[idx_c], snr_c, label='snr_c')
            plt.plot(wl[idx_n], snr_n, label='snr_n')
            plt.legend()
            plt.show()

            # ---------- choose finer grid in the overlap ----------------
            dlam_c = np.median(np.diff(wave[idx_c]))
            dlam_n = np.median(np.diff(wl[idx_n]))
            new_is_finer = dlam_n < dlam_c - 1e-12

            if new_is_finer:
                fine_wl = wl[idx_n]
                fine_f = fl[idx_n]
                fine_sn = snr_n
                coarse_f = np.interp(fine_wl, wave[idx_c], flux[idx_c])
                coarse_sn = np.interp(fine_wl, wave[idx_c], snr[idx_c])
                w_fine, w_coarse = fine_sn ** 2, coarse_sn ** 2
                blended_flux = (fine_f * w_fine + coarse_f * w_coarse) / (w_fine + w_coarse)
                blended_snr = np.sqrt(w_fine + w_coarse)
                fl[idx_n] = blended_flux
                sn[idx_n] = blended_snr
                # append non-overlap part of new band
                keep = wl > right
                wave = np.concatenate((wave, wl[keep]))
                flux = np.concatenate((flux, fl[keep]))
                snr = np.concatenate((snr, sn[keep]))
            else:
                fine_wl = wave[idx_c]
                fine_f = flux[idx_c]
                fine_sn = snr_c
                coarse_f = np.interp(fine_wl, wl[idx_n], fl[idx_n])
                coarse_sn = np.interp(fine_wl, wl[idx_n], sn[idx_n])
                w_fine, w_coarse = fine_sn ** 2, coarse_sn ** 2
                blended_flux = (fine_f * w_fine + coarse_f * w_coarse) / (w_fine + w_coarse)
                blended_snr = np.sqrt(w_fine + w_coarse)
                flux[idx_c] = blended_flux
                snr[idx_c] = blended_snr
                # prepend part of old combo < left
                keep = wave < left
                wl = np.concatenate((wave[keep], wl))
                fl = np.concatenate((flux[keep], fl))
                sn = np.concatenate((snr[keep], sn))
                wave, flux, snr = wl, fl, sn

            # ---------- keep global arrays sorted -----------------------
            order = np.argsort(wave)
            wave, flux, snr = wave[order], flux[order], snr[order]

        return wave, flux, snr

    def combine_cleaned_bands_old(
            self, wl_list, flux_list, snr_list=None,
            edge_frac=0.05, snr_window=41
    ):
        """
        Merge already *normalised* band spectra à la original _combine_spectra.

        Parameters
        ----------
        wl_list   : list of 1-D np.ndarray (wavelengths, Å – increasing)
        flux_list : list of 1-D np.ndarray (normalised flux, ~1)
        snr_list  : list of 1-D np.ndarray  (optional; same length as wl_list)
                    – If None, SNR is estimated as 1/rolling_std.
        edge_frac : float  (0–1)  fraction of the RIGHT-edge of the overlap
                    used for continuum matching.
        snr_window: int    window size for noise estimate if snr_list is None.

        Returns
        -------
        wave, flux, snr   : 1-D np.ndarray
        """

        # --- start with the first band ---------------------------------
        wave = wl_list[0].copy()
        flux = flux_list[0].copy()
        snr = snr_list[0].copy()

        # keep everything sorted along the way
        sort = np.argsort(wave)
        wave, flux, snr = wave[sort], flux[sort], snr[sort]

        # ---------------------------------------------------------------
        for wl, fl, sn in zip(wl_list[1:], flux_list[1:], snr_list[1:]):

            wl = wl.copy();
            fl = fl.copy();
            sn = sn.copy()
            left, right = max(wave[0], wl[0]), min(wave[-1], wl[-1])

            # ---------- NO OVERLAP → simple append ----------------------
            if left >= right:
                wave = np.concatenate((wave, wl))
                flux = np.concatenate((flux, fl))
                snr = np.concatenate((snr, sn))
                sort = np.argsort(wave)
                wave, flux, snr = wave[sort], flux[sort], snr[sort]
                continue

            # ---------- locate indices of overlap ----------------------
            idx_c = np.where((wave >= left) & (wave <= right))[0]
            idx_n = np.where((wl >= left) & (wl <= right))[0]

            # ---------- continuum alignment ----------------------------
            tail_c = idx_c[-max(1, int(len(idx_c) * edge_frac)):]
            tail_n = idx_n[-max(1, int(len(idx_n) * edge_frac)):]

            # ---------- pick finer grid in overlap ---------------------
            dlam_c = np.median(np.diff(wave[idx_c]))
            dlam_n = np.median(np.diff(wl[idx_n]))
            new_is_finer = dlam_n < dlam_c - 1e-12  # tiny epsilon

            if new_is_finer:
                fine_wl, fine_f, fine_sn = wl[idx_n], fl[idx_n], sn[idx_n]
                coarse_f = np.interp(fine_wl, wave[idx_c], flux[idx_c])
                coarse_sn = np.interp(fine_wl, wave[idx_c], snr[idx_c])
            else:
                fine_wl, fine_f, fine_sn = wave[idx_c], flux[idx_c], snr[idx_c]
                coarse_f = np.interp(fine_wl, wl[idx_n], fl[idx_n])
                coarse_sn = np.interp(fine_wl, wl[idx_n], sn[idx_n])

            # ---------- SNR² weighting blend ---------------------------
            w_fine, w_coarse = fine_sn ** 2, coarse_sn ** 2
            total_w = w_fine + w_coarse
            blended_flux = (fine_f * w_fine + coarse_f * w_coarse) / total_w
            blended_snr = np.sqrt(total_w)

            # ---------- write blend back & append non-overlap ----------
            if new_is_finer:
                fl[idx_n] = blended_flux
                sn[idx_n] = blended_snr

                # append λ > right
                keep = wl > right
                wave = np.concatenate((wave, wl[keep]))
                flux = np.concatenate((flux, fl[keep]))
                snr = np.concatenate((snr, sn[keep]))
            else:
                flux[idx_c] = blended_flux
                snr[idx_c] = blended_snr

                # prepend λ < left
                keep = wave < left
                wl = np.concatenate((wave[keep], wl))
                fl = np.concatenate((flux[keep], fl))
                sn = np.concatenate((snr[keep], sn))
                wave, flux, snr = wl, fl, sn

            sort = np.argsort(wave)
            wave, flux, snr = wave[sort], flux[sort], snr[sort]

        # ----------------------------------------------------------------
        return wave, flux, snr

    def stitch_cleaned_normalized_flux(self,
                                       epoch_nums=None,
                                       bands=('UVB', 'VIS', 'NIR'),
                                       overwrite=False,
                                       backup=True):
        """
        For each epoch (or all epochs), load each band’s pre‐computed
        'cleaned_normalized_flux', stitch them into one spectrum by
        SNR‐weighted averaging in overlap regions, and save the result
        under the 'COMBINED' band as 'cleaned_normalized_flux'.

        Parameters
        ----------
        epoch_nums : int, str, or list of those, optional
            Epoch(s) to process; defaults to all epochs.
        bands : tuple of str
            The bands to stitch (in wavelength order).
        window_size_snr : int
            Sliding‐window width (pixels) for local noise estimation.
        overwrite : bool
            If True, overwrite existing COMBINED results.
        backup : bool
            If True, create a backup before overwriting.
        """

        # 1) determine epochs
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, (list, tuple, np.ndarray)):
            epoch_nums = [epoch_nums]

        for epoch in epoch_nums:
            # 2) load each band’s cleaned_normalized_flux
            wl_list, fl_list, snr_list = [], [], []
            for b in bands:
                data = self.load_property('cleaned_normalized_flux', epoch, b)
                if (data is None
                        or 'wavelengths' not in data
                        or 'normalized_flux' not in data):
                    print(f"[Epoch {epoch}] missing cleaned_normalized_flux for band {b}; skipping.")
                    break
                wl_list.append(np.array(data['wavelengths']))
                fl_list.append(np.array(data['normalized_flux']))
                SNR = self.load_observation(epoch_num=epoch, band=b).data['SNR'][0]
                snr_list.append(np.array(SNR))
            else:
                # 4)
                flux_reduced_list = [np.ones_like(fl) for fl in fl_list]
                comb_wl, comb_flux, comb_snr, _, align_info = self._combine_spectra(
                    wl_list, fl_list, snr_list, flux_reduced_list, align=False
                )

                # 5) save to COMBINED
                out = {"wavelengths": comb_wl, "normalized_flux": comb_flux}
                self.save_property(
                    "cleaned_normalized_flux",
                    out,
                    epoch,
                    "COMBINED",
                    overwrite=overwrite,
                    backup=backup,
                )
                print(
                    f"[Epoch {epoch}] saved stitched cleaned_normalized_flux "
                    f"({len(comb_wl)} pixels) under COMBINED."
                )

            # end for-bands else
        # end for-epoch loop

    ########################################               CCF                 ########################################

    def combined_normalized_template(self, epoch_num, band=None):
        fits_file = self.load_observation(1, 'COMBINED')
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

    ########################################                cleaning data using 2D images                ########################################

    def clean_flux_and_normalize(self, epoch_num, band, bottom_spacial=None, top_spacial=None, exclude_spacial=None):
        """
        Cleans the 2D flux image, normalizes the flux, and compares with external normalized data.

        Parameters:
            epoch_num: int, the epoch number for the observation
            band: str, the band of the observation (e.g., 'VIS', 'UVB', etc.)
            bottom_spacial: int, optional
                Bottom spatial coordinate to include in the region of interest. Default is automatically detected.
            top_spacial: int, optional
                Top spatial coordinate to include in the region of interest. Default is automatically detected.
            exclude_spacial: tuple(int, int), optional
                A range of spatial coordinates to exclude from summation (e.g., contaminating star region).
        """
        # Load the 2D image
        fits_file_2D = self.load_2D_observation(epoch_num, band)
        image_data = fits_file_2D.primary_data

        # Detect spatial limits if not provided
        if bottom_spacial is None or top_spacial is None:
            if band == 'NIR':
                bottom_spacial, top_spacial = (-52, -24)
            else:
                bottom_spacial, top_spacial = (-68, -30)
        print(f"The top limit is: {top_spacial}, and the bottom limit is: {bottom_spacial}")

        # Exclude spatial coordinates if specified
        exclude_start, exclude_end = None, None
        if exclude_spacial:
            exclude_start, exclude_end = exclude_spacial
            flipped_exclude_start = image_data.shape[0] - exclude_start
            flipped_exclude_end = image_data.shape[0] - exclude_end

            # Set the flipped range to zero
            image_data[flipped_exclude_end:flipped_exclude_start, :] = 0
            print(f"Excluding spatial range (flipped) from {flipped_exclude_end} to {flipped_exclude_start}.")

        # Crop the image to the specified region
        image_data_central = image_data[bottom_spacial:top_spacial, :]
        spacial_coordinate = np.arange(0, len(image_data), 1)[bottom_spacial:top_spacial]
        print(f'spacial_coordinate: {spacial_coordinate}')

        # Sum the flux along the columns
        summed_flux = np.sum(image_data_central, axis=0)

        # Get wavelengths from the 2D data
        fits_file_1D = self.load_observation(epoch_num, band)
        wavelengths_2D = fits_file_1D.data['WAVE'][0]

        # Load anchor points for normalization
        anchor_points = self.load_property('norm_anchor_wavelengths', epoch_num, band='COMBINED')
        anchor_points_in_range = anchor_points[
            (anchor_points >= wavelengths_2D.min()) & (anchor_points <= wavelengths_2D.max())
            ]
        closest_indices = [np.abs(wavelengths_2D - ap).argmin() for ap in anchor_points_in_range]
        selected_flux = [ut.robust_mean(summed_flux[index - 10:index + 10], 1) for index in closest_indices]

        if len(selected_flux) != len(anchor_points_in_range):
            raise ValueError("Mismatch between selected_flux and anchor_points sizes.")

        # Interpolate the continuum flux
        continuum_flux_interpolated = np.interp(wavelengths_2D, anchor_points_in_range, selected_flux)

        # Normalize the summed flux
        normalized_summed_flux = summed_flux / continuum_flux_interpolated

        # Plot the 2D flux image with excluded regions
        title = f"2D Flux Image for {self.star_name} Band {band} (Epoch {epoch_num})"
        # Plot the 2D image
        p2D.Plot2DImage(
            image_data,
            wavelengths_2D,
            band,
            title=title,
            ValMin=-600,
            ValMax=600,
            see_all=True
        )

        # Add vertical lines for excluded regions on top of the 2D plot
        if exclude_spacial:
            plt.axhline(exclude_start, linestyle="dotted", color="red", label="Excluded Start")
            plt.axhline(exclude_end, linestyle="dotted", color="red", label="Excluded End")
            plt.legend()
            # plt.show()

        # Load external normalized flux for comparison (same band)
        external_data = self.load_property('normalized_flux', epoch_num, band='COMBINED')
        external_normalized_flux = external_data['normalized_flux']
        external_wavelengths = external_data['wavelengths']

        # Filter external data to match the wavelength range of the current band
        mask_band = (external_wavelengths >= wavelengths_2D.min()) & (external_wavelengths <= wavelengths_2D.max())
        external_normalized_flux_band = external_normalized_flux[mask_band]
        external_wavelengths_band = external_wavelengths[mask_band]

        # Interpolate `normalized_summed_flux` to match the resolution of `external_normalized_flux_band`
        normalized_summed_flux_resampled = np.interp(external_wavelengths_band, wavelengths_2D, normalized_summed_flux)

        # Calculate the difference and relative difference
        flux_difference = normalized_summed_flux_resampled - external_normalized_flux_band
        relative_difference = flux_difference / external_normalized_flux_band

        # Plot summed flux
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths_2D, summed_flux, label=f'Summed Flux ({band})', color='blue')
        plt.scatter(anchor_points_in_range, selected_flux, label='anchor points', color='red')
        # plt.plot(external_wavelengths_band, external_normalized_flux_band, label='External Flux', color='orange', alpha=0.7)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('summed Flux')
        plt.title(f'Summed Flux Comparison for {self.star_name} band {band} (Epoch {epoch_num})')
        plt.legend()
        plt.grid(True)
        plt.show()

        summed_flux_horizontaly = np.sum(image_data_central, axis=1)
        # Plot summed horizontaly flux
        plt.figure(figsize=(15, 9))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.plot(spacial_coordinate, summed_flux_horizontaly, label=f'Summed Flux horizontaly({band})', color='blue')
        # plt.scatter(anchor_points_in_range,selected_flux, label = 'anchor points', color = 'red')
        # plt.plot(external_wavelengths_band, external_normalized_flux_band, label='External Flux', color='orange', alpha=0.7)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('summed Flux')
        plt.title(f'Summed Flux Comparison for {self.star_name} band {band} (Epoch {epoch_num})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the normalized flux comparison
        plt.figure(figsize=(12, 6))
        plt.plot(external_wavelengths_band, normalized_summed_flux_resampled,
                 label=f'Cleaned Normalized Summed Flux ({band})', color='blue')
        plt.plot(external_wavelengths_band, external_normalized_flux_band, label='Non-Cleaned Normalized Flux',
                 color='orange', alpha=0.7)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Flux')
        plt.title(f'Normalized Flux Comparison for {self.star_name} band {band} (Epoch {epoch_num})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the differences
        plt.figure(figsize=(12, 6))
        plt.plot(external_wavelengths_band, flux_difference, label='Flux Difference', color='red')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux Difference')
        plt.title(
            f'Flux Difference (Normalized Summed Flux - External) for {self.star_name} band {band} (Epoch {epoch_num})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the relative differences
        plt.figure(figsize=(12, 6))
        plt.plot(external_wavelengths_band, relative_difference, label='Relative Difference', color='purple')
        plt.plot(external_wavelengths_band, np.ones(len(external_wavelengths_band)) / 10, linestyle='dashed',
                 color='red')
        plt.plot(external_wavelengths_band, -np.ones(len(external_wavelengths_band)) / 10, linestyle='dashed',
                 color='red')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Relative Difference')
        plt.title(
            f'Relative Flux Difference (Diff / External Flux) for {self.star_name} band {band} (Epoch {epoch_num})')
        plt.legend()
        plt.grid(True)
        plt.show()

        return normalized_summed_flux_resampled, external_wavelengths_band, (bottom_spacial, top_spacial)
