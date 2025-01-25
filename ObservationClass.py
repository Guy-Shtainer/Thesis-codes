import os
import glob
import re
import datetime
import shutil
import pprint
import numpy as np
from astropy.io import fits
from astropy.time import Time
import pandas as pd
from collections.abc import MutableMapping
from collections import defaultdict
import multiprocess
from IPython.display import display
from FitsClass import FITSFile as myfits 
from StarClass import Star
from NRESClass import NRES
import specs


class ObservationManager:
    def __init__(self,data_dir = f'Data/',backup_dir = f'Backups/', specs_filepath = f'specs.py'):
        """
        Initializes an ObservationManager instance.
        
        This class is responsible for managing and organizing the observations data for various stars. 
        It works with a dictionary that maps star names to their file paths and stores the base path where 
        the organized FITS files are located.

        Parameters:
        - data_dir (str, optional): The root directory where the organized data for stars is stored. 
          Defaults to 'Data/'.
        
        Attributes:
        - star_names (list): A list of star names loaded from the specs file.
        - data_dir (str): The root path where the organized data is stored.
        
        Example:
        obs_manager = ObservationManager(data_dir='Data/')
        """
        # Add a short list of star names that use the NRES class
        self.NRES_stars = ['WR 52', 'WR17']
        
        self.star_names = np.concatenate((specs.star_names,self.NRES_stars))  # from specs.py
        self.data_dir = data_dir
        self.specs_filepath = specs_filepath
        self.backup_dir = backup_dir
        self.star_instances = {}


    def create_star_instance(self, star_name, data_dir = None, backup_dir = None):
        """
        Creates and returns a Star instance if the star is found in the list of star names.

        Parameters:
        - star_name (str): The name of the star to create an instance for.
        - data_dir (str, optional): The root directory where the organized data is stored. If not specified than determined by ObservationManager class.
        - backup_dir (str, optional): The root directory where the backup data is stored. If not specified than determined by ObservationManager class.

        Returns:
        - Star instance: A new instance of the Star class if the star exists.
        """

        if data_dir is None:
            data_dir = self.data_dir
        if backup_dir is None:
            backup_dir = self.backup_dir

        # If the star is in the NRES list, return an NRES object
        if star_name in self.NRES_stars:
            return NRES(star_name=star_name, data_dir=data_dir, backup_dir=backup_dir)
        else:
            return Star(star_name=star_name, data_dir=data_dir, backup_dir=backup_dir)

    def load_star_instance(self, star_name, data_dir = None, backup_dir = None):
        """
        Loads and returns a Star instance for the given star name.
    
        Parameters:
        - star_name (str): The name of the star to load.
        - data_dir (str, optional): The directory containing the star's data. Defaults to None.
        - backup_dir (str, optional): The root directory where the backup data is stored. If not specified than determined by ObservationManager class.
    
        Returns:
        - Star: An instance of the Star class corresponding to the given star_name.
            - Returns None if the star is not recognized.
    
        Raises:
        - KeyError: If there is an unexpected issue accessing the star instances.
        """
        # 1) Check if star is recognized at all
        if star_name not in self.star_names:
            print(f"Error: Star '{star_name}' is not in the list of star names in specs.py.")
            return None

        # 2) Set defaults
        if data_dir is None:
            data_dir = self.data_dir
        if backup_dir is None:
            backup_dir = self.backup_dir

        # 3) If already loaded this star, just return it
        if star_name in self.star_instances:
            return self.star_instances[star_name]

        # 4) Otherwise, create it using create_star_instance
        star_instance = self.create_star_instance(star_name, data_dir=data_dir, backup_dir=backup_dir)
        self.star_instances[star_name] = star_instance
        return star_instance
    
    def clean_all_stars(self):
        """
        Invokes the 'clean' method on all stars managed by the ObservationManager.
        
        Deletes empty folders within the 'output' directories for each star.
        
        Returns:
            None
        """
        for star_name in self.star_names:
            star = self.load_star_instance(star_name)
            print(f"Cleaning star '{star_name}'...")
            star.clean()
        print("Cleaning completed for all stars.")

    def organize_star_files(self, fits_directory, output_directory = None):
        """
        Reads star names from specs.py and organizes their FITS files into folders.

        Parameters:
        - specs_filepath (str): Path to the specs.py file containing star_names.
        - fits_directory (str): Directory path where the FITS files are located.
        - output_directory (str): Directory path where the organized folders should be created.
        """
        # Step 1: Read star_names from specs.py
        import importlib.util
        star_names = specs.star_names

        print(f"Star names found: {star_names}")

        if output_directory == None:
            output_directory = self.data_dir

        # Step 2: Scan fits_directory for FITS files
        fits_files = glob.glob(os.path.join(fits_directory, '**', '*.fits'), recursive=True)

        print(f"Found {len(fits_files)} FITS files in {fits_directory}")

        # Step 3: Organize files for each star
        for star_name in star_names:
            print(f"Organizing files for star: {star_name}")

            # Create star directory in output_directory
            star_dir = os.path.join(output_directory, star_name)
            os.makedirs(star_dir, exist_ok=True)

            # Filter FITS files for this star
            star_fits_files = []

            for filepath in fits_files:
                try:
                    with fits.open(filepath) as hdulist:
                        header = hdulist[0].header
                        file_star_name = header.get('OBJECT')
                        if file_star_name.strip() == star_name:
                            star_fits_files.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

            print(f"Found {len(star_fits_files)} FITS files for {star_name}")

            # Group files by date (to define epochs)
            files_by_date = {}
            for filepath in star_fits_files:
                try:
                    with fits.open(filepath) as hdulist:
                        header = hdulist[0].header
                        # Extract classification information
                        date_obs = header.get('DATE-OBS')
                        dispelem = header.get('DISPELEM')
                        if not all([date_obs, dispelem]):
                            print(f"Skipping {filepath}: Missing DATE-OBS or DISPELEM")
                            continue

                        # Convert date_obs to date string (YYYY-MM-DD)
                        date_str = date_obs.split('T')[0]
                        # Group files by date
                        if date_str not in files_by_date:
                            files_by_date[date_str] = []
                        files_by_date[date_str].append(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

            # Assign epoch numbers based on sorted dates
            sorted_dates = sorted(files_by_date.keys())
            date_to_epoch = {date: f'epoch{idx + 1}' for idx, date in enumerate(sorted_dates)}

            # Process each epoch
            for date_str, files in files_by_date.items():
                epoch = date_to_epoch[date_str]
                print(f"Processing {len(files)} files for {star_name} on {date_str} ({epoch})")

                # Create epoch directory
                epoch_dir = os.path.join(star_dir, epoch)
                os.makedirs(epoch_dir, exist_ok=True)

                # Process each FITS file
                for filepath in files:
                    try:
                        with fits.open(filepath) as hdulist:
                            header = hdulist[0].header
                            date_obs = header.get('DATE-OBS')
                            if not date_obs:
                                print(f"Skipping {filepath}: Missing DATE-OBS")
                                continue

                            # Extract observation time from DATE-OBS
                            date_obs_time = date_obs.strip()

                            # Extract PROV# filenames and their timestamps
                            prov_times = {}
                            for i in range(1, 10):
                                prov_key = f'PROV{i}'
                                prov_filename = header.get(prov_key)
                                if not prov_filename:
                                    break
                                # Extract timestamp from prov_filename
                                # Assuming filenames like 'XSHOO.YYYY-MM-DDThh:mm:ss.sss.fits'
                                prov_basename = os.path.basename(prov_filename)
                                prov_parts = prov_basename.split('.')
                                if len(prov_parts) >= 2:
                                    prov_timestamp = prov_parts[1]  # 'YYYY-MM-DDThh:mm:ss.sss'
                                    prov_times[prov_timestamp] = i  # Store sub-exposure number

                            # Determine sub-exposure number by matching times
                            sub_exp_num = None
                            for prov_timestamp, sub_num in prov_times.items():
                                if prov_timestamp == date_obs_time:
                                    sub_exp_num = sub_num
                                    break

                            if not sub_exp_num:
                                print(f"Sub-exposure not found for {filepath}. Assigning to sub-exposure 1.")
                                sub_exp_num = 1

                            # Get band information
                            band = header.get('DISPELEM', 'Unknown').strip()

                            # Create directory structure
                            # subexp_dir = os.path.join(epoch_dir, f'sub-exposure {sub_exp_num}')
                            # band_dir = os.path.join(subexp_dir, band)
                            band_dir = os.path.join(epoch_dir, band)
                            os.makedirs(band_dir, exist_ok=True)

                            # Copy the FITS file to the band directory
                            dest_filepath = os.path.join(band_dir, os.path.basename(filepath))
                            shutil.copy2(filepath, dest_filepath)
                            print(f"Copied {filepath} to {dest_filepath}")

                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

    def organize_nres_files(self, fits_directory, output_directory=None):
        """
        Example method for organizing .fits.fz files.
    
        Using astropy.time.Time for date parsing to avoid Python version issues with fromisoformat().
        """
    
        if output_directory is None:
            output_directory = self.data_dir
    
        fits_files = glob.glob(os.path.join(fits_directory, '**', '*.fits.fz'), recursive=True)
        print(f"Found {len(fits_files)} .fits.fz files in {fits_directory}")
    
        if not fits_files:
            print("No .fits.fz files found. Nothing to organize.")
            return
    
        # Collect metadata
        files_metadata = []
        for fpath in fits_files:
            try:
                with fits.open(fpath) as hdul:

                    # Derive file_type from filename
                    header_num = 1
                    basename_lower = os.path.basename(fpath).lower()
                    if '1d' in basename_lower:
                        file_type = '1D'
                        header_num = 0
                    elif '2d' in basename_lower:
                        file_type = '2D'
                    elif 'e00' in basename_lower:
                        file_type = 'raw'
                    else:
                        file_type = 'unknown'
                    
                    header = hdul[header_num].header
                    star_name = header.get('OBJECT', 'Unknown').strip()
    
                    date_obs_str = header.get('DATE-OBS')
                    if not date_obs_str:
                        print(f"Skipping {fpath}: Missing DATE-OBS in header.")
                        continue
    
                    # Use astropy's Time to parse
                    try:
                        t = Time(date_obs_str, format='isot')  # or 'fits'
                        date_obs = t.to_datetime()
                    except ValueError:
                        print(f"Skipping {fpath}: Could not parse DATE-OBS '{date_obs_str}'")
                        continue
    
                    
    
                    parent_dir = os.path.abspath(os.path.dirname(fpath))
    
                    files_metadata.append({
                        'star': star_name,
                        'date_obs': date_obs,
                        'file_type': file_type,
                        'parent_dir': parent_dir,
                        'full_path': fpath,
                    })
    
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue
    
        # Group by star + file_type
        grouped = defaultdict(lambda: defaultdict(list))
        for fd in files_metadata:
            grouped[fd['star']][fd['file_type']].append(fd)
    
        # Organize files
        for star_name, type_dict in grouped.items():
            print(f"\nOrganizing files for star: {star_name}")
    
            for file_type, file_list in type_dict.items():
                # Sort by date_obs ascending
                file_list.sort(key=lambda d: d['date_obs'])
    
                # Base dir
                base_dir = os.path.join(output_directory, star_name, file_type)
                os.makedirs(base_dir, exist_ok=True)
    
                epoch_counter = 1
                epoch_folder = os.path.join(base_dir, f"epoch{epoch_counter}")
                os.makedirs(epoch_folder, exist_ok=True)
    
                prev_date = None
                prev_parent = None
                file_index_in_epoch = 0
    
                for fdict in file_list:
                    current_date = fdict['date_obs']
                    current_parent = fdict['parent_dir']
    
                    if prev_date is not None:
                        # Compare times
                        date_diff_days = abs((current_date - prev_date).days)
                        folder_changed = (current_parent != prev_parent)
                        if date_diff_days > 1 or folder_changed:
                            epoch_counter += 1
                            epoch_folder = os.path.join(base_dir, f"epoch{epoch_counter}")
                            os.makedirs(epoch_folder, exist_ok=True)
                            file_index_in_epoch = 0
    
                    file_index_in_epoch += 1
                    new_filename = f"spectra{file_index_in_epoch}.fits.fz"
                    dest_path = os.path.join(epoch_folder, new_filename)
    
                    # Copy the file
                    try:
                        shutil.copy2(fdict['full_path'], dest_path)
                        print(f"  Copied -> {dest_path} (DATE-OBS={current_date}, epoch={epoch_counter})")
                    except Exception as e:
                        print(f"  Error copying {fdict['full_path']}: {e}")
    
                    prev_date = current_date
                    prev_parent = current_parent
    
        print("\nNRES files organized successfully!")

    def organize_nres_files2(self, rawdata_directory, output_directory=None):
        """
        Organize NRES .fits.fz files into:
        
          output_directory/
            star_name/
              epoch1/
                1/
                  1D/
                    file1.fits.fz
                  2D/
                    file2.fits.fz
                  raw/
                    file3.fits.fz
                2/
                  1D/
                  2D/
                  raw/
              epoch2/
                ...
        
        Steps:
          1) Parse each .fits.fz to get (star_name, date_obs, data_type).
          2) Group by (star_name, rounded_date_obs) so that each group 
             can hold up to 3 data_types (1D, 2D, raw).
          3) Sort these groups by date_obs ascending.
          4) Increment 'epoch' whenever gap > 1 day from the previous group (in the same star).
          5) Within an epoch, each group is assigned a 'spectrum' index. 
             Under that spectrum, create subfolders named 1D/2D/raw, 
             putting the corresponding file(s) in each subfolder.
        """
    
        if output_directory is None:
            output_directory = self.data_dir
    
        # 1) Find all .fits.fz files
        all_fits = glob.glob(os.path.join(rawdata_directory, '**', '*.fits.fz'), recursive=True)
        if not all_fits:
            print("No .fits.fz files found under rawdata_directory. Nothing to organize.")
            return
    
        print(f"Found {len(all_fits)} .fits.fz files in {rawdata_directory}.")
    
        # 2) For each file, read star_name, date_obs, data_type
        file_list = []
        for fpath in all_fits:
            basename = os.path.basename(fpath).strip().lower()
    
            # Infer data_type from filename
            header_num = 1
            if '1d' in basename:
                data_type = '1D'
                header_num = 0
            elif '2d' in basename:
                data_type = '2D'
            elif 'e00' in basename:
                data_type = 'raw'
            else:
                data_type = 'unknown'  # or skip if you prefer
    
            # Read star_name & date_obs from header
            star_name = 'Unknown'
            date_obs  = None
            try:
                with fits.open(fpath) as hdul:
                    hdr = hdul[header_num].header
                    star_name = hdr.get('OBJECT', 'Unknown').strip()
    
                    date_str = hdr.get('DATE-OBS')
                    if date_str:
                        t = Time(date_str, format='isot')  # robust parse
                        date_obs = t.to_datetime()
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue
    
            if not date_obs:
                print(f"No valid DATE-OBS in {fpath}. Skipping.")
                continue
    
            file_list.append({
                'path':       fpath,
                'basename':   os.path.basename(fpath),
                'star':       star_name,
                'date_obs':   date_obs,
                'data_type':  data_type
            })
    
        if not file_list:
            print("No valid NRES .fits.fz files with DATE-OBS found.")
            return
    
        # 3) Group by (star_name, "observation time" to nearest second).
        #    We'll treat anything with the exact same second as the same "spectrum".
        #    If you need more tolerant grouping (e.g., ±2s), adapt below.
        
        # star_obs[star_name] = dictionary:
        #   key = date_obs_rounded (datetime, truncated or rounded to second)
        #   val = list of dict(...) for each file that has that date
        star_obs = defaultdict(lambda: defaultdict(list))
    
        for info in file_list:
            # round to nearest second
            # or use e.g.:   date_obs_rounded = info['date_obs'].replace(microsecond=0)
            # or a custom round function to handle weird microseconds
            dt = info['date_obs']
            date_obs_rounded = dt.replace(microsecond=0)
            
            star_obs[info['star']][date_obs_rounded].append(info)
    
        # Now each star has a dictionary { date_rounded -> [files with 1D/2D/raw] }
    
        # 4) For each star, we want to sort by the actual date and then define epochs.
        for star_name, date_dict in star_obs.items():
            # Make a sorted list of (date_rounded, [files]) by the actual date_rounded ascending
            sorted_times = sorted(date_dict.keys())
            
            # We'll store them as a list of (datetime, list_of_files, real_min_date_for_sorting)
            # Because some 1D/2D might have slightly different microseconds, if you want 
            # the earliest or average among them. For now, we just use the "rounded" as is.
            grouped_observations = []
            for dt_rounded in sorted_times:
                # Among the files in date_dict[dt_rounded], we might have slightly different microseconds
                # let's get the earliest real date to sort them precisely
                real_min_date = min(x['date_obs'] for x in date_dict[dt_rounded])
                grouped_observations.append((dt_rounded, date_dict[dt_rounded], real_min_date))
    
            # Sort by real_min_date just in case there's any sub-second difference
            grouped_observations.sort(key=lambda x: x[2])
    
            # Build star_dir
            star_dir = os.path.join(output_directory, star_name)
            os.makedirs(star_dir, exist_ok=True)
    
            epoch_counter = 1
            epoch_dir = os.path.join(star_dir, f"epoch{epoch_counter}")
            os.makedirs(epoch_dir, exist_ok=True)
    
            prev_date = None
            # keep track of spectra index within the epoch
            spectrum_index = 0
    
            for idx, (dt_rounded, files_in_this_obs, real_date) in enumerate(grouped_observations):
                if prev_date is not None:
                    gap_days = (real_date - prev_date).days
                    # If gap > 1 day => new epoch
                    if abs(gap_days) > 1.0:
                        epoch_counter += 1
                        epoch_dir = os.path.join(star_dir, f"epoch{epoch_counter}")
                        os.makedirs(epoch_dir, exist_ok=True)
                        spectrum_index = 0
    
                spectrum_index += 1
    
                # So: epochN/spectrum_index, and inside that folder -> 1D/2D/raw as needed
                spec_dir = os.path.join(epoch_dir, str(spectrum_index))
                os.makedirs(spec_dir, exist_ok=True)
    
                # Now place each file in [files_in_this_obs] into the appropriate data_type subfolder
                for fdict in files_in_this_obs:
                    dt_folder = fdict['data_type']
                    dt_path = os.path.join(spec_dir, dt_folder)
                    os.makedirs(dt_path, exist_ok=True)
    
                    src = fdict['path']
                    dst = os.path.join(dt_path, fdict['basename'])
    
                    try:
                        shutil.copy2(src, dst)
                        print(f"Copied '{src}' => '{dst}' [star={star_name}, epoch={epoch_counter}, spec={spectrum_index}]")
                    except Exception as e:
                        print(f"Error copying {src} to {dst}: {e}")
    
                prev_date = real_date
    
        print("\nNRES files organized successfully!")

    
    def organize_star_2D_images(self, fits_directory, output_directory=None):
        """
        Reads star names from specs.py and organizes 2D spectral image FITS files into folders.
        Creates a "2D image" folder within the star/epoch/band structure to store the 2D image files.
    
        Parameters:
        - fits_directory (str): Directory path where the FITS files are located.
        - output_directory (str): Directory path where the organized folders should be created.
        """
        import importlib.util
        star_names = specs.star_names
    
        print(f"Star names found: {star_names}")
    
        if output_directory is None:
            output_directory = self.data_dir
    
        # Step 2: Scan fits_directory for FITS files
        fits_files = glob.glob(os.path.join(fits_directory, '**', '*.fits'), recursive=True)
        print(f"Found {len(fits_files)} FITS files in {fits_directory}")
    
        # Step 3: Organize files for each star
        for star_name in star_names:
            print(f"Organizing files for star: {star_name}")
    
            # Create star directory in output_directory
            star_dir = os.path.join(output_directory, star_name)
            os.makedirs(star_dir, exist_ok=True)
    
            # Filter FITS files for this star
            star_fits_files = []
    
            for filepath in fits_files:
                try:
                    with fits.open(filepath) as hdulist:
                        header = hdulist[0].header
                        file_star_name = header.get('OBJECT')
                        if file_star_name.strip() == star_name:
                            star_fits_files.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
            print(f"Found {len(star_fits_files)} FITS files for {star_name}")
    
            # Group files by date (to define epochs)
            files_by_date = {}
            for filepath in star_fits_files:
                try:
                    with fits.open(filepath) as hdulist:
                        header = hdulist[0].header
                        date_obs = header.get('DATE-OBS')
                        dispelem = header.get('ESO SEQ ARM')
                        if not all([date_obs, dispelem]):
                            print(f"Skipping {filepath}: Missing DATE-OBS or DISPELEM")
                            continue
    
                        # Convert date_obs to date string (YYYY-MM-DD)
                        date_str = date_obs.split('T')[0]
                        # Group files by date
                        if date_str not in files_by_date:
                            files_by_date[date_str] = []
                        files_by_date[date_str].append(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
            # Assign epoch numbers based on sorted dates
            sorted_dates = sorted(files_by_date.keys())
            date_to_epoch = {date: f'epoch{idx + 1}' for idx, date in enumerate(sorted_dates)}
    
            # Process each epoch
            for date_str, files in files_by_date.items():
                epoch = date_to_epoch[date_str]
                print(f"Processing {len(files)} files for {star_name} on {date_str} ({epoch})")
    
                # Create epoch directory
                epoch_dir = os.path.join(star_dir, epoch)
                os.makedirs(epoch_dir, exist_ok=True)
    
                # Process each FITS file
                for filepath in files:
                    try:
                        with fits.open(filepath) as hdulist:
                            header = hdulist[0].header
                            date_obs = header.get('DATE-OBS')
                            if not date_obs:
                                print(f"Skipping {filepath}: Missing DATE-OBS")
                                continue
    
                            # Extract observation time from DATE-OBS
                            date_obs_time = date_obs.strip()
    
                            # Extract PROV# filenames and their timestamps
                            prov_times = {}
                            for i in range(1, 10):
                                prov_key = f'PROV{i}'
                                prov_filename = header.get(prov_key)
                                if not prov_filename:
                                    break
                                prov_basename = os.path.basename(prov_filename)
                                prov_parts = prov_basename.split('.')
                                if len(prov_parts) >= 2:
                                    prov_timestamp = prov_parts[1]
                                    prov_times[prov_timestamp] = i  # Store sub-exposure number
    
                            # Determine sub-exposure number by matching times
                            sub_exp_num = None
                            for prov_timestamp, sub_num in prov_times.items():
                                if prov_timestamp == date_obs_time:
                                    sub_exp_num = sub_num
                                    break
                            if not sub_exp_num:
                                print(f"Sub-exposure not found for {filepath}. Assigning to sub-exposure 1.")
                                sub_exp_num = 1
    
                            # Get band information
                            band = header.get('ESO SEQ ARM', 'Unknown').strip()
    
                            # Create directory structure with "2D image" folder
                            image_dir = os.path.join(epoch_dir, band, '2D image')
                            os.makedirs(image_dir, exist_ok=True)
    
                            # Copy the FITS file to the "2D image" directory
                            dest_filepath = os.path.join(image_dir, os.path.basename(filepath))
                            shutil.copy2(filepath, dest_filepath)
                            print(f"Copied {filepath} to {dest_filepath}")
    
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")


    def get_star_names_from_fits(self, search_path):
        """
        Scans the given directory recursively for FITS files,
        extracts the star names from their headers, and returns a list of unique star names.
        
        Parameters:
        - search_path (str): The root directory to start the recursive search.
        
        Returns:
        - List[str]: A list of unique star names found in the FITS files.
        """
        
        fits_files = glob.glob(os.path.join(search_path, '**', '*.fits'), recursive=True)
        star_names = set()
        for filepath in fits_files:
            try:
                with fits.open(filepath) as hdulist:
                    header = hdulist[0].header
                    star_name = header.get('OBJECT')
                    if star_name:
                        star_names.add(star_name)
                    else:
                        print(f"No 'STARNAME' found in header of {filepath}")
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        return list(star_names)

    def get_observation_dict(self, organized_data_path = None):
        """
        Creates a nested dictionary representing the observation data hierarchy.

        The dictionary structure is:
        {
        'star_name': {
            'epoch#': {
                'BandName': 'filename'
                }
            }
        }

        Parameters:
        - organized_data_path (str): The root directory containing the organized data.

        Returns:
        - observation_dict (dict): The nested dictionary of observations.
        """
        observation_dict = {}

        if organized_data_path == None:
            organized_data_path = self.data_dir

        # fits_files = glob.glob(os.path.join(fits_directory, '**', '*.fits'), recursive=True)
        
        # Traverse the directory tree
        for root, dirs, files in os.walk(organized_data_path):
            for file in files:
                if file.endswith('.fits'):
                    # Get the relative path components
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, organized_data_path)
                    path_components = rel_path.split(os.sep)
    
                    # Skip files not in the expected directory structure
                    if len(path_components) < 3:
                        continue
    
                    # Extract information based on the number of path components
                    if len(path_components) == 4:
                        # Structure: star_name / epoch# / BandName / filename
                        star_name, epoch, band_name, filename = path_components
                    else:
                        # Skip files not matching expected patterns
                        continue
    
                    # Initialize nested dictionaries
                    if star_name not in observation_dict:
                        observation_dict[star_name] = {}
                    if epoch not in observation_dict[star_name]:
                        observation_dict[star_name][epoch] = {}
                    if band_name not in observation_dict[star_name][epoch]:
                        # Assign the filename directly
                        observation_dict[star_name][epoch][band_name] = filename
                    else:
                        # If there's already a filename, warn about duplicate files
                        print(f"Warning: Duplicate file found for {star_name} / {epoch} / {band_name}")

        return observation_dict


    def create_observation_table_for_stars(self, stars=None, epoch_list=None, band_list=None, attributes_list=None):
        """
        Creates an observation table for a list of stars. If no list of stars is provided, it iterates through all stars.
    
        Parameters:
        - stars (list, optional): List of star names to create observation tables for. If None, iterates over all stars.
        - sub_exposure_list (list, optional): List of sub-exposure numbers to filter by.
        - band_list (list, optional): List of bands to filter by.
        - attributes_list (list, optional): List of additional FITS file attributes to extract.
    
        Returns:
        - pd.DataFrame: A Pandas DataFrame containing the observation data for the selected stars.
        """
        all_data = []
    
        # If no stars are provided, use all stars from specs.py
        if stars is None:
            stars = self.star_names
    
        # Loop through each star
        for star_name in stars:
            star_instance = self.create_star_instance(star_name)
            if star_instance:
                # Call the scan_fits_files method of Star class with the specified filters
                star_data = star_instance.create_observation_table(
                    epoch_list=epoch_list,
                    band_list=band_list,
                    attributes_list=attributes_list,
                    print_table=False  # Disable printing here to aggregate data
                )
                all_data.append(star_data)
    
        # Combine all data into a single DataFrame
        if all_data:
            final_table = pd.concat(all_data, ignore_index=True)
            return final_table
        else:
            print("No data collected from stars.")
            return pd.DataFrame()  # Return an empty DataFrame if no data was found

    def create_property_table_for_stars(self, property_name, stars=None, epoch_nums=None, band_list=None, to_print=False):
        """
        Creates a table of a specified property for a list of stars and epochs.
    
        Parameters:
        - property_name (str): The name of the property to retrieve.
        - stars (list, optional): List of star names. If None, iterates over all stars.
        - epoch_nums (list, optional): List of epoch numbers to filter by.
        - band_list (list, optional): List of bands to filter by.
        - to_print (bool, optional): Whether to print additional information.
    
        Returns:
        - pd.DataFrame: A Pandas DataFrame containing the property data for the selected stars.
        """
    
        all_data = []
    
        # If no stars are provided, use all stars
        if stars is None:
            stars = self.star_names
    
        epoch_nums_was_none = False
        if epoch_nums is None:
            epoch_nums_was_none = True
    
        bands_was_none = False
        if band_list is None:
            bands_was_none = True
    
        # Loop through each star
        for star_name in stars:
            star = self.create_star_instance(star_name)
            if star:
                # Get list of epochs
                if epoch_nums_was_none:
                    epoch_nums = [int(epoch[-1]) for epoch in specs.obs_file_names[star_name].keys()]
    
                # Loop through each epoch
                for epoch_num in epoch_nums:
                    # Get list of bands
                    if bands_was_none:
                        bands = [band for band in specs.obs_file_names[star_name][f'epoch{epoch_num}'].keys()]
                    try:
                        for band in bands:
                            # Load the property
                            property_data = star.load_property(property_name, epoch_num=epoch_num, band=band, to_print=to_print)
                            if property_data is not None:
                                # Flatten the property data if it's a dictionary
                                if isinstance(property_data, MutableMapping):
                                    property_data_flat = self._flatten_dict(property_data)
                                else:
                                    property_data_flat = { (property_name,): property_data }
    
                                # Create a DataFrame row with tuple keys for MultiIndex
                                row = {
                                    ('Star',): star_name,
                                    ('Epoch',): epoch_num,
                                    ('Band',): band,
                                    **property_data_flat
                                }
                                all_data.append(row)
                    except Exception as e:
                        star.print(f'Error with epoch {epoch_num}, band {band} for star {star_name}: {e}', to_print)
    
        # Create DataFrame from collected data
        if all_data:
            final_table = pd.DataFrame(all_data)
    
            # Ensure all columns are tuples for MultiIndex
            final_table.columns = pd.MultiIndex.from_tuples(final_table.columns)
    
            # Optionally sort the MultiIndex columns for better readability
            final_table = final_table.sort_index(axis=1)
    
            return final_table
        else:
            print("No data collected for the specified property.")
            return pd.DataFrame()
    
    def _flatten_dict(self, d, parent_key=()):
        """
        Flattens a nested dictionary into a dictionary with tuple keys for MultiIndex.
    
        Parameters:
        - d (dict): The dictionary to flatten.
        - parent_key (tuple, optional): The base key tuple for recursion.
    
        Returns:
        - dict: A flattened dictionary with tuple keys.
        """
        items = {}
        for k, v in d.items():
            new_key = parent_key + (str(k),)
            if isinstance(v, MutableMapping):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items


    

    def process_star(self, args):
        """
        Function to process a star in parallel.
    
        Parameters:
            args (tuple): A tuple containing the arguments needed to process the star.
    
        Returns:
            tuple: A tuple containing the star name and the result of the method execution.
        """
        (star_name, data_dir, backup_dir, method_name, params, epoch_numbers, bands, overwrite, backup, parallel) = args
        try:
            # Re-instantiate the Star object within the process
            star = Star(star_name, data_dir, backup_dir)
            # Ensure the star has the specified method
            if hasattr(star, method_name):
                star_method = getattr(star, method_name)
                result = star.execute_method(
                    method=star_method,
                    params=params,
                    epoch_numbers=epoch_numbers,
                    bands=bands,
                    overwrite=overwrite,
                    backup=backup,
                    parallel=parallel  # Allow internal parallelism
                )
                print(f"Executed method '{method_name}' on star '{star_name}'.")
                return (star_name, result)
            else:
                print(f"Star '{star_name}' does not have method '{method_name}'. Skipping.")
                return (star_name, None)
        except Exception as e:
            print(f"Error executing method '{method_name}' on star '{star_name}': {e}")
            return (star_name, None)
    
    def execute_method_on_stars(self, method_name, stars=None, epoch_numbers=None, bands=None,
                                params={}, overwrite=False, backup=True, parallel=False, max_workers = None):
        """
        Executes a specified method on multiple stars using star.execute_method().

        Parameters:
            method_name (str): The name of the method from the Star class to execute.
            stars (list or str, optional): List of star names or a single star name to process.
                                           If None, all stars are processed. Defaults to None.
            epoch_numbers (list, optional): List of epoch numbers to process. Defaults to None.
            bands (list, optional): List of bands to process. Defaults to None.
            params (dict): Parameters to pass to the method.
            overwrite (bool): Whether to overwrite existing outputs. Defaults to False.
            backup (bool): Whether to backup existing outputs before overwriting. Defaults to True.
            parallel (bool): Whether to execute methods in parallel across stars. Defaults to False.

        Returns:
            dict: A dictionary mapping star names to the results of the method execution.
        """
        # If stars is None, use all stars
        if stars is None:
            stars_to_process = self.star_names
        elif isinstance(stars, str): # If stars is a string, convert it to a list
                stars_to_process = [stars]
        else:
            stars_to_process = stars
        # # Filter the stars list to include only the specified stars
        # stars_to_process = [star for star in self.star_names if star.star_name in stars]

        # Prepare the results dictionary
        results = {}

        if parallel:
            # Number of processes is the number of stars to process
            if max_workers == None:
                max_workers = multiprocess.cpu_count() - 1
            # Prepare arguments for each star
            args_list = []
            for star_name in stars_to_process:
                args = (
                    star_name,
                    self.data_dir,
                    self.backup_dir,
                    method_name,
                    params,
                    epoch_numbers,
                    bands,
                    overwrite,
                    backup,
                    parallel  # Allow internal parallelism
                )
                args_list.append(args)

            # Use multiprocessing Pool
            with multiprocess.Pool(processes=max_workers) as pool:
                # Map the function to the arguments
                results_list = pool.map(self.process_star, args_list)

            # Collect results
            for star_name, result in results_list:
                results[star_name] = result
        else:
            for star_name in stars_to_process:
                star = Star(star_name, self.data_dir, self.backup_dir)
                if hasattr(star, method_name):
                    star_method = getattr(star, method_name)
                    try:
                        result = star.execute_method(
                            method=star_method,
                            params=params,
                            epoch_numbers=epoch_numbers,
                            bands=bands,
                            overwrite=overwrite,
                            backup=backup,
                            parallel=parallel  # Use the same 'parallel' parameter
                        )
                        results[star.star_name] = result
                        print(f"Executed method '{method_name}' on star '{star.star_name}'.")
                    except Exception as e:
                        print(f"Error executing method '{method_name}' on star '{star.star_name}': {e}")
                        results[star.star_name] = None
                else:
                    print(f"Star '{star.star_name}' does not have method '{method_name}'. Skipping.")
                    results[star.star_name] = None

        return results
        
    def update_specs_file(self, variable_name, new_value, specs_filepath = None):
        """
        Updates a variable in the specs.py file with a new value,
        and writes a comment before it indicating the previous value.

        Parameters:
        - specs_filepath (str): The path to the specs.py file.
        - variable_name (str): The name of the variable to update.
        - new_value (Any): The new value to assign to the variable.
        """

        if specs_filepath == None:
            specs_filepath = self.specs_filepath
        
        # Get the directory of specs.py
        specs_dir = os.path.dirname(specs_filepath)

        # Path to the Backup folder
        backup_folder = os.path.join(specs_dir, 'Backups')
        os.makedirs(backup_folder, exist_ok=True)  # Ensure Backup folder exists

        # Determine the next backup filename
        # List existing backups in the Backup folder
        backup_files = os.listdir(backup_folder)
        backup_numbers = []

        # Pattern to match backup filenames, e.g., 'specs_1.bak'
        backup_pattern = re.compile(r'specs_(\d+)\.bak')

        for filename in backup_files:
            match = backup_pattern.match(filename)
            if match:
                backup_numbers.append(int(match.group(1)))

        if backup_numbers:
            next_number = max(backup_numbers) + 1
        else:
            next_number = 1

        backup_filename = f'specs_{next_number}.bak'
        backup_filepath = os.path.join(backup_folder, backup_filename)

        # Copy specs.py to the backup filepath
        shutil.copy2(specs_filepath, backup_filepath)
        print(f"Backup created: {backup_filepath}")

        # Read the contents of specs.py
        with open(specs_filepath, 'r') as file:
            lines = file.readlines()

        # Prepare the new variable assignment
        new_value_str = pprint.pformat(new_value, indent=0, width=80, compact=False)

        variable_found = False
        updated_lines = []
        idx = 0
        n = len(lines)

        # Regular expression to match the start of the variable assignment
        start_pattern = re.compile(rf'^(\s*){variable_name}\s*=\s*(.*)')

        while idx < n:
            line = lines[idx]
            start_match = start_pattern.match(line)

            if start_match and not variable_found:
                variable_found = True
                indent = start_match.group(1)
                old_value_lines = [line.rstrip('\n')]

                # Collect all lines belonging to the variable assignment
                open_brackets = 0
                in_string = False
                triple_quote = False
                line_content = line.split('=', 1)[1]
                code = line_content

                # Update open brackets count for the first line
                open_brackets += line_content.count('(') - line_content.count(')')
                open_brackets += line_content.count('[') - line_content.count(']')
                open_brackets += line_content.count('{') - line_content.count('}')

                idx += 1  # Move to the next line

                # Continue collecting lines until the assignment ends
                while idx < n and (open_brackets > 0 or lines[idx - 1].rstrip('\n').endswith('\\')):
                    next_line = lines[idx]
                    code += next_line
                    old_value_lines.append(next_line.rstrip('\n'))

                    # Update open brackets count
                    open_brackets += next_line.count('(') - next_line.count(')')
                    open_brackets += next_line.count('[') - next_line.count(']')
                    open_brackets += next_line.count('{') - next_line.count('}')
                    idx += 1  # Move to the next line

                # Prepare the comment line with the old value
                # old_value = '\n# '.join(old_value_lines)
                old_value = ' '.join(old_value_lines)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                comment_line = f"# Previous value of {variable_name} at {timestamp}: # {old_value}\n"

                # Add the comment and new variable assignment to updated_lines
                updated_lines[-1] = comment_line
                new_assignment_line = f"{variable_name} = {new_value_str}\n"
                updated_lines.append(new_assignment_line)
            else:
                # Keep the current line
                updated_lines.append(line)
                idx += 1  # Move to the next line

        if not variable_found:
            print(f"Variable {variable_name} not found in {specs_filepath}")
            # Optionally, append the variable at the end
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            updated_lines.append(f"\n# Added {variable_name} on {timestamp}\n")
            new_assignment_line = f"{variable_name} = {new_value_str}\n"
            updated_lines.append(new_assignment_line)

        # Write the updated contents back to specs.py
        with open(specs_filepath, 'w') as file:
            file.writelines(updated_lines)
        print(f"{specs_filepath} updated successfully.")



    # Additional methods as needed (e.g., gathering summary statistics)
