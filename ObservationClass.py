import os
import glob
import re
import datetime
import shutil
import pprint
from astropy.io import fits
import pandas as pd
import multiprocess
from IPython.display import display
from FitsClass import FITSFile as myfits 
from StarClass import Star
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
        self.star_names = specs.star_names  # List of star names loaded from the specs file
        self.data_dir = data_dir  # Base path where all the organized star data is stored
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

        if data_dir == None:
            data_dir = self.data_dir

        if backup_dir == None:
            backup_dir = self.backup_dir

        # Create and return the Star instance
        star_instance = Star(star_name=star_name, data_dir = data_dir, backup_dir = backup_dir)
        return star_instance

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
         # Check if the star exists in the list of star names
        if star_name not in self.star_names:
            print(f"Error: Star '{star_name}' is not in the list of star names in specs.py.")
            return None

        if data_dir == None:
            data_dir = self.data_dir

        if backup_dir == None:
            backup_dir = self.backup_dir
            
        try:
            return self.star_instances[star_name]
        except:
            star_instance = self.create_star_instance(star_name, data_dir = data_dir, backup_dir = backup_dir)
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
                                params={}, overwrite=False, backup=True, parallel=False):
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
