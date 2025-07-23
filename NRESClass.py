import os
import shutil
import glob
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
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

# Tomer's tools
from CCF import CCFclass
import plot as p
import TwoDImage as p2D

# My tools
from FitsClass import FITSFile as myfits

import requests
from bs4 import BeautifulSoup
import re

class NRES:
    def __init__(self, star_name, data_dir, backup_dir,to_print = True):
        """
        Initialize the NRES object.
        
        New data structure on disk is assumed:
            star_name/
                epoch{N}/
                    {spectra_num}/
                        {data_type}/
                            original_filename.fits.fz
        
        No reliance on specs.py or observation_dict. 
        """
        self.star_name = star_name
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        
        # Some internal caches (optional)
        self.observations = {}         # If you want to cache loaded FITSFile objects
        self.normalized_spectra = {}   # If you store normalized data

        # Optional SIMBAD usage
        self.BAT_ID = self.get_bat_id()

        # The rest of these arrays in the original code:
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

    ########################################  Directory Helpers  ########################################

    def _epoch_path(self, epoch_num):
        """
        e.g. data_dir / star_name / epoch{epoch_num}
        """
        return os.path.join(self.data_dir, self.star_name, f"epoch{epoch_num}")

    def _spectra_path(self, epoch_num, spectra_num):
        """
        e.g. data_dir / star_name / epoch{epoch_num} / {spectra_num}
        """
        return os.path.join(self._epoch_path(epoch_num), str(spectra_num))

    def _data_type_path(self, epoch_num, spectra_num, data_type):
        """
        e.g. data_dir / star_name / epoch{epoch_num} / {spectra_num} / data_type
        """
        return os.path.join(self._spectra_path(epoch_num, spectra_num), data_type)

    ########################################  File Handling  ########################################

    def get_file_path(self, epoch_num, spectra_num, data_type='1D'):
        """
        Finds the .fits.fz file for the given (epoch, spectra_num, data_type).
        Defaults to data_type='1D' if not specified.
        
        star_name/epoch{N}/{spectra_num}/{data_type}/some_file.fits.fz

        Returns the full path or None if not found.
        If multiple .fits.fz exist, returns the first one found (arbitrary).
        """
        dt_path = self._data_type_path(epoch_num, spectra_num, data_type)
        if not os.path.isdir(dt_path):
            self.print(f"No folder for epoch={epoch_num}, spectra={spectra_num}, data_type={data_type}")
            return None
        
        # Look for .fits.fz file
        fits_candidates = glob.glob(os.path.join(dt_path, '*.fits.fz'))
        if not fits_candidates:
            self.print(f"No .fits.fz in {dt_path}")
            return None
        
        # Just return the first one
        return fits_candidates[0]

    def delete_files(self, epoch_nums=None, spectra_nums=None, data_types=None,
                     backup_flag=True, property_to_delete='', delete_all=False, 
                     delete_all_in_folder=False):
        """
        Deletes files or folders for this star for the specified epochs, spectra, and data types.
        
        - property_to_delete: name of the property (without .npz) or folder in 'output' to delete
        - If delete_all=True, we skip checks (i.e., if epoch_nums/spectra_nums/data_types are None, we do them all).
        - If delete_all_in_folder=True, we bypass user confirmation for files in the folder.
        """
        # Validate or interpret arguments
        if not property_to_delete:
            print("No property_to_delete provided. Nothing to delete.")
            return
        
        all_epochs = False
        all_specs = False
        all_dtypes = False
        
        if epoch_nums is None or spectra_nums is None or data_types is None:
            if not delete_all:
                raise ValueError("Some of epoch_nums, spectra_nums, data_types are None, and delete_all=False. Aborting.")
            # If delete_all is True, set them to "all"
            if epoch_nums is None:
                all_epochs = True
            if spectra_nums is None:
                all_specs = True
            if data_types is None:
                all_dtypes = True
        
        # Resolve actual lists
        if isinstance(epoch_nums, (int, str)):
            epoch_nums = [epoch_nums]
        if isinstance(spectra_nums, (int, str)):
            spectra_nums = [spectra_nums]
        if isinstance(data_types, str):
            data_types = [data_types]

        if all_epochs:
            epoch_nums = self.get_all_epoch_numbers()

        # For each epoch
        for ep in (epoch_nums if not all_epochs else self.get_all_epoch_numbers()):
            ep_str = f"epoch{ep}"
            ep_path = os.path.join(self.data_dir, self.star_name, ep_str)
            if not os.path.isdir(ep_path):
                continue

            # For each spectra
            if all_specs:
                # read subfolders as spectra 
                possible_specs = [d for d in os.listdir(ep_path)
                                  if os.path.isdir(os.path.join(ep_path, d)) and d.isdigit()]
            else:
                possible_specs = [str(s) for s in spectra_nums]

            for s_num_str in possible_specs:
                s_path = os.path.join(ep_path, s_num_str)
                if not os.path.isdir(s_path):
                    continue

                # For each data type
                if all_dtypes:
                    # read subfolders as data_type
                    possible_dtypes = [d for d in os.listdir(s_path)
                                       if os.path.isdir(os.path.join(s_path, d))]
                else:
                    possible_dtypes = data_types or []

                for dt in possible_dtypes:
                    dt_path = os.path.join(s_path, dt)
                    if not os.path.isdir(dt_path):
                        continue

                    # The 'output' folder
                    output_dir = os.path.join(dt_path, 'output')
                    if not os.path.isdir(output_dir):
                        continue

                    # property_to_delete => .npz
                    property_path = os.path.join(output_dir, property_to_delete + '.npz')
                    if os.path.isfile(property_path):
                        self._delete_file(property_path, backup_flag)
                    elif os.path.isdir(property_path):
                        self._handle_folder_deletion(property_path, backup_flag, delete_all_in_folder)
                    else:
                        # Check if there are multiple matching
                        pattern = os.path.join(output_dir, property_to_delete + '*')
                        matches = glob.glob(pattern)
                        if matches:
                            self._handle_matching_files(matches, backup_flag, delete_all_in_folder)
                        else:
                            print(f"No file/folder matching {property_to_delete} in {output_dir}")

    def _delete_file(self, file_path, backup_flag):
        """Helper method: backs up and then deletes a file."""
        if backup_flag:
            self.backup_property(file_path, overwrite=False)
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    def _handle_folder_deletion(self, folder_path, backup_flag, delete_all_in_folder):
        """Helper method to delete files in a folder."""
        files_in_folder = [f for f in os.listdir(folder_path)
                           if os.path.isfile(os.path.join(folder_path, f))]
        if not files_in_folder:
            print(f"The folder {folder_path} is empty.")
            return

        if delete_all_in_folder:
            selected_indices = list(range(1, len(files_in_folder) + 1))
        else:
            print(f"\nFolder '{folder_path}' contains:")
            for i, f in enumerate(files_in_folder, start=1):
                print(f"{i}. {f}")
            while True:
                resp = input("Which files to delete? (comma list or 'all'): ")
                if resp.strip().lower() == 'all':
                    selected_indices = list(range(1, len(files_in_folder) + 1))
                    break
                else:
                    try:
                        selected_indices = [int(x) for x in resp.split(',')]
                        if all(1 <= x <= len(files_in_folder) for x in selected_indices):
                            break
                        else:
                            print("Invalid selection.")
                    except:
                        print("Invalid input.")

        for idx in selected_indices:
            fullpath = os.path.join(folder_path, files_in_folder[idx - 1])
            self._delete_file(fullpath, backup_flag)

        # Try removing folder if empty
        if not os.listdir(folder_path):
            try:
                os.rmdir(folder_path)
                print(f"Deleted empty folder: {folder_path}")
            except Exception as e:
                print(f"Error removing folder {folder_path}: {e}")

    def _handle_matching_files(self, matching_files, backup_flag, delete_all_in_folder):
        """Helper to handle multiple matching files for property deletion."""
        if not matching_files:
            print("No files to delete.")
            return
        
        print("Found matching files:")
        for i, mf in enumerate(matching_files, start=1):
            print(f"{i}. {mf}")
        if delete_all_in_folder:
            selected_indices = list(range(1, len(matching_files)+1))
        else:
            while True:
                resp = input("Which to delete? (comma list or 'all'): ")
                if resp.strip().lower() == 'all':
                    selected_indices = list(range(1, len(matching_files)+1))
                    break
                else:
                    try:
                        selected_indices = [int(x) for x in resp.split(',')]
                        if all(1 <= x <= len(matching_files) for x in selected_indices):
                            break
                        else:
                            print("Invalid selection.")
                    except:
                        print("Invalid input.")
        
        for idx in selected_indices:
            fp = matching_files[idx - 1]
            if os.path.isdir(fp):
                self._handle_folder_deletion(fp, backup_flag, delete_all_in_folder)
            else:
                self._delete_file(fp, backup_flag)

    def clean(self):
        """
        Cleans up empty folders within the 'output' directories for every epoch, spectra, data_type.
        """
        star_path = os.path.join(self.data_dir, self.star_name)
        if not os.path.isdir(star_path):
            print(f"No data for star {self.star_name} in {self.data_dir}.")
            return

        # star_name/epoch{N}/spectra_num/data_type/output
        for ep_name in os.listdir(star_path):
            ep_path = os.path.join(star_path, ep_name)
            if not os.path.isdir(ep_path) or not ep_name.startswith("epoch"):
                continue

            for spec_folder in os.listdir(ep_path):
                spec_path = os.path.join(ep_path, spec_folder)
                if not os.path.isdir(spec_path) or not spec_folder.isdigit():
                    continue

                for dt_folder in os.listdir(spec_path):
                    dt_path = os.path.join(spec_path, dt_folder)
                    if not os.path.isdir(dt_path):
                        continue

                    output_dir = os.path.join(dt_path, 'output')
                    if not os.path.isdir(output_dir):
                        continue

                    # Clean subfolders inside 'output'
                    for item in os.listdir(output_dir):
                        item_path = os.path.join(output_dir, item)
                        if os.path.isdir(item_path):
                            # check if empty or only .ipynb_checkpoints
                            contents = os.listdir(item_path)
                            if (not contents) or \
                               (len(contents) == 1 and contents[0] == '.ipynb_checkpoints'):
                                # remove .ipynb_checkpoints if present
                                if '.ipynb_checkpoints' in contents:
                                    cfile = os.path.join(item_path, '.ipynb_checkpoints')
                                    try:
                                        os.chmod(cfile, 0o777)
                                        os.remove(cfile)
                                    except Exception as e:
                                        print(f"Error removing {cfile}: {e}")
                                try:
                                    os.rmdir(item_path)
                                    print(f"Removed empty folder {item_path}")
                                except Exception as e:
                                    print(f"Error removing folder {item_path}: {e}")

    ########################################  Load / Save Properties  ########################################

    def _load_file(self, file_path):
        """Loads data from a .npz file (or None if error)."""
        try:
            if file_path.endswith('.npz'):
                with np.load(file_path, allow_pickle=True) as data:
                    if len(data.files) == 1 and 'data' in data.files:
                        return data['data']
                    else:
                        return dict(data)
            else:
                print(f"Unsupported format: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    def _generate_output_file_path(self, method_name, params, multiple_params):
        """
        Build a path for saving method results, e.g.:

        star_name/epochX/spectraY/data_type/output/method_name(.npz or subfolder).
        """
        ep = params['epoch_num']
        sp = params['spectra_num']
        dt = params.get('data_type', '1D')  # default to 1D if missing

        output_dir = os.path.join(self._data_type_path(ep, sp, dt), 'output')
        os.makedirs(output_dir, exist_ok=True)

        if multiple_params:
            # subfolder with method name
            method_dir = os.path.join(output_dir, method_name)
            os.makedirs(method_dir, exist_ok=True)
            # build param-based name
            param_str = '_'.join(f"{k}{v}" for k,v in sorted(params.items()) if k not in ['epoch_num','spectra_num','data_type'])
            param_str = param_str.replace('/','_').replace('\\','_').replace(':','_')
            filename = f"{param_str}.npz"
            return os.path.join(method_dir, filename)
        else:
            # single param => method_name.npz in output_dir
            return os.path.join(output_dir, f"{method_name}.npz")

    def _save_result(self, output_file_path, result, params):
        """Save (result, params) to .npz."""
        np.savez_compressed(output_file_path, result=result, params=params)

    def save_property(self, property_name, property_data, epoch_num, spectra_num, data_type='1D',
                      overwrite=False, backup=True):
        """
        Save a property as .npz under:
        
        star_name/epoch{epoch_num}/{spectra_num}/{data_type}/output/property_name.npz

        Defaults to data_type='1D' if not specified.
        """
        base_dir = self._data_type_path(epoch_num, spectra_num, data_type)
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"No folder found for epoch={epoch_num}, spectra={spectra_num}, data_type={data_type}")

        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        outpath = os.path.join(output_dir, f"{property_name}.npz")
        if os.path.exists(outpath):
            if not overwrite:
                raise FileExistsError(f"{outpath} exists. Use overwrite=True to replace.")
            if backup:
                self.print(f"Backing up before overwrite: {outpath}")
                self.backup_property(outpath, overwrite=True)

        if isinstance(property_data, dict):
            np.savez(outpath, **property_data)
        else:
            np.savez(outpath, data=property_data)
        print(f"Saved property to {outpath}")

    def backup_property(self, output_path, overwrite):
        """
        Moves file to Backups/<overwritten|deleted>/... with timestamp.
        """
        if not os.path.isfile(output_path):
            raise FileNotFoundError(f"Cannot backup non-existent {output_path}")

        backup_type = 'overwritten' if overwrite else 'deleted'
        # Build backup dir
        backup_base = "Backups"
        # e.g. "Backups/overwritten/star_name/epochX/spectraX/data_type"
        # So let's parse out the partial path after self.data_dir
        rel_to_data = os.path.relpath(os.path.dirname(output_path), self.data_dir)
        backup_dir = os.path.join(backup_base, backup_type, rel_to_data)
        os.makedirs(backup_dir, exist_ok=True)

        prop_name = os.path.splitext(os.path.basename(output_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{prop_name}_backup_{timestamp}.npz"
        backup_dest = os.path.join(backup_dir, backup_name)

        os.rename(output_path, backup_dest)
        print(f"Backed up {output_path} -> {backup_dest}")

    ########################################  Directory Queries  ########################################

    def get_all_epoch_numbers(self):
        """
        Scan star_name/ for folders named epochN => return the N's as ints (sorted).
        """
        base = os.path.join(self.data_dir, self.star_name)
        if not os.path.isdir(base):
            return []
        epochs = []
        for d in os.listdir(base):
            if d.startswith('epoch'):
                try:
                    num = int(d.replace('epoch',''))
                    epochs.append(num)
                except:
                    pass
        return sorted(epochs)

    def get_all_spectra_in_epoch(self, epoch_num):
        """
        Return a sorted list of integer 'spectra numbers' found in star_name/epoch{epoch_num}.
        """
        ep_path = self._epoch_path(epoch_num)
        if not os.path.isdir(ep_path):
            return []
        specs = []
        for d in os.listdir(ep_path):
            if os.path.isdir(os.path.join(ep_path, d)) and d.isdigit():
                specs.append(int(d))
        return sorted(specs)

    def get_all_data_types(self, epoch_num, spectra_num):
        """
        Return subfolder names under star_name/epoch{N}/{spectra_num}.
        e.g. ['1D','2D','raw'] if they exist.
        """
        sp_path = self._spectra_path(epoch_num, spectra_num)
        if not os.path.isdir(sp_path):
            return []
        dtypes = []
        for d in os.listdir(sp_path):
            dtp = os.path.join(sp_path, d)
            if os.path.isdir(dtp):
                dtypes.append(d)
        return dtypes

    def list_available_properties(self):
        """
        List all .npz or folders in 'output' subdirectories for every epoch, spectrum, data_type.
        """
        star_path = os.path.join(self.data_dir, self.star_name)
        if not os.path.isdir(star_path):
            print(f"No data for star {self.star_name}")
            return
        
        table_data = []
        for ep in self.get_all_epoch_numbers():
            ep_path = self._epoch_path(ep)
            for sp in self.get_all_spectra_in_epoch(ep):
                sp_path = self._spectra_path(ep, sp)
                for dt in self.get_all_data_types(ep, sp):
                    dt_path = self._data_type_path(ep, sp, dt)
                    output_dir = os.path.join(dt_path, 'output')
                    if os.path.isdir(output_dir):
                        items = os.listdir(output_dir)
                        if items:
                            for it in items:
                                item_path = os.path.join(output_dir, it)
                                if os.path.isdir(item_path):
                                    # folder
                                    numf = len([f for f in os.listdir(item_path)
                                                if os.path.isfile(os.path.join(item_path, f))])
                                    table_data.append({
                                        'Epoch': ep,
                                        'Spectra': sp,
                                        'DataType': dt,
                                        'Property': it,
                                        'Type': 'Folder',
                                        'Details': f'{numf} files'
                                    })
                                else:
                                    # file
                                    table_data.append({
                                        'Epoch': ep,
                                        'Spectra': sp,
                                        'DataType': dt,
                                        'Property': it,
                                        'Type': 'File',
                                        'Details': ''
                                    })
                        else:
                            table_data.append({
                                'Epoch': ep,
                                'Spectra': sp,
                                'DataType': dt,
                                'Property': '(empty output)',
                                'Type': '',
                                'Details': ''
                            })
        
        if not table_data:
            print("No properties found.")
            return
        
        print(f"\nAvailable properties for star '{self.star_name}':\n")
        header = "{:<10} {:<10} {:<10} {:<40} {:<10} {:<15}".format(
            'Epoch', 'Spectra', 'DataType', 'Property', 'Type', 'Details'
        )
        print(header)
        print('-' * len(header))
        for row in table_data:
            print("{:<10} {:<10} {:<10} {:<40} {:<10} {:<15}".format(
                row['Epoch'], row['Spectra'], row['DataType'],
                row['Property'], row['Type'], row['Details']
            ))
        print()

    def load_property(self, property_name, epoch_num, spectra_num, data_type='1D', to_print=True):
        """
        Load a .npz from star_name/epoch{epoch_num}/{spectra_num}/{data_type}/output/property_name.npz
        If property_name is a folder, ask which file from that folder to load.
        Default data_type='1D' if not specified.
        """
        dt_path = self._data_type_path(epoch_num, spectra_num, data_type)
        output_dir = os.path.join(dt_path, 'output')
        if not os.path.isdir(output_dir):
            self.print(f"No output dir for epoch={epoch_num}, spec={spectra_num}, data_type={data_type}", to_print)
            return None
        
        prop_file = os.path.join(output_dir, property_name + '.npz')
        if os.path.isfile(prop_file):
            return self._load_file(prop_file)
        elif os.path.isdir(prop_file):
            # folder approach
            file_list = [f for f in os.listdir(prop_file) if os.path.isfile(os.path.join(prop_file, f))]
            if not file_list:
                self.print(f"Folder {prop_file} is empty.", to_print)
                return None
            print(f"Folder {prop_file} contains:")
            for i,f in enumerate(file_list, start=1):
                print(f"{i}. {f}")
            while True:
                resp = input("Choose file number to load: ")
                try:
                    idx = int(resp)
                    if 1 <= idx <= len(file_list):
                        break
                    else:
                        self.print("Invalid selection.", to_print)
                except:
                    self.print("Invalid input.", to_print)
            chosen = file_list[idx-1]
            return self._load_file(os.path.join(prop_file, chosen))
        else:
            self.print(f"No file/folder '{property_name}' in {output_dir}", to_print)
            return None

    ########################################  Loading Observations  ########################################

    def load_observation(self, epoch_num, spectra_num, data_type='1D'):
        """
        Returns a FITSFile object for star_name/epoch{epoch_num}/{spectra_num}/{data_type}/...
        Default data_type='1D' if not specified.
        """
        fpath = self.get_file_path(epoch_num, spectra_num, data_type)
        if not fpath:
            print(f"No .fits.fz found for epoch={epoch_num}, spec={spectra_num}, data_type={data_type}")
            return None
        try:
            ff = myfits(fpath)
            ff.load_data()
            return ff
        except Exception as e:
            print(f"Error loading FITS from {fpath}: {e}")
            return None

    ########################################  Creating an Observation Table  ########################################

    def create_observation_table(self, epoch_list=None, spectra_list=None,
                                 data_types=None, attributes_list=None, print_table=True):
        """
        Dynamically scans the star's new folder structure for .fits.fz files,
        extracts header attributes, and returns a pandas DataFrame.

        epoch_list, spectra_list, data_types can filter the search.
        attributes_list => any header keywords to retrieve.
        """
        if epoch_list is None:
            epoch_list = self.get_all_epoch_numbers()
        elif isinstance(epoch_list, (int,str)):
            epoch_list = [epoch_list]

        if spectra_list and isinstance(spectra_list, (int,str)):
            spectra_list = [spectra_list]

        if data_types and isinstance(data_types, str):
            data_types = [data_types]

        rows = []
        for ep in epoch_list:
            possible_specs = spectra_list or self.get_all_spectra_in_epoch(ep)
            for sp in possible_specs:
                possible_dtypes = data_types or self.get_all_data_types(ep, sp)
                for dt in possible_dtypes:
                    fpath = self.get_file_path(ep, sp, dt)
                    if not fpath:
                        continue
                    # read FITS headers
                    try:
                        with fits.open(fpath) as hdul:
                            row = {
                                'Epoch': ep,
                                'Spectra': sp,
                                'DataType': dt,
                                'File': os.path.basename(fpath)
                            }
                            if attributes_list:
                                for attr in attributes_list:
                                    val = hdul[0].header.get(attr, 'Unknown')
                                    if val == 'Unknown' and len(hdul)>1:
                                        val2 = hdul[1].header.get(attr, 'Unknown')
                                        if val2 != 'Unknown':
                                            val = val2
                                    row[attr] = val
                            rows.append(row)
                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")

        df = pd.DataFrame(rows)
        if print_table:
            display(df)
        return df

    ########################################  Plot Methods (Left Empty) ########################################

    def plot_raw_spectra(self, epoch_num, spectra_num,
                         subtract_sky=True,
                         blaze_correction=False,
                         just_sky=False,
                         just_target=True):
        """
        Plot raw spectra for this NRES star from a single epoch & spectrum,
        filtering out any zero-flux points.
    
        Parameters
        ----------
        epoch_num : int or str
            Which epoch folder "epoch{N}" to load from.
        spectra_num : int or str
            Which spectrum folder inside that epoch.
        subtract_sky : bool
            If True, subtract sky from object. We assume the flux array has pairs:
              - flux[2*i]   = sky    (even indices)
              - flux[2*i+1] = object (odd indices)
            and similarly for blaze.
        blaze_correction : bool
            If True, divides both sky and object by their respective blaze arrays
            before plotting.
        just_sky : bool
            If True, only plots the sky component(s).
        just_target : bool
            If True, only plots the target (object) component(s).
    
        Notes
        -----
        - If both just_sky=False and just_target=False, no plots will be produced.
          You might want to handle that case (e.g., error out or default to plotting both).
        - If both just_sky=True and just_target=True, you'll get both traces (or
          one trace if subtract_sky=True, since that merges the target’s trace into
          (object - sky)).
    
        Behavior
        --------
          - Loads data via `self.load_observation(epoch_num, spectra_num, "1D")`.
          - Expects data like:
              data['wavelength'] -> array of shape (2*N, ...)
              data['flux']       -> array of shape (2*N, ...)
              data['blaze']      -> array of shape (2*N, ...) [if relevant]
          - Filters out any flux == 0 points before plotting.
          - If subtract_sky=True, it plots (object - sky). Otherwise, it plots
            sky and object separately (as determined by `just_sky`/`just_target`).
        """
    
        # 1) Load the 1D data
        fits_file = self.load_observation(epoch_num, spectra_num, "1D")
        if not fits_file:
            print(f"Failed to load observation for epoch={epoch_num}, spec={spectra_num}.")
            return
    
        data = fits_file.data
    
        # 2) Direct dictionary access
        wave_arrays  = np.array(data['wavelength'], dtype=object)
        flux_arrays  = np.array(data['flux'],       dtype=object)
        blaze_arrays = np.array(data['blaze'],      dtype=object)
    
        orders = len(wave_arrays)
        if orders == 0:
            print("No wavelength data found in the FITS. Aborting.")
            return
    
        # 3) Prepare figure
        plt.figure(figsize=(8, 6))
    
        # We'll assume pairs: (sky = even, object = odd)
        n_pairs = orders // 2
    
        # Check for the trivial case where user sets both flags to False
        if not just_sky and not just_target:
            print("Both just_sky=False and just_target=False → no plots to produce.")
            return
    
        for order_idx in range(n_pairs):
            sky_idx = 2 * order_idx
            obj_idx = 2 * order_idx + 1
    
            wave_sky = wave_arrays[sky_idx]
            flux_sky = flux_arrays[sky_idx]
            wave_obj = wave_arrays[obj_idx]
            flux_obj = flux_arrays[obj_idx]
    
            # If blaze correction applies
            if blaze_correction and (blaze_arrays is not None) and (len(blaze_arrays) == orders):
                blaze_sky = blaze_arrays[sky_idx]
                blaze_obj = blaze_arrays[obj_idx]
    
                valid_sky = (blaze_sky != 0)
                valid_obj = (blaze_obj != 0)
    
                # Copy to avoid altering original arrays
                flux_sky = flux_sky.copy()
                flux_obj = flux_obj.copy()
    
                flux_sky[valid_sky] /= blaze_sky[valid_sky]
                flux_obj[valid_obj] /= blaze_obj[valid_obj]
    
            # --- Plotting Logic ---
            if subtract_sky:
                # We'll plot (object - sky) as the 'object flux' (if just_target)
                flux_obj_sub = flux_obj - flux_sky
                mask_sub = (flux_obj_sub != 0) & (wave_obj != 0)
    
                w_obj_filtered = wave_obj[mask_sub]
                f_obj_filtered = flux_obj_sub[mask_sub]
    
                if just_target:
                    # Plot the subtracted result as the "target"
                    plt.plot(w_obj_filtered, f_obj_filtered, label="(obj - sky)" if order_idx == 0 else "")
                if just_sky:
                    # If we want sky alone as well, we plot the original sky
                    mask_sky = (flux_sky != 0) & (wave_sky != 0)
                    w_sky_filtered = wave_sky[mask_sky]
                    f_sky_filtered = flux_sky[mask_sky]
                    plt.plot(w_sky_filtered, f_sky_filtered, label="sky" if order_idx == 0 else "")
    
            else:
                # No subtraction: plot sky and/or object as separate lines
                if just_sky:
                    mask_sky = (flux_sky != 0) & (wave_sky != 0)
                    w_sky_filtered = wave_sky[mask_sky]
                    f_sky_filtered = flux_sky[mask_sky]
                    plt.plot(w_sky_filtered, f_sky_filtered,
                             label="sky" if order_idx == 0 else "")
    
                if just_target:
                    mask_obj = (flux_obj != 0) & (wave_obj != 0)
                    w_obj_filtered = wave_obj[mask_obj]
                    f_obj_filtered = flux_obj[mask_obj]
                    plt.plot(w_obj_filtered, f_obj_filtered,
                             label="object" if order_idx == 0 else "")
    
        plt.title(f"NRES Raw Spectra: {self.star_name}, epoch={epoch_num}, spec={spectra_num}")
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Flux")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    def _stitch_spectra_by_snr(self,wave_list, flux_list, snr_list):
        """
        Combine multiple spectra into a single spectrum by handling overlaps and
        combining fluxes in the overlapping region using an SNR-weighted average.
        
        Parameters
        ----------
        wave_list : list of 1D numpy arrays
            Each entry is a wavelength array for one spectrum (must be sorted!).
        flux_list : list of 1D numpy arrays
            Each entry is a flux array corresponding to the wavelengths in wave_list.
        snr_list : list of 1D numpy arrays
            Each entry is an SNR array corresponding to the wavelengths in wave_list.
    
        Returns
        -------
        combined_wave : 1D numpy array
            Combined wavelength array.
        combined_flux : 1D numpy array
            Combined flux array.
        combined_snr : 1D numpy array
            Combined SNR array.
        """
    
        def find_overlap(combined_wave, wv_current):
            # Remove leading and trailing zeros from combined_wave
            combined_wave_nonzero = combined_wave[np.nonzero(combined_wave)[0][0]: np.nonzero(combined_wave)[0][-1] + 1]
        
            # Remove leading and trailing zeros from wv_current
            wv_current_nonzero = wv_current[np.nonzero(wv_current)[0][0]: np.nonzero(wv_current)[0][-1] + 1]
        
            # Calculate the overlap
            overlap_start = max(combined_wave_nonzero[0], wv_current_nonzero[0])
            overlap_end = min(combined_wave_nonzero[-1], wv_current_nonzero[-1])
        
            return overlap_start, overlap_end
            
        # Start from the first spectrum in the list
        combined_wave = wave_list[0].copy()
        combined_flux = flux_list[0].copy()
        combined_snr  = snr_list[0].copy()

        # Remove any zeros in wave, flux, or SNR
        # mask = (combined_wave != 0) & (combined_flux != 0) & (combined_snr != 0)
        mask = (combined_wave != 0) 
        combined_wave = combined_wave[mask]
        combined_flux = combined_flux[mask]
        combined_snr  = combined_snr[mask]
        
        # Ensure ascending order (in case flipping gave you descending order)
        # (Only needed if the original was descending. 
        #  If flip already yields ascending, you can skip this sort.)
        idx_sort = np.argsort(combined_wave)
        combined_wave = combined_wave[idx_sort]
        combined_flux = combined_flux[idx_sort]
        combined_snr  = combined_snr[idx_sort]
        
        # Loop over all other spectra in the list
        for i in range(1, len(wave_list)):
            wv_current = wave_list[i]
            fl_current = flux_list[i]
            sn_current = snr_list[i]
            # print(wv_current[500:530])

            # Remove zeros
            # mask_cur = (wv_current != 0) & (fl_current != 0) & (sn_current != 0)
            mask_cur = (wv_current != 0)
            wv_current = wv_current[mask_cur]
            fl_current = fl_current[mask_cur]
            sn_current = sn_current[mask_cur]
        
            # Sort ascending if needed
            idx_sort_cur = np.argsort(wv_current)
            wv_current   = wv_current[idx_sort_cur]
            fl_current   = fl_current[idx_sort_cur]
            sn_current   = sn_current[idx_sort_cur]
            
            # 1) Identify the overlap between combined_wave and wv_current
            overlap_start, overlap_end = find_overlap(combined_wave, wv_current)
            # print(overlap_start < overlap_end)
            
            if overlap_start < overlap_end:
                # We have an overlapping region
                # Get indices for combined and current spectra in the overlapping region
                idx_combined = np.where((combined_wave >= overlap_start) & 
                                        (combined_wave <= overlap_end))[0]
                idx_current  = np.where((wv_current   >= overlap_start) & 
                                        (wv_current   <= overlap_end))[0]
                # print(f'overlaps are: {overlap_start} and {overlap_end}')
                # print(f'wave in combined: {combined_wave[idx_combined]}')
                # print(f'wave in current: {wv_current[idx_current]}')
                # print(f'len of overlap in combined: {len(combined_wave[idx_combined[0]:])}, and also {len(idx_combined)} and maximum of combined: {max(combined_wave)}')
                # print(f'len of overlap in wv_current: {len(wv_current)}, and also {len(idx_current)} and minimum of current: {min(wv_current)}')
    
                # 2) For correct averaging, we want flux at the same wavelength points.
                #    Whichever spectrum has finer sampling, we will interpolate the other one onto it.
                dw_combined = np.mean(np.diff(combined_wave[idx_combined])) if len(idx_combined) > 1 else 1e10
                dw_current  = np.mean(np.diff(wv_current[idx_current]))     if len(idx_current)  > 1 else 1e10
                # print(np.diff(combined_wave[idx_combined]))
                # print(np.diff(wv_current[idx_current]))
                # print(dw_combined <= dw_current)
    
                if dw_combined <= dw_current:
                    # Combined array is finer (or equal); interpolate current onto combined
                    wv_fine  = combined_wave[idx_combined]
                    fl_fine  = combined_flux[idx_combined]
                    sn_fine  = combined_snr[idx_combined]
                    # print(f'sn_fine is: {sn_fine[:50]}')
                    
                    fl_interp = np.interp(wv_fine, wv_current[idx_current], fl_current[idx_current])
                    sn_interp = np.interp(wv_fine, wv_current[idx_current], sn_current[idx_current])
                    
                    # Compute SNR weights
                    w1 = sn_fine**2
                    w2 = sn_interp**2
                    wsum = w1 + w2
                    # print(f'w1/w2 : {w1[:20]/w2[:20]}, \nsum: {wsum[:20]}')
                    # print(f'the last wavelength of the finer is: {wv_fine[-1]} and the first of the interp is: {wv_current[0]}')
                    
                    # Weighted flux and updated SNR in the overlap
                    combined_flux[idx_combined] = (fl_fine * w1 + fl_interp * w2) / wsum
                    combined_snr[idx_combined]  = np.sqrt(wsum)
                    
                    # Append non-overlapping part of the current spectrum
                    idx_right = np.where(wv_current > overlap_end)[0]
                    if idx_right.size > 0:
                        combined_wave = np.concatenate([combined_wave, wv_current[idx_right]])
                        combined_flux = np.concatenate([combined_flux, fl_current[idx_right]])
                        combined_snr  = np.concatenate([combined_snr,  sn_current[idx_right]])
                    
                else:
                    # Current array is finer; interpolate combined onto current
                    wv_fine  = wv_current[idx_current]
                    fl_fine  = fl_current[idx_current]
                    sn_fine  = sn_current[idx_current]
                    
                    fl_interp = np.interp(wv_fine, combined_wave[idx_combined], combined_flux[idx_combined])
                    sn_interp = np.interp(wv_fine, combined_wave[idx_combined], combined_snr[idx_combined])
                    
                    # Compute SNR weights
                    w1 = sn_fine**2
                    w2 = sn_interp**2
                    wsum = w1 + w2
                    
                    # Weighted flux and updated SNR in the overlap
                    fl_overlap = (fl_fine * w1 + fl_interp * w2) / wsum
                    sn_overlap = np.sqrt(wsum)
                    
                    # We'll rewrite the current arrays in the overlap
                    fl_current[idx_current] = fl_overlap
                    sn_current[idx_current] = sn_overlap
                    
                    # Now we must keep the left side of combined if it does not overlap
                    idx_left = np.where(combined_wave < overlap_start)[0]
                    if idx_left.size > 0:
                        wv_current = np.concatenate([combined_wave[idx_left], wv_current])
                        fl_current = np.concatenate([combined_flux[idx_left], fl_current])
                        sn_current = np.concatenate([combined_snr[idx_left],  sn_current])
                    
                    # The 'current' arrays become our new combined arrays
                    combined_wave = wv_current
                    combined_flux = fl_current
                    combined_snr  = sn_current
    
            else:
                # No overlap, just concatenate
                combined_wave = np.concatenate([combined_wave, wv_current])
                combined_flux = np.concatenate([combined_flux, fl_current])
                combined_snr  = np.concatenate([combined_snr,  sn_current])
            
            # Sort combined arrays by wavelength
            idx_sort = np.argsort(combined_wave)
            combined_wave = combined_wave[idx_sort]
            combined_flux = combined_flux[idx_sort]
            combined_snr  = combined_snr[idx_sort]
    
        return combined_wave, combined_flux, combined_snr
    
    def plot_stitched_spectra(self, epoch_num, spectra_num,
                              subtract_sky=True,
                              my_SNR=False,
                              plot_SNR=False,
                              window_size = 20):
        """
        Load multiple (sky, object) pairs, optionally subtract sky and do blaze 
        correction, then stitch them with SNR weighting. Plots one final flux curve.
        
        Parameters
        ----------
        epoch_num : int or str
            Which "epoch{N}" folder to load from.
        spectra_num : int or str
            Which spectrum folder inside that epoch.
        subtract_sky : bool
            If True, subtract sky from object in each pair.
            (We assume wave_arrays, flux_arrays come in pairs of (sky, obj).)
        my_SNR : bool
            If True, compute SNR using a robust sliding-window approach, where
            for each point i, SNR(i) = robust_mean(window_around_i) / robust_std(window_around_i).
            If False, compute a default SNR from propagated uncertainties.
        plot_SNR : bool
            If True, plot the final stitched SNR after plotting the stitched flux.
    
        Returns
        -------
        combined_wave, combined_flux, combined_snr : np.ndarray
            The stitched wavelength, flux, and SNR arrays.
        """
    
        # 1) Load data
        fits_file = self.load_observation(epoch_num, spectra_num, "1D")
        if not fits_file:
            print(f"Failed to load observation for epoch={epoch_num}, spec={spectra_num}")
            return
    
        data = fits_file.data
        wave_arrays         = np.array(data['wavelength'])
        flux_arrays         = np.array(data['flux'])
        uncertainty_arrays  = np.array(data['uncertainty'])
        blaze_arrays        = np.array(data['blaze'])
        blaze_errors_arrays = np.array(data['blaze_error'])
    
        orders = len(wave_arrays)
        if orders == 0:
            print("No wavelength data found in the FITS. Aborting.")
            return
    
        # 2) We'll collect (wave_obj, flux_obj, snr_obj) for each order
        wave_list = []
        flux_list = []
        snr_list  = []
    
        n_pairs = orders // 2
        for order_idx in range(n_pairs):
            # We'll use the object index for wave (assuming wave_arrays[2*i+1] is the target)
            wave_obj = wave_arrays[order_idx*2 + 1]
            wave_list.append(wave_obj)
    
            # -- Blaze correction + optional sky subtraction inline --
            # object_flux / object_blaze - sky_flux / sky_blaze
            flux_obj = flux_arrays[order_idx*2 + 1] / blaze_arrays[order_idx*2 + 1]
            flux_sky = flux_arrays[order_idx*2]     / blaze_arrays[order_idx*2]
            tmp_current_fluxes = flux_obj - flux_sky
            flux_list.append(tmp_current_fluxes)
    
            # -- Compute SNR in either robust mode or default propagated mode --
            if my_SNR:
                # We want, for each point i, SNR(i) = local_mean / local_std
                # where local_mean and local_std are robustly computed in a window around i.
    
                flux_corrected = tmp_current_fluxes
                n_points = len(flux_corrected)
    
                local_snr = np.full_like(flux_corrected, np.nan)
    
                for i in range(n_points):
                    start_idx = max(0, i - window_size)
                    end_idx   = min(n_points, i + window_size + 1)
    
                    window_flux = flux_corrected[start_idx:end_idx]
    
                    # 1) robust_mean
                    local_mean = ut.robust_mean(window_flux, 3)
    
                    # 2) robust_std
                    local_std = ut.robust_std(window_flux, 3)
    
                    if local_std > 0:
                        # SNR = local_mean / local_std
                        local_snr[i] = local_mean / local_std
                    else:
                        local_snr[i] = np.nan
    
                snr_list.append(local_snr)
    
            else:
                # --- Default SNR from propagated uncertainties ---
                sky_unc   = uncertainty_arrays[order_idx*2]     / blaze_arrays[order_idx*2]
                obj_unc   = uncertainty_arrays[order_idx*2 + 1] / blaze_arrays[order_idx*2 + 1]
                blaze_sky_unc = (
                    flux_arrays[order_idx*2]
                    * blaze_errors_arrays[order_idx*2]
                    / (blaze_arrays[order_idx*2]**2)
                )
                blaze_obj_unc = (
                    flux_arrays[order_idx*2 + 1]
                    * blaze_errors_arrays[order_idx*2 + 1]
                    / (blaze_arrays[order_idx*2 + 1]**2)
                )
    
                total_unc = np.sqrt(
                    sky_unc**2 + obj_unc**2
                    + blaze_sky_unc**2 + blaze_obj_unc**2
                )
                default_snr = tmp_current_fluxes / total_unc
                snr_list.append(default_snr)
    
        # 3) Stitch them (assuming you have self._stitch_spectra_by_snr)
        # Note: We flip them here if your method expects them in a certain order
        wave_list = np.flip(wave_list)
        flux_list = np.flip(flux_list)
        snr_list  = np.flip(snr_list)
    
        combined_wave, combined_flux, combined_snr = self._stitch_spectra_by_snr(
            wave_list, flux_list, snr_list
        )
    
        # 4) Plot final stitched flux
        plt.figure(figsize=(8, 5))
        plt.plot(combined_wave, combined_flux, color='blue', label='Stitched Flux')
        plt.title(f"NRES Stitched Spectra: {self.star_name}, epoch={epoch_num}, spec={spectra_num}")
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Flux (sky-sub, blaze-corr, stitched)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        # 5) Optionally plot the final stitched SNR
        if plot_SNR:
            plt.figure(figsize=(8, 5))
            plt.plot(combined_wave, combined_snr, 'r-', label='Stitched SNR')
            plt.title(f"NRES Stitched SNR: {self.star_name}, epoch={epoch_num}, spec={spectra_num}")
            plt.xlabel("Wavelength (A)")
            plt.ylabel("SNR = local_mean / local_std (or default SNR)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
        return combined_wave, combined_flux, combined_snr

    def plot_normalized_spectra(self, epoch_nums=None, spectra_nums=None, save=False,
                                separate=True, initial_separation=100, bin_window=10, clean=True):
        """
        Improved implementation of the plot_normalized_spectra function:
        Plots normalized spectra for the specified epoch(s) and spectra number(s).
        If multiple curves are plotted and `separate` is True, a slider is added to adjust the vertical offset.
        Precomputes data to improve dynamic slider performance.

        Parameters
        ----------
        epoch_nums : int, str, or list, optional
            The epoch number(s) to plot. If None, all epochs are used.
        spectra_nums : int, str, or list, optional
            The spectra number(s) to plot. If None, all spectra for each epoch are used.
        save : bool, optional
            If True, the figure is saved.
        separate : bool, optional
            If True, each spectrum is vertically offset; the offset is controlled via a slider.
        initial_separation : float, optional
            The initial vertical separation (offset) to apply between curves. Default is 100.
        bin_window : int, optional
            The binning window size (number of points per bin). Default is 10.
        clean : bool, optional
            If True (default), the method will attempt to load 'clean_normalized_flux'. If not available,
            it will fall back to 'normalized_flux' and indicate it in the legend.

        Returns
        -------
        None
        """
        from itertools import product
        import os
        from datetime import datetime
        from matplotlib.widgets import Slider

        # Process epoch_nums
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, list):
            epoch_nums = [epoch_nums]

        # Process spectra_nums
        if spectra_nums is None:
            specs_all = []
            for ep in epoch_nums:
                specs_all.extend(self.get_all_spectra_in_epoch(ep))
            spectra_nums = list(set(specs_all))
        elif not isinstance(spectra_nums, list):
            spectra_nums = [spectra_nums]

        # Generate all (epoch, spectra) combinations
        combinations = list(product(epoch_nums, spectra_nums))
        if len(combinations) == 0:
            print("No valid epoch/spectra combinations found.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        curves = []
        combo_list = []  # Store (epoch, spec) for each curve
        precomputed_data = []  # Store precomputed curve data to avoid redundant calculations

        # Loop over combinations and plot
        for i, (ep, sp) in enumerate(combinations):
            label_suffix = ""
            try:
                if clean:
                    norm_data = self.load_property('clean_normalized_flux', ep, sp)
                    if norm_data is None:
                        raise ValueError("Clean property not found.")
                    label_suffix = " (clean)"
                else:
                    raise ValueError("Clean flag False")
            except Exception as e:
                norm_data = self.load_property('normalized_flux', ep, sp)
                label_suffix = " (unclean)"

            wavelengths = norm_data['wavelengths']
            normalized_flux = norm_data['normalized_flux']
            # Bin the data if bin_window > 1
            if bin_window > 1:
                binned_wave = np.array([np.mean(wavelengths[j:j + bin_window])
                                        for j in range(0, len(wavelengths), bin_window)])
                binned_norm = np.array([ut.robust_mean(normalized_flux[j:j + bin_window], 3)
                                        for j in range(0, len(normalized_flux), bin_window)])
            else:
                binned_wave = wavelengths
                binned_norm = normalized_flux

            # Precompute curve data to avoid recalculations
            precomputed_data.append((binned_wave, binned_norm))
            offset = initial_separation * i if separate else 0

            line, = ax.plot(binned_wave, binned_norm + offset,
                            label=f"Epoch {ep}, Spec {sp}{label_suffix}")
            curves.append(line)
            combo_list.append((ep, sp))

        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Normalized Spectra for {self.star_name}')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Add slider for vertical separation if needed
        if separate and len(curves) > 1:
            ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
            slider = Slider(ax_slider, 'Separation', 0, 500, valinit=initial_separation)

            def update(val):
                sep = slider.val
                for j, (binned_wave, binned_norm) in enumerate(precomputed_data):
                    curves[j].set_ydata(binned_norm + sep * j)
                fig.canvas.draw_idle()

            slider.on_changed(update)

        if save:
            output_dir = os.path.join(self.data_dir, self.star_name, 'output', 'Figures')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.star_name}_normalized_{timestamp}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            print(f"Figure saved to '{save_path}'")

        plt.show()

    def plot_normalized_spectra_old(self, epoch_nums=None, spectra_nums=None, save=False,
                                separate=True, initial_separation=100, bin_window=10, clean=True):
        """
        Plots normalized spectra for the specified epoch(s) and spectra number(s).
        If multiple curves are plotted and `separate` is True, a slider is added to adjust the vertical offset.
        Additionally, if bin_window > 1, the spectra are binned over the specified window.

        A new flag `clean` (default True) is added: if True, the function first tries to load the
        property 'clean_normalized_flux' and, if found, plots that. If not, it falls back to using
        'normalized_flux' and adds a note "(unclean)" in the legend.

        Parameters
        ----------
        epoch_nums : int, str, or list, optional
            The epoch number(s) to plot. If None, all epochs are used.
        spectra_nums : int, str, or list, optional
            The spectra number(s) to plot. If None, all spectra for each epoch are used.
        save : bool, optional
            If True, the figure is saved.
        separate : bool, optional
            If True, each spectrum is vertically offset; the offset is controlled via a slider.
        initial_separation : float, optional
            The initial vertical separation (offset) to apply between curves. Default is 100.
        bin_window : int, optional
            The binning window size (number of points per bin). Default is 10.
        clean : bool, optional
            If True (default), the method will attempt to load 'clean_normalized_flux'. If not available,
            it will fall back to 'normalized_flux' and indicate it in the legend.

        Returns
        -------
        None
        """
        from itertools import product
        import os
        from datetime import datetime

        # Process epoch_nums
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, list):
            epoch_nums = [epoch_nums]

        # Process spectra_nums
        if spectra_nums is None:
            specs_all = []
            for ep in epoch_nums:
                specs_all.extend(self.get_all_spectra_in_epoch(ep))
            spectra_nums = list(set(specs_all))
        elif not isinstance(spectra_nums, list):
            spectra_nums = [spectra_nums]

        # Generate all (epoch, spectra) combinations.
        combinations = list(product(epoch_nums, spectra_nums))
        if len(combinations) == 0:
            print("No valid epoch/spectra combinations found.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        curves = []
        combo_list = []  # store (epoch, spec) for each curve

        # Loop over combinations and plot
        for i, (ep, sp) in enumerate(combinations):
            label_suffix = ""
            try:
                if clean:
                    norm_data = self.load_property('clean_normalized_flux', ep, sp)
                    if norm_data is None:
                        raise ValueError("Clean property not found.")
                    label_suffix = " (clean)"
                else:
                    raise ValueError("Clean flag False")
            except Exception as e:
                norm_data = self.load_property('normalized_flux', ep, sp)
                label_suffix = " (unclean)"

            wavelengths = norm_data['wavelengths']
            normalized_flux = norm_data['normalized_flux']
            # Bin the data if bin_window > 1.
            if bin_window > 1:
                binned_wave = np.array([np.mean(wavelengths[j:j + bin_window])
                                        for j in range(0, len(wavelengths), bin_window)])
                binned_norm = np.array([ut.robust_mean(normalized_flux[j:j + bin_window], 3)
                                        for j in range(0, len(normalized_flux), bin_window)])
            else:
                binned_wave = wavelengths
                binned_norm = normalized_flux

            # Apply vertical offset if separation is enabled.
            if separate:
                offset = initial_separation * i
            else:
                offset = 0

            line, = ax.plot(binned_wave, binned_norm + offset,
                            label=f"Epoch {ep}, Spec {sp}{label_suffix}")
            curves.append(line)
            combo_list.append((ep, sp))

        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Normalized Spectra for {self.star_name}')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Add slider for vertical separation if needed.
        if separate and len(curves) > 1:
            from matplotlib.widgets import Slider
            ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
            slider = Slider(ax_slider, 'Separation', 0, 500, valinit=initial_separation)

            def update(val):
                sep = slider.val
                for j, (ep, sp) in enumerate(combo_list):
                    try:
                        norm_data = self.load_property('normalized_flux', ep, sp)
                        wavelengths = norm_data['wavelengths']
                        normalized_flux = norm_data['normalized_flux']
                        if bin_window > 1:
                            binned_wave = np.array([np.mean(wavelengths[k:k + bin_window])
                                                    for k in range(0, len(wavelengths), bin_window)])
                            binned_norm = np.array([ut.robust_mean(normalized_flux[k:k + bin_window], 3)
                                                    for k in range(0, len(normalized_flux), bin_window)])
                        else:
                            binned_wave = wavelengths
                            binned_norm = normalized_flux
                        curves[j].set_ydata(binned_norm + sep * j)
                    except Exception as e:
                        print(f"Error updating curve for Epoch {ep}, Spec {sp}: {e}")
                fig.canvas.draw_idle()

            slider.on_changed(update)

        if save:
            output_dir = os.path.join(self.data_dir, self.star_name, 'output', 'Figures')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.star_name}_normalized_{timestamp}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            print(f"Figure saved to '{save_path}'")

        plt.show()

    def plot_normalized_spectra_old(self, epoch_nums=None, spectra_nums=None, save=False,
                                separate=True, initial_separation=100, bin_window=10):
        """
        Plots normalized spectra for the specified epoch(s) and spectra number(s).
        If multiple curves are plotted and `separate` is True, a slider is added to adjust the vertical offset.
        Additionally, if bin_window > 1, the spectra are binned over the specified window to improve clarity.

        Parameters
        ----------
        epoch_nums : int, str, or list, optional
            The epoch number(s) to plot. If None, all epochs are used.
        spectra_nums : int, str, or list, optional
            The spectra number(s) to plot. If None, all spectra for each epoch are used.
        save : bool, optional
            If True, saves the figure.
        separate : bool, optional
            If True, each spectrum is vertically offset; controlled via a slider.
        initial_separation : float, optional
            The initial vertical separation offset. Default is 100.
        bin_window : int, optional
            The number of data points to bin together. Default is 10.

        Returns
        -------
        None
        """
        from itertools import product
        import os
        from datetime import datetime

        # Process epoch_nums
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif not isinstance(epoch_nums, list):
            epoch_nums = [epoch_nums]

        # Process spectra_nums
        if spectra_nums is None:
            specs_all = []
            for ep in epoch_nums:
                specs_all.extend(self.get_all_spectra_in_epoch(ep))
            spectra_nums = list(set(specs_all))
        elif not isinstance(spectra_nums, list):
            spectra_nums = [spectra_nums]

        # Generate combinations of (epoch, spectra)
        combinations = list(product(epoch_nums, spectra_nums))
        if len(combinations) == 0:
            print("No valid epoch/spectra combinations found.")
            return

        plt.figure(figsize=(12, 6))
        curves = []
        combo_list = []  # Keep track of which combo corresponds to which curve

        for i, (ep, sp) in enumerate(combinations):
            try:
                norm_data = self.load_property('normalized_flux', ep, sp)
                mask =np.where(norm_data['normalized_flux'] > 0)
                wavelengths = norm_data['wavelengths'][mask]
                normalized_flux = norm_data['normalized_flux'][mask]
                # Bin the data if bin_window > 1.
                if bin_window > 1:
                    binned_wave = np.array([np.mean(wavelengths[j:j + bin_window])
                                            for j in range(0, len(wavelengths), bin_window)])
                    # Use robust mean with sigma=3 for flux binning.
                    binned_norm = np.array([np.mean(normalized_flux[j:j + bin_window])
                                            for j in range(0, len(normalized_flux), bin_window)])
                else:
                    binned_wave = wavelengths
                    binned_norm = normalized_flux

                # Compute vertical offset if separation is enabled.
                if separate:
                    offset = initial_separation * i
                else:
                    offset = 0
                line, = plt.plot(binned_wave, binned_norm + offset,
                                 label=f"Epoch {ep}, Spec {sp}")
                curves.append(line)
                combo_list.append((ep, sp))
            except Exception as e:
                print(f"Error loading normalized spectrum for Epoch {ep}, Spec {sp}: {e}")
                continue

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Normalized Flux')
        plt.title(f"Normalized Spectra for {self.star_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # If separation is enabled and more than one curve is plotted, add a slider.
        if separate and len(curves) > 1:
            from matplotlib.widgets import Slider
            ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
            slider = Slider(ax_slider, 'Separation', 0, 500, valinit=initial_separation)

            def update(val):
                sep = slider.val
                for j, (ep, sp) in enumerate(combo_list):
                    try:
                        norm_data = self.load_property('normalized_flux', ep, sp)
                        mask = np.where(norm_data['normalized_flux'] > 0)
                        wavelengths = norm_data['wavelengths'][mask]
                        normalized_flux = norm_data['normalized_flux'][mask]
                        if bin_window > 1:
                            binned_wave = np.array([np.mean(wavelengths[k:k + bin_window])
                                                    for k in range(0, len(wavelengths), bin_window)])
                            binned_norm = np.array([np.mean(normalized_flux[k:k + bin_window])
                                                    for k in range(0, len(normalized_flux), bin_window)])
                        else:
                            binned_wave = wavelengths
                            binned_norm = normalized_flux
                        curves[j].set_ydata(binned_norm + sep * j)
                    except Exception as e:
                        print(f"Error updating curve for Epoch {ep}, Spec {sp}: {e}")
                plt.gcf().canvas.draw_idle()

            slider.on_changed(update)

        # Optionally save the figure.
        if save:
            output_dir = os.path.join(self.data_dir, self.star_name, 'output', 'Figures')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.star_name}_normalized_{timestamp}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            print(f"Figure saved to '{save_path}'")

        plt.show()

    def plot_spectra(self, *args, **kwargs):
        """ Empty placeholder method for plotting spectra. """
        pass

    def plot_spectra_errors(self, *args, **kwargs):
        """ Empty placeholder method for plotting spectra errors. """
        pass

    ########################################  Method Executor  ########################################

    def execute_method_old(self, method, params={}, epoch_nums=None, spectra_nums=None,
                       data_types=None, overwrite=False, backup=True, save=True,
                       parallel=False, max_workers=None):
        """
        Similar logic to the old "execute_method", but adapted to new structure:
        
        For each combination of epoch, spectra_num, data_type, plus param combos => call method.
        """
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif isinstance(epoch_nums, (int,str)):
            epoch_nums = [epoch_nums]

        if spectra_nums and isinstance(spectra_nums, (int,str)):
            spectra_nums = [spectra_nums]

        if data_types and isinstance(data_types, str):
            data_types = [data_types]

        # Build combos
        combos = []
        for ep in epoch_nums:
            s_list = spectra_nums if spectra_nums else self.get_all_spectra_in_epoch(ep)
            for sp in s_list:
                dt_list = data_types if data_types else self.get_all_data_types(ep, sp)
                # If dt_list is empty => fallback to '1D' only
                if not dt_list:
                    dt_list = ['1D']
                for dt in dt_list:
                    combos.append((ep, sp, dt))

        # Expand param sets
        param_keys = list(params.keys())
        param_values_lists = []
        for v in params.values():
            if isinstance(v, list):
                param_values_lists.append(v)
            else:
                param_values_lists.append([v])

        from itertools import product
        final_params = []
        for ep, sp, dt in combos:
            for pvals in product(*param_values_lists):
                p = {
                    'epoch_num': ep,
                    'spectra_num': sp,
                    'data_type': dt
                }
                for k,v in zip(param_keys, pvals):
                    p[k] = v
                final_params.append(p)

        multiple = (len(final_params) > 1)
        if parallel:
            if max_workers is None:
                max_workers = multiprocess.cpu_count() - 1
            with multiprocess.Pool(processes=max_workers) as pool:
                f = partial(self._method_wrapper, method,
                            overwrite=overwrite, backup=backup,
                            multiple_params=multiple, save=save)
                results = pool.map(f, final_params)
        else:
            results = []
            for fp in final_params:
                out = self._method_wrapper(method, fp, overwrite, backup, multiple, save)
                results.append(out)
        return results

    def execute_method(self, method,filename, params={}, epoch_nums=None, spectra_nums=None,
                       overwrite=False, backup=True, save=True, parallel=False, 
                       max_workers=None):
        """
        Executes a given method across multiple epochs and spectra.
    
        Parameters:
            method : function
                The function to execute.
            params : dict
                Parameters to pass to the function.
            epoch_nums : list or int/str
                Epoch numbers to process.
            spectra_nums : list or int/str
                Spectra numbers to process.
            overwrite : bool
                Whether to overwrite existing data.
            backup : bool
                Whether to create a backup before overwriting.
            save : bool
                Whether to save the result.
            parallel : bool
                Whether to execute in parallel.
            max_workers : int or None
                Number of parallel workers.
            filename : str or None
                The filename to use for saving the results.
    
        Returns:
            list : A list of results from executing the method.
        """
        if epoch_nums is None:
            epoch_nums = self.get_all_epoch_numbers()
        elif isinstance(epoch_nums, (int, str)):
            epoch_nums = [epoch_nums]
    
        if spectra_nums and isinstance(spectra_nums, (int, str)):
            spectra_nums = [spectra_nums]
    
        # Build combinations of epochs and spectra
        from itertools import product
        param_keys = list(params.keys())
        param_values_lists = [[v] if not isinstance(v, list) else v for v in params.values()]
    
        final_params = []
        for ep in epoch_nums:
            for sp in (spectra_nums if spectra_nums else self.get_all_spectra_in_epoch(ep)):
                for pvals in product(*param_values_lists):
                    p = {'epoch_num': ep, 'spectra_num': sp, 'data_type': '1D'}
                    for k, v in zip(param_keys, pvals):
                        p[k] = v
                    final_params.append(p)
    
        multiple = len(final_params) > 1
    
        if parallel:
            if max_workers is None:
                max_workers = multiprocess.cpu_count() - 1
            with multiprocess.Pool(processes=max_workers) as pool:
                f = partial(self._method_wrapper, method, filename,
                            overwrite=overwrite, backup=backup,
                            multiple_params=multiple, save=save)
                results = pool.map(f, final_params)
        else:
            results = []
            for fp in final_params:
                out = self._method_wrapper(method, fp, filename, overwrite, backup, multiple, save)
                results.append(out)
    
        return results



    def _method_wrapper(self, method, params, filename, overwrite, backup, multiple_params, save):
        """
        Wrapper to execute a method with given parameters and handle saving.
    
        Parameters:
            method : function
                The function to execute.
            params : dict
                Parameters to pass to the function.
            filename : str or None
                Filename to use in self.save_property().
            overwrite : bool
                Whether to overwrite existing data.
            backup : bool
                Whether to create a backup before overwriting.
            multiple_params : bool
                Whether multiple parameter sets are being used.
            save : bool
                Whether to save the result.
    
        Returns:
            result : any
                The result of executing the function.
        """
        try:
            result = method(**params)
    
            if save and filename:
                epoch_num = params.get('epoch_num')
                spectra_num = params.get('spectra_num')
                data_type = params.get('data_type', '1D')
    
                if epoch_num is None or spectra_num is None:
                    print(f"Skipping save: Missing epoch_num or spectra_num in params: {params}")
                else:
                    # Save using provided filename
                    self.save_property(filename, result, epoch_num, spectra_num,
                                       data_type=data_type, overwrite=overwrite, backup=backup)
                    print(f"Saved result as '{filename}' for epoch {epoch_num}, spec {spectra_num}.")
    
            return result
    
        except Exception as e:
            print(f"Error running method {method.__name__} with {params}: {e}")
            return None

    
    def _method_wrapper_old(self, method, params, overwrite, backup, multiple_params, save):
        """
        Internal helper for method execution + save.
        """
        outpath = self._generate_output_file_path(method.__name__, params, multiple_params)
        if os.path.exists(outpath):
            if not overwrite:
                print(f"Output {outpath} exists. Skipping.")
                return None
            else:
                if backup:
                    self.backup_property(outpath, overwrite=True)
                print(f"Overwriting {outpath}.")

        try:
            result = method(**params)
            if save:
                self._save_result(outpath, result, params)
                print(f"Saved result to {outpath}")
            else:
                print("Skipping save; returning result only.")
            return result
        except Exception as e:
            print(f"Error running method {method.__name__} with {params}: {e}")
            return None

    ########################################  SIMBAD / get_bat_id  ########################################

    def get_bat_id(self):
        """
        Just a direct copy of your SIMBAD-based method to fetch BAT99. 
        """
        base_url = 'https://simbad.u-strasbg.fr/simbad/sim-basic'
        params = {'Ident': self.star_name, 'submit': 'SIMBAD search'}
        try:
            resp = requests.get(base_url, params=params)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
        html = resp.text
        idx = html.find('BAT99')
        for _ in range(3):
            idx = html.find('BAT99', idx+1)
        idx2 = html.find('\n', idx+10)
        bat_num = html[idx+10:idx2]
        try:
            _ = int(bat_num)
            return bat_num
        except:
            print(f"BAT99 not found properly. Indices {idx}, {idx2}. Found {bat_num}")
            return None

    ########################################  Data Manipulation  ########################################

    def get_stitched_spectra(self, epoch_num, spectra_num, subtract_sky=True):
        """
        Similar to plot_stitched_spectra, but returns (wave, flux, snr)
        WITHOUT opening any figure or calling plt.show().
        
        Parameters
        ----------
        epoch_num : int or str
            Which epoch folder to load from.
        spectra_num : int or str
            Which spectrum folder inside that epoch.
        subtract_sky : bool
            If True, subtract sky from object in each pair.
            (We assume wave_arrays, flux_arrays come in pairs of (sky, obj).)
        
        Returns
        -------
        combined_wave : 1D np.array
        combined_flux : 1D np.array
        combined_snr  : 1D np.array
            Stitched arrays, or None if something failed.
        """
        # 1) Load data from your 1D FITS
        fits_file = self.load_observation(epoch_num, spectra_num, "1D")
        if not fits_file:
            print(f"Failed to load observation for epoch={epoch_num}, spec={spectra_num}")
            return None, None, None
    
        data = fits_file.data
        wave_arrays  = np.array(data['wavelength'])[::-1]
        flux_arrays  = np.array(data['flux'])[::-1]
        uncertainty_arrays = np.array(data['uncertainty'])[::-1]
        blaze_arrays = np.array(data['blaze'])[::-1]
        blaze_errors_arrays = np.array(data['blaze_error'])[::-1]
    
        orders = len(wave_arrays)
        if orders == 0:
            print("No wavelength data found in the FITS. Aborting.")
            return None, None, None
    
        # 2) Collect wave/flux/snr for each order
        wave_list = []
        flux_list = []
        snr_list  = []
    
        n_pairs = orders // 2
        for order_idx in range(n_pairs):
            # We'll call the sky is [2*order_idx], object is [2*order_idx + 1]
            # Adjust if your data format is reversed
            wave_obj = wave_arrays[order_idx*2 + 1]
            flux_obj = flux_arrays[order_idx*2 + 1] / blaze_arrays[order_idx*2 + 1]
            wave_sky = wave_arrays[order_idx*2]
            flux_sky = flux_arrays[order_idx*2] / blaze_arrays[order_idx*2]
    
            if subtract_sky:
                tmp_flux = flux_obj - flux_sky
            else:
                tmp_flux = flux_obj
    
            # SNR from uncertainties + blaze
            unc_obj = uncertainty_arrays[order_idx*2 + 1] / blaze_arrays[order_idx*2 + 1]
            unc_sky = uncertainty_arrays[order_idx*2]     / blaze_arrays[order_idx*2]
            blaze_err_obj = blaze_errors_arrays[order_idx*2 + 1]
            blaze_err_sky = blaze_errors_arrays[order_idx*2]
            
            # Example combining total error in quadrature:
            flux_obj_blazeerr = (
                flux_arrays[order_idx*2 + 1]
                * blaze_err_obj
                / (blaze_arrays[order_idx*2 + 1]**2)
            )
            flux_sky_blazeerr = (
                flux_arrays[order_idx*2]
                * blaze_err_sky
                / (blaze_arrays[order_idx*2]**2)
            )
            total_uncert = np.sqrt(
                unc_obj**2
                + unc_sky**2
                + flux_obj_blazeerr**2
                + flux_sky_blazeerr**2
            )
            tmp_snr = tmp_flux / total_uncert
            
            wave_list.append(wave_obj)
            flux_list.append(tmp_flux)
            snr_list.append(tmp_snr)
    
        # 3) Flip order if needed (e.g. if short to long wave is reversed)
        wave_list = np.flip(wave_list)
        flux_list = np.flip(flux_list)
        snr_list  = np.flip(snr_list)
    
        # 4) Stitch them with your existing method
        combined_wave, combined_flux, combined_snr = self._stitch_spectra_by_snr(
            wave_list, flux_list, snr_list
        )
    
        return combined_wave, combined_flux, combined_snr

    def get_stitched_spectra2(self, epoch_num, spectra_num, subtract_sky=True, direct_snr_method=False,window_size = 20):
        """
        Similar to plot_stitched_spectra, but returns (wave, flux, snr)
        WITHOUT opening any figure or calling plt.show().
        
        Parameters
        ----------
        epoch_num : int or str
            Which epoch folder to load from.
        spectra_num : int or str
            Which spectrum folder inside that epoch.
        subtract_sky : bool
            If True, subtract sky from object in each pair.
            (We assume wave_arrays, flux_arrays come in pairs of (sky, obj).)
        direct_snr_method : bool
            If True, compute SNR using a direct method: for each flux point in the corrected flux,
            use a sliding window of 20 flux values and compute its robust standard deviation (with 3σ)
            via ut.robust_std, then SNR = flux / std.
            If False, compute SNR using the original method that propagates uncertainties and blaze errors.
        
        Returns
        -------
        combined_wave : 1D np.array
        combined_flux : 1D np.array
        combined_snr  : 1D np.array
            Stitched arrays, or None if something failed.
        """
        # 1) Load data from your 1D FITS
        fits_file = self.load_observation(epoch_num, spectra_num, "1D")
        if not fits_file:
            print(f"Failed to load observation for epoch={epoch_num}, spec={spectra_num}")
            return None, None, None
    
        data = fits_file.data
        # print(data['wavelength'][40:45,30:40])
        wave_arrays  = np.array(data['wavelength'])[::-1]
        # print(wave_arrays[40:45,300:310])
        flux_arrays  = np.array(data['flux'])[::-1]
        uncertainty_arrays = np.array(data['uncertainty'])[::-1]
        blaze_arrays = np.array(data['blaze'])[::-1]
        blaze_errors_arrays = np.array(data['blaze_error'])[::-1]
        # beyond this point the data is arranged in an ascending manner
    
        orders = len(wave_arrays)
        if orders == 0:
            print("No wavelength data found in the FITS. Aborting.")
            return None, None, None
    
        # 2) Collect wave/flux/snr for each order
        wave_list = []
        flux_list = []
        snr_list  = []
    
        n_pairs = orders // 2
        for order_idx in range(n_pairs):
            # We'll call the sky is [2*order_idx], object is [2*order_idx + 1]
            # Adjust if your data format is reversed
            wave_obj = wave_arrays[order_idx*2]
            flux_obj = flux_arrays[order_idx*2] / blaze_arrays[order_idx*2]
            wave_sky = wave_arrays[order_idx*2 + 1]
            flux_sky = flux_arrays[order_idx*2 + 1] / blaze_arrays[order_idx*2 + 1]
    
            if subtract_sky:
                tmp_flux = flux_obj - flux_sky
            else:
                tmp_flux = flux_obj
    
            # Calculate SNR
            if direct_snr_method:
                # New direct method: for each flux point, take a sliding window of 20 points,
                # compute its robust std with 3 sigma via ut.robust_std, then SNR = flux / std.
                tmp_snr = np.empty_like(tmp_flux)
                window_size = 4
                half_window = window_size // 2
                for i in range(len(tmp_flux)):
                    start = max(0, i - half_window)
                    end = min(len(tmp_flux), i + half_window)
                    window = tmp_flux[start:end]
                    sigma = ut.robust_std(window, 3)
                    # sigma = np.std(windows)
                    # Avoid division by zero
                    if sigma == 0:
                        tmp_snr[i] = 0
                    else:
                        tmp_snr[i] = ut.robust_mean(window,3) / sigma
                        if ut.robust_mean(window,3) < 0:
                            print(f'bad order! {order_idx}')
                        # tmp_snr[i] = np.mean(window) / sigma
                        # tmp_snr[i] = tmp_flux[i] / sigma
            else:
                # Original method: compute uncertainties + blaze errors
                unc_obj = uncertainty_arrays[order_idx*2 + 1] / blaze_arrays[order_idx*2 + 1]
                unc_sky = uncertainty_arrays[order_idx*2]     / blaze_arrays[order_idx*2]
                blaze_err_obj = blaze_errors_arrays[order_idx*2 + 1]
                blaze_err_sky = blaze_errors_arrays[order_idx*2]
                
                flux_obj_blazeerr = (
                    flux_arrays[order_idx*2 + 1]
                    * blaze_err_obj
                    / (blaze_arrays[order_idx*2 + 1]**2)
                )
                flux_sky_blazeerr = (
                    flux_arrays[order_idx*2]
                    * blaze_err_sky
                    / (blaze_arrays[order_idx*2]**2)
                )
                total_uncert = np.sqrt(
                    unc_obj**2
                    + unc_sky**2
                    + flux_obj_blazeerr**2
                    + flux_sky_blazeerr**2
                )
                tmp_snr = tmp_flux / total_uncert
    
            wave_list.append(wave_obj)
            flux_list.append(tmp_flux)
            snr_list.append(tmp_snr)
            # if order_idx == 20:
            #     # print(f'wave: {wave_list}')
            #     print(f'flux: {flux_list[0]}')
            plt.plot(wave_obj,tmp_flux)
    
        # 3) Flip order if needed (e.g. if short to long wave is reversed)
        # wave_list = np.flip(wave_list)
        # flux_list = np.flip(flux_list)
        # snr_list  = np.flip(snr_list)
    
        # 4) Stitch them with your existing method

        plt.show()
        combined_wave, combined_flux, combined_snr = self._stitch_spectra_by_snr(
            wave_list, flux_list, snr_list
        )
    
        return combined_wave, combined_flux, combined_snr

    def get_stitched_spectra3(self, epoch_num, spectra_num, subtract_sky=True, direct_snr_method=False, window_size=20, remove_low_blaze=True):
        """
        Similar to plot_stitched_spectra, but returns (wave, flux, snr)
        WITHOUT opening any figure or calling plt.show().
        
        Parameters
        ----------
        epoch_num : int or str
            Which epoch folder to load from.
        spectra_num : int or str
            Which spectrum folder inside that epoch.
        subtract_sky : bool
            If True, subtract sky from object in each pair.
            (We assume wave_arrays, flux_arrays come in pairs of (sky, obj).)
        direct_snr_method : bool
            If True, compute SNR using a direct method: for each flux point in the corrected flux,
            use a sliding window of `window_size` flux values and compute its robust standard deviation (with 3σ)
            via ut.robust_std, then SNR = robust_mean(window, 3) / std.
            If False, compute SNR using the original method that propagates uncertainties and blaze errors.
        window_size : int, optional
            Window size to use for the direct SNR method.
        remove_low_blaze : bool, optional
            If True (default), remove data points where either the object or sky blaze value is below 10%
            of its order maximum.
        
        Returns
        -------
        combined_wave : 1D np.array
            Combined wavelength array.
        combined_flux : 1D np.array
            Combined flux array.
        combined_snr : 1D np.array
            Combined SNR array.
            Stitched arrays, or None if something failed.
        """
        # 1) Load data from your 1D FITS
        fits_file = self.load_observation(epoch_num, spectra_num, "1D")
        if not fits_file:
            print(f"Failed to load observation for epoch={epoch_num}, spec={spectra_num}")
            return None, None, None
    
        data = fits_file.data
        # Reverse the order of the arrays (assuming they come in descending order)
        wave_arrays         = np.array(data['wavelength'])[::-1]
        flux_arrays         = np.array(data['flux'])[::-1]
        uncertainty_arrays  = np.array(data['uncertainty'])[::-1]
        blaze_arrays        = np.array(data['blaze'])[::-1]
        blaze_errors_arrays = np.array(data['blaze_error'])[::-1]
        # beyond this point the data is arranged in an ascending manner
    
        orders = len(wave_arrays)
        if orders == 0:
            print("No wavelength data found in the FITS. Aborting.")
            return None, None, None
    
        # 2) Collect wave/flux/snr for each order
        wave_list = []
        flux_list = []
        snr_list  = []
    
        n_pairs = orders // 2
        for order_idx in range(n_pairs):
            # Assume: sky is at index 2*order_idx, object at index 2*order_idx + 1.
            # Extract object and sky arrays:
            index_obj = order_idx * 2
            index_sky = order_idx * 2 + 1
            if self.star_name == 'WR17' and (epoch_num == 2 or epoch_num ==3):
                index_obj = order_idx * 2 + 1
                index_sky = order_idx * 2
    
            wave_obj = wave_arrays[index_obj]
            flux_obj = flux_arrays[index_obj] / blaze_arrays[index_obj]
            wave_sky = wave_arrays[index_sky]
            flux_sky = flux_arrays[index_sky] / blaze_arrays[index_sky]
    
            # If enabled, remove data points where either blaze value is below 10% of its max.
            if remove_low_blaze:
                max_blaze_obj = np.max(blaze_arrays[index_obj])
                max_blaze_sky = np.max(blaze_arrays[index_sky])
                # Create a mask: if either object or sky blaze is below 10% of its max, discard that point.
                mask = (blaze_arrays[index_obj] >= 0.1 * max_blaze_obj) & (blaze_arrays[index_sky] >= 0.1 * max_blaze_sky)
                wave_obj = wave_obj[mask]
                flux_obj = flux_obj[mask]
                wave_sky = wave_sky[mask]
                flux_sky = flux_sky[mask]
    
            # Subtract sky if requested.
            if subtract_sky:
                tmp_flux = flux_obj - flux_sky
            else:
                tmp_flux = flux_obj
    
            # Calculate SNR.
            if direct_snr_method:
                tmp_snr = np.empty_like(tmp_flux)
                half_window = window_size // 2
                for i in range(len(tmp_flux)):
                    start = max(0, i - half_window)
                    end = min(len(tmp_flux), i + half_window)
                    window = tmp_flux[start:end]
                    sigma = ut.robust_std(window, 3)
                    # Avoid division by zero
                    if sigma == 0:
                        tmp_snr[i] = 0
                    else:
                        tmp_snr[i] = ut.robust_mean(window, 3) / sigma
                        # tmp_snr[i] = ut.robust_mean(window, 3) 
                        # tmp_snr[i] = np.mean(window)/sigma
            else:
                unc_obj = uncertainty_arrays[index_sky] / blaze_arrays[index_sky]
                unc_sky = uncertainty_arrays[index_obj] / blaze_arrays[index_obj]
                blaze_err_obj = blaze_errors_arrays[index_sky]
                blaze_err_sky = blaze_errors_arrays[index_obj]
                
                flux_obj_blazeerr = (
                    flux_arrays[index_sky] * blaze_err_obj / (blaze_arrays[index_sky]**2)
                )
                flux_sky_blazeerr = (
                    flux_arrays[index_obj] * blaze_err_sky / (blaze_arrays[index_obj]**2)
                )
                total_uncert = np.sqrt(
                    unc_obj**2 + unc_sky**2 + flux_obj_blazeerr**2 + flux_sky_blazeerr**2
                )
                tmp_snr = tmp_flux / total_uncert
    
            wave_list.append(wave_obj)
            flux_list.append(tmp_flux)
            snr_list.append(tmp_snr)
            # plt.plot(wave_obj, tmp_flux)
    
        # plt.show()
        # 4) Stitch them with your existing method.
        combined_wave, combined_flux, combined_snr = self._stitch_spectra_by_snr(
            wave_list, flux_list, snr_list
        )
    
        return combined_wave, combined_flux, combined_snr

    def clean_bad_pixels(self, epoch=None, spectra_nums=None, backup=True, overwrite=True):
        """
        Cleans the normalized flux data for one or more epochs by identifying wavelengths
        (bad pixels) where the flux is non-positive in every spectrum (based solely on the
        original data), and replaces those bad pixel values with linear interpolation from
        neighboring good pixels.

        For any wavelength where at least one spectrum has a non-positive value but not
        all do, it prints the (wavelength, average flux) from the original data.

        The cleaned normalized flux for each spectrum is saved using self.save_property()
        under the property name 'clean_normalized_flux'.

        Parameters
        ----------
        epoch : int, str, list, or None
            The epoch identifier(s). If None, all epochs are processed.
        spectra_nums : list or None
            A list of spectra numbers to process for each epoch. If None, all spectra in that epoch are processed.
        backup : bool, optional
            Whether to back up existing data before overwriting (default True).
        overwrite : bool, optional
            Whether to overwrite existing saved data (default True).

        Returns
        -------
        None
        """
        import numpy as np

        # Determine the epochs to process.
        if epoch is None:
            epoch_list = self.get_all_epoch_numbers()
        elif not isinstance(epoch, list):
            epoch_list = [epoch]
        else:
            epoch_list = epoch

        for ep in epoch_list:
            # Determine spectra for this epoch.
            if spectra_nums is None:
                sp_list = self.get_all_spectra_in_epoch(ep)
            elif not isinstance(spectra_nums, list):
                sp_list = [spectra_nums]
            else:
                sp_list = spectra_nums

            # Load the original normalized flux data for each spectrum in this epoch.
            orig_norm_data = {}
            for sp in sp_list:
                try:
                    norm_data = self.load_property('normalized_flux', ep, sp)
                    if norm_data is None:
                        print(f"Normalized flux not found for epoch {ep}, spec {sp}. Skipping.")
                        continue
                    orig_norm_data[sp] = norm_data
                except Exception as e:
                    print(f"Error loading normalized flux for epoch {ep}, spec {sp}: {e}")
                    continue

            if not orig_norm_data:
                print(f"No normalized flux data available for epoch {ep}.")
                continue

            # Assume all spectra share the same wavelength grid.
            first_sp = list(orig_norm_data.keys())[0]
            wavelengths = np.array(orig_norm_data[first_sp]['wavelengths'])
            n_points = len(wavelengths)

            # Build a flux matrix using the original data.
            flux_matrix = []
            valid_specs = []
            for sp in sorted(orig_norm_data.keys()):
                flux = np.array(orig_norm_data[sp]['normalized_flux'])
                if len(flux) != n_points:
                    print(f"Warning: Spectrum {sp} has a different wavelength grid. Skipping.")
                    continue
                flux_matrix.append(flux)
                valid_specs.append(sp)
            if len(flux_matrix) == 0:
                print(f"No spectra with consistent wavelength grid found for epoch {ep}.")
                continue
            flux_matrix = np.array(flux_matrix)  # Shape: (M, N)

            # Create a "bad mask": for each wavelength index, mark it as bad if every spectrum has flux <= 0.
            bad_mask = np.all(flux_matrix <= 0, axis=0)

            # For indices where at least one spectrum is non-positive but not all are,
            # print the (wavelength, average flux) from the original data.
            for i in range(n_points):
                if np.any(flux_matrix[:, i] <= 0) and not bad_mask[i]:
                    avg_flux = np.mean(flux_matrix[:, i])
                    print(
                        f"Retaining pixel at wavelength {wavelengths[i]:.2f} with average flux {avg_flux:.2f} (not negative in all spectra)")

            # For each valid spectrum, interpolate to fix bad pixels using the original flux.
            for sp in valid_specs:
                try:
                    orig_flux = np.array(orig_norm_data[sp]['normalized_flux'])
                    good_idx = np.where(orig_flux > 0)[0]
                    if len(good_idx) < 2:
                        print(
                            f"Not enough good pixels for epoch {ep}, spec {sp} to interpolate. Skipping cleaning for this spectrum.")
                        continue
                    cleaned_flux = orig_flux.copy()
                    bad_idx = np.where(bad_mask)[0]
                    cleaned_flux[bad_idx] = np.interp(wavelengths[bad_idx],
                                                      wavelengths[good_idx],
                                                      orig_flux[good_idx])
                    clean_data = {'wavelengths': wavelengths, 'normalized_flux': cleaned_flux}
                    self.save_property('clean_normalized_flux', clean_data, ep, sp, overwrite=overwrite, backup=backup)
                    print(f"Cleaned normalized flux saved for epoch {ep}, spec {sp}.")
                except Exception as e:
                    print(f"Error cleaning spectrum for epoch {ep}, spec {sp}: {e}")
                    continue
