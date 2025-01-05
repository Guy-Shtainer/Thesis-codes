#!/usr/bin/env python3
"""
Simulations class to create mock SB2 binaries (by Tomer Shenar, with modifications).
Encapsulates data generation (run_simulation) and plotting (plot_saved_simulation).

Folder Structure:
  simulations/SNR<value>/
    Atemp.txt, Btemp.txt, obs_0, obs_1, ... etc.
Usage Example:
  python this_script.py
    -> Generates data in simulations/SNR100 (by default)
    -> Plots those files with a Next/Previous interface.
"""

import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, convolve


###############################
# Helper Function to Load Data
###############################
def load_data(file_path):
    """
    Loads two-column data (e.g., wavelength & flux) from the given file.
    Returns (wavelength_array, flux_array) or (None, None) if invalid.
    """
    if not os.path.isfile(file_path):
        print(f"[WARNING] File '{file_path}' does not exist.")
        return None, None

    wavelengths, fluxes = [], []
    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue  # skip empty lines
            parts = stripped.split()
            if len(parts) < 2:
                continue  # skip lines that don't have at least 2 columns
            try:
                w_val = float(parts[0])
                f_val = float(parts[1])
                wavelengths.append(w_val)
                fluxes.append(f_val)
            except ValueError:
                # skip lines that can't be parsed as floats
                continue

    if not wavelengths or not fluxes:
        print(f"[WARNING] No valid data found in '{file_path}'.")
        return None, None

    return np.array(wavelengths), np.array(fluxes)


#################################
# Main Simulations Class
#################################
class Simulations:
    """
    A class that creates mock SB2 (or SB1) data in a folder:
       simulations/SNR<value>/
    and also provides a method (plot_saved_simulation) that
    lets you browse these files with Previous/Next buttons.
    """

    def __init__(
        self,
        T0=0.0,             # Reference time
        P=473.45,           # Period
        e=0.0,              # Eccentricity
        omega_deg=90.0,     # Omega in degrees
        K1=0.0,
        K2=135.0,
        Gamma=0.0,
        lamB=4000.0,
        lamR=5000.0,
        Resolution=53000,
        SamplingFac=3,
        S2N=100.0,
        Q=0.0,              # Flux ratio (F2 / (F1+F2))
        specnum=30,         # Number of epochs
        NPeriods=1.0,       # Max # of periods to randomly sample MJD
        NebularLines=False,
        Nebwaves=None,
        MaskPath="./Data/LMC WC Temaplates/lmc-wc_14-10_line.txt",
        MaskPath2="./Data/LMC WC Temaplates/lmc-wc_14-10_line.txt",
    ):
        """
        Initialize the simulation parameters.
        """
        self.T0 = T0
        self.P = P
        self.e = e
        self.omega = np.radians(omega_deg)  # Convert degrees -> radians
        self.K1 = K1
        self.K2 = K2
        self.Gamma = Gamma

        self.lamB = lamB
        self.lamR = lamR
        self.Resolution = Resolution
        self.SamplingFac = SamplingFac
        self.S2N = S2N
        self.Q = Q
        self.specnum = specnum
        self.NPeriods = NPeriods

        self.NebularLines = NebularLines
        if Nebwaves is None:
            Nebwaves = []
        self.Nebwaves = np.array(Nebwaves)

        self.MaskPath = MaskPath
        self.MaskPath2 = MaskPath2

        # Where the simulation data will be saved
        self.top_folder = "simulations"
        self.snr_folder = f"SNR{int(self.S2N)}"
        self.output_dir = os.path.join(self.top_folder, self.snr_folder)

        # Speed of light in km/s
        self.clight = 2.9979e5

        # Precompute e-factor
        self.efac = np.sqrt((1.0 + self.e) / (1.0 - self.e))

    ##########################
    # 1) Simulation Routines
    ##########################
    def kepler(self, E, M):
        """
        Solve Kepler's equation E - e sin(E) = M for E using iteration.
        """
        E2 = (M - self.e*(E*np.cos(E) - np.sin(E))) / (1.0 - self.e*np.cos(E))
        eps = np.abs(E2 - E)
        if np.all(eps < 1e-10):
            return E2
        else:
            return self.kepler(E2, M)

    def v1v2(self, nu):
        """
        Compute the radial velocities of two components for true anomaly nu.
        v1 = Gamma + K1*(cos(omega+nu) + e cos(omega))
        v2 = Gamma + K2*(cos(pi+omega+nu) + e cos(pi+omega))
        """
        v1 = self.Gamma + self.K1*(np.cos(self.omega + nu) + self.e*np.cos(self.omega))
        v2 = self.Gamma + self.K2*(np.cos(np.pi + self.omega + nu) + self.e*np.cos(np.pi + self.omega))
        return v1, v2

    def run_simulation(self, overwrite=False):
        """
        Main method that creates mock SB2 data in "simulations/SNR<value>/"
        based on the simulation parameters.
    
        Parameters:
            overwrite (bool): If True, overwrites existing simulation files. Defaults to False.
        """
        # Check if simulation files already exist
        if os.path.exists(self.output_dir) and not overwrite:
            print(f"[INFO] Simulation files already exist in '{self.output_dir}'. Set overwrite=True to rerun the simulation.")
            return
    
        # 1) Ensure directory is empty
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
        # Remove old files if present
        for f in glob.glob(os.path.join(self.output_dir, "obs*")):
            os.remove(f)
        for f in glob.glob(os.path.join(self.output_dir, "ObsDat.t*")):
            os.remove(f)
    
        # 2) Prepare wave grid
        lammid = 0.5 * (self.lamB + self.lamR)
        DlamRes = lammid / self.Resolution
        Dlam = lammid / self.Resolution / self.SamplingFac
        wavegrid = np.arange(self.lamB, self.lamR, Dlam)
    
        # Prepare convolution kernel
        stdConv = DlamRes / Dlam / np.sqrt(2 * np.log(2)) / 2.0
        kernel = Gaussian1DKernel(stddev=stdConv)
    
        # Load & convolve mask
        mask1 = np.loadtxt(self.MaskPath)
        mask2 = np.loadtxt(self.MaskPath2)
    
        # Slight jitter to avoid interpolation corner cases
        Waves1 = mask1[:, 0] + np.random.normal(0, 1e-10, len(mask1))
        Waves2 = mask2[:, 0] + np.random.normal(0, 1e-10, len(mask2))
    
        # Interpolate
        Mask_interp = interp1d(Waves1, mask1[:, 1], bounds_error=False, fill_value=1., kind='cubic')(wavegrid)
        Mask2_interp = interp1d(Waves2, mask2[:, 1], bounds_error=False, fill_value=1., kind='cubic')(wavegrid)
    
        # Convolve with kernel
        Mask = convolve(Mask_interp, kernel, normalize_kernel=True, boundary='extend')
        Mask2 = convolve(Mask2_interp, kernel, normalize_kernel=True, boundary='extend')
    
        # Save Atemp/Btemp
        np.savetxt(os.path.join(self.output_dir, "Atemp.txt"), np.c_[wavegrid, Mask])
        np.savetxt(os.path.join(self.output_dir, "Btemp.txt"), np.c_[wavegrid, Mask2])
    
        # Possibly add nebular lines
        if self.NebularLines and len(self.Nebwaves) > 0:
            Mask3_pre = np.zeros_like(wavegrid)
            IndsNebWaves = [np.argmin(np.abs(wavegrid - wv)) for wv in self.Nebwaves]
            for idx in IndsNebWaves:
                Mask3_pre[idx] += 1.0
            Mask3 = convolve(Mask3_pre, kernel, normalize_kernel=True, boundary='extend')
        else:
            Mask3 = np.zeros_like(wavegrid)
    
        # 3) Generate epochs
        obs_dat_path = os.path.join(self.output_dir, "ObsDat.txt")
        with open(obs_dat_path, "w") as phasesfile:
            phasesfile.write("MJD obsname\n")
    
            sig = 1.0 / self.S2N
    
            for i in range(self.specnum):
                # Random MJD in [0, NPeriods*P]
                MJD = random.uniform(0.0, self.NPeriods) * self.P
                phase = (MJD - self.T0) / self.P - int((MJD - self.T0) / self.P)
                M = 2.0 * np.pi * phase
    
                # Solve Kepler -> E -> nu
                E = self.kepler(1.0, M)
                nu = 2.0 * np.arctan(self.efac * np.tan(0.5 * E))
    
                # RV shifts
                v1, v2 = self.v1v2(nu)
                Facshift1 = np.sqrt((1.0 + v1 / self.clight) / (1.0 - v1 / self.clight))
                Facshift2 = np.sqrt((1.0 + v2 / self.clight) / (1.0 - v2 / self.clight))
    
                # Shift masks
                Maskshift1 = interp1d(wavegrid * Facshift1, Mask, bounds_error=False, fill_value=1., kind='cubic')(wavegrid)
                Maskshift2 = interp1d(wavegrid * Facshift2, Mask2, bounds_error=False, fill_value=1., kind='cubic')(wavegrid)
    
                # Combine flux
                combined_flux = (1.0 - self.Q) * Maskshift1 + self.Q * Maskshift2 + Mask3
    
                # Add noise
                noisy_flux = combined_flux + np.random.normal(0.0, sig, len(wavegrid))
    
                obsname = os.path.join(self.output_dir, f"obs_{i}")
                np.savetxt(obsname, np.c_[wavegrid, noisy_flux])
                phasesfile.write(f"{MJD} {obsname}\n")
    
        print(f"[INFO] Simulation complete. Files saved to '{self.output_dir}'.")

    ##############################
    # 2) Plotting Method
    ##############################
    def plot_saved_simulation(self, snr_override=None):
        """
        Plots the saved data from:
            simulations/SNR<snr_value>/
        using a Next/Previous button interface.

        :param snr_override: If provided, override the default S2N
                             (e.g., pass 1000 to plot from 'simulations/SNR1000').
        """
        if snr_override is None:
            folder_snr = int(self.S2N)
        else:
            folder_snr = int(snr_override)

        folder_path = os.path.join("simulations", f"SNR{folder_snr}")

        # Validate folder
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"[ERROR] Folder not found: {folder_path}")

        # Gather files in sorted order
        all_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ])
        if not all_files:
            raise FileNotFoundError(f"No data files found in '{folder_path}'.")

        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(bottom=0.2)

        current_index = 0

        # Plot function
        def update_plot():
            ax.clear()
            current_file = os.path.join(folder_path, all_files[current_index])
            wave, flux = load_data(current_file)
            if wave is not None and flux is not None:
                ax.plot(wave, flux, '-', color='blue')
                title_text = (
                    f"File: {all_files[current_index]}  "
                    f"({current_index+1}/{len(all_files)})"
                )
                ax.set_title(title_text)
                ax.set_xlabel("Wavelength")
                ax.set_ylabel("Flux")
                ax.grid(True)
            else:
                ax.set_title(f"No valid data in {all_files[current_index]}")
            fig.canvas.draw_idle()

        update_plot()

        # Buttons
        button_width = 0.15
        button_height = 0.075
        spacing = 0.02

        left1 = 0.3
        left2 = left1 + button_width + spacing

        ax_prev = fig.add_axes([left1, 0.05, button_width, button_height])
        ax_next = fig.add_axes([left2, 0.05, button_width, button_height])

        btn_prev = Button(ax_prev, "Previous")
        btn_next = Button(ax_next, "Next")

        def on_prev(event):
            nonlocal current_index
            current_index = (current_index - 1) % len(all_files)
            update_plot()

        def on_next(event):
            nonlocal current_index
            current_index = (current_index + 1) % len(all_files)
            update_plot()

        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

        plt.show()

    ##############################
    # 3) CCF Method Integration
    ##############################
    def run_CCF(self, template_wave, template_flux, CrossCorRangeA, observation_folder=None):
        """
        Compute radial velocities using CCFclass on the simulated observations.
        
        :param template_wave: Wavelength array of the template spectrum.
        :param template_flux: Flux array of the template spectrum.
        :param observation_folder: Folder containing the simulated observation files.
                                   If None, uses the default simulation output folder.
        """
        if observation_folder is None:
            observation_folder = self.output_dir

        if not os.path.isdir(observation_folder):
            raise FileNotFoundError(f"[ERROR] Observation folder not found: {observation_folder}")

        # Initialize CCFclass
        from CCF import CCFclass
        ccf = CCFclass(CrossCorRangeA=CrossCorRangeA)

        # Gather observation files
        observation_files = sorted(
            f for f in os.listdir(observation_folder)
            if os.path.isfile(os.path.join(observation_folder, f)) and f.startswith("obs_")
        )

        if not observation_files:
            raise FileNotFoundError(f"No observation files found in '{observation_folder}'.")

        results = []

        # Process each observation
        for obs_file in observation_files:
            obs_path = os.path.join(observation_folder, obs_file)
            obs_wave, obs_flux = load_data(obs_path)
            if obs_wave is None or obs_flux is None:
                print(f"[WARNING] Skipping invalid file: {obs_file}")
                continue

            # Compute RV and error
            RV, RV_error = ccf.compute_RV(obs_wave, obs_flux, template_wave, template_flux)
            # results.append([obs_file, float(RV), float(RV_error)])
            results.append([RV, RV_error])
            # print(f"[INFO] File: {obs_file}, RV: {RV:.2f} km/s, RV Error: {RV_error:.2f} km/s")

        return results

    ##############################
    # 4) File Loading Method
    ##############################
    def load_two_column_file(self, file_path):
        """
        Load a file with two columns (wavelength and flux) into NumPy arrays.

        :param file_path: Path to the file containing two columns of data.
        :return: Two NumPy arrays: (wavelength, flux). Returns (None, None) if an error occurs.
        """
        try:
            data = np.loadtxt(file_path)
            if data.shape[1] != 2:
                print(f"[WARNING] File does not have exactly two columns: {file_path}")
                return None, None
            return data[:, 0], data[:, 1]
        except Exception as e:
            print(f"[ERROR] Failed to load file: {file_path}. Error: {e}")
            return None, None


######################################
# Standalone usage example (demo)
######################################
def main():
    """
    1) Create a Simulations instance with default SNR=100.
    2) Run the simulation -> saves into 'simulations/SNR100'.
    3) Then plot from that folder with the built-in method.
    """
    sim = Simulations(S2N=100)
    sim.run_simulation()
    sim.plot_saved_simulation()  # or sim.plot_saved_simulation(snr_override=100)


if __name__ == "__main__":
    main()


    
