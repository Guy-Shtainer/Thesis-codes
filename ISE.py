import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, LassoSelector
from matplotlib.path import Path
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import os
import re
import threading
import utils as ut
from ObservationClass import ObservationManager as obsm
import argparse
import specs
import gc


def interactive_normalization(star, epoch_numbers, band='COMBINED', filter_func=None,
                              overwrite=False, backup=True, load_saved=False):
    """
    Interactive method to manually adjust points for normalization across multiple epochs.

    Parameters:
        star: The star object.
        epoch_numbers (list): List of epoch numbers to process.
        band (str): The band of the spectrum (default 'COMBINED').
        filter_func (callable or None): A function to filter points automatically. If None, all points are used.
        overwrite (bool): Whether to overwrite existing saved properties. Default is False.
        backup (bool): Whether to create a backup if overwriting. Default is True.
        load_saved (bool): Whether to load saved anchor points if available. Default is False.
    """
    import threading
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np
    import gc

    proceed_event = threading.Event()
    skip_to_next_star = False  # Flag to skip to next star without saving
    current_epoch_idx = 0  # To keep track of the current epoch index

    # Initialize variables to store data
    wavelength = None
    flux = None

    first_normalization = True

    # Variables to store the shared anchor points across epochs
    selected_wavelengths_tmp = []
    press_event = {'x': None, 'y': None, 'button': None}

    # Initialize plot with three subplots
    fig, (ax1, ax_mid, ax2) = plt.subplots(3, 1, figsize=(10, 9))
    plt.subplots_adjust(bottom=0.2, hspace=0.4)

    # Load saved anchor points once at the beginning if load_saved is True
    if load_saved:
        try:
            epoch_number = epoch_numbers[0]  # Use the first epoch to load the anchor points
            saved_wavelengths = star.load_property('norm_anchor_wavelengths', epoch_number, band)
            if saved_wavelengths is not None and len(saved_wavelengths) > 0:
                selected_wavelengths_tmp = saved_wavelengths.tolist()
                print(f"Loaded saved anchor points.")
            else:
                print(f"No saved anchor points found.")
        except FileNotFoundError:
            print(f"No saved anchor points file.")
        except Exception as e:
            print(f"Error loading saved anchor points: {e}")

    def load_data():
        nonlocal wavelength, flux
        epoch_number = epoch_numbers[current_epoch_idx]
        # Load observation for the current epoch and band
        fits_file = star.load_observation(epoch_number, band)
        wavelength = fits_file.data['WAVE'][0]
        flux = fits_file.data['FLUX'][0]

    def update_plot(new_epoch=False):
        # Update the main plot with all data points and selected points

        # Save current axes limits
        if ax1.has_data():
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()
        else:
            xlim = None
            ylim = None

        ax1.clear()

        # Plot all data points on ax1
        ax1.plot(wavelength, flux, '.', color='gray', markersize=2, label='Data')

        # Get fluxes at selected wavelengths
        if selected_wavelengths_tmp:
            selected_fluxes_current_epoch = np.interp(selected_wavelengths_tmp, wavelength, flux)
        else:
            selected_fluxes_current_epoch = []

        ax1.plot(selected_wavelengths_tmp, selected_fluxes_current_epoch, 'o', color='red', markersize=5, label='Selected Points')

        ax1.set_ylabel('Flux')
        ax1.set_title(f'Interactive Normalization - Star: {star.star_name}, Epoch: {epoch_numbers[current_epoch_idx]}, Band: {band}')
        ax1.legend()

        # Restore axes limits if they were set
        if xlim is not None and ylim is not None:
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
        else:
            ax1.relim()
            ax1.autoscale_view()

        # Plot data and interpolated flux on ax_mid
        plot_interpolated_flux()

        # Automatically update the normalization plot
        plot_normalized_flux(new_epoch)

        fig.canvas.draw_idle()

    def plot_interpolated_flux():
        # Save current axes limits
        if ax_mid.has_data():
            xlim = ax_mid.get_xlim()
            ylim = ax_mid.get_ylim()
        else:
            xlim = None
            ylim = None

        ax_mid.clear()

        # Always plot the data points
        ax_mid.plot(wavelength, flux, '.', color='gray', markersize=2, label='Data')

        if len(selected_wavelengths_tmp) >= 2:
            selected_fluxes_current_epoch = np.interp(selected_wavelengths_tmp, wavelength, flux)

            # Perform linear interpolation between selected points
            sorted_indices = np.argsort(selected_wavelengths_tmp)
            selected_wavelengths_sorted = np.array(selected_wavelengths_tmp)[sorted_indices]
            selected_fluxes_sorted = np.array(selected_fluxes_current_epoch)[sorted_indices]

            interpolated_flux = np.interp(wavelength, selected_wavelengths_sorted, selected_fluxes_sorted)

            # Plot interpolated flux
            ax_mid.plot(wavelength, interpolated_flux, '-', color='green', label='Interpolated Flux')

            # Plot selected points
            ax_mid.plot(selected_wavelengths_tmp, selected_fluxes_current_epoch, 'o', color='red', markersize=5, label='Selected Points')

            ax_mid.set_title(f'Interpolation - Epoch: {epoch_numbers[current_epoch_idx]}')
        else:
            ax_mid.set_title("Not enough points selected for interpolation.")

        ax_mid.set_ylabel('Flux')
        ax_mid.legend()

        # Restore axes limits if they were set
        if xlim is not None and ylim is not None:
            ax_mid.set_xlim(xlim)
            ax_mid.set_ylim(ylim)
        else:
            ax_mid.relim()
            ax_mid.autoscale_view()

    def plot_normalized_flux(new_epoch=False):
        nonlocal first_normalization
        # Save current axes limits
        if ax2.has_data():
            xlim = ax2.get_xlim()
            ylim = ax2.get_ylim()
        else:
            xlim = None
            ylim = None

        ax2.clear()

        if len(selected_wavelengths_tmp) >= 2:
            selected_fluxes_current_epoch = np.interp(selected_wavelengths_tmp, wavelength, flux)

            # Perform linear interpolation between selected points
            sorted_indices = np.argsort(selected_wavelengths_tmp)
            selected_wavelengths_sorted = np.array(selected_wavelengths_tmp)[sorted_indices]
            selected_fluxes_sorted = np.array(selected_fluxes_current_epoch)[sorted_indices]

            interpolated_flux = np.interp(wavelength, selected_wavelengths_sorted, selected_fluxes_sorted)
            normalized_flux = flux / interpolated_flux

            ax2.plot(wavelength, normalized_flux, '-', color='blue', label='Normalized Flux')
            ax2.axhline(y=1, color='red', linestyle='--', label='y=1 (Interpolated Flux)')
            ax2.set_title(f'Normalized Flux - Epoch: {epoch_numbers[current_epoch_idx]}')
        else:
            interpolated_flux = np.interp(wavelength, np.take(wavelength, [50, 1500, -50]), np.take(flux, [50, 1500, -50]))
            normalized_flux = flux / interpolated_flux
            ax2.plot(wavelength, normalized_flux, '-', color='blue', label='Normalized Flux')
            ax2.axhline(y=1, color='red', linestyle='--', label='y=1 (Interpolated Flux)')
            ax2.set_title("Not enough points selected for interpolation.")

        ax2.set_xlabel('Wavelength')
        ax2.set_ylabel('Normalized Flux')
        ax2.legend()

        # Restore axes limits if they were set
        if xlim is not None and ylim is not None:
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
        else:
            ax2.relim()
            ax2.autoscale_view()

    def onpress(event):
        if event.inaxes != ax1:
            return
        press_event['x'] = event.xdata
        press_event['y'] = event.ydata
        press_event['button'] = event.button

    def onrelease(event):
        if event.inaxes != ax1:
            return
        dx = event.xdata - press_event['x']
        dy = event.ydata - press_event['y']
        dist = np.hypot(dx, dy)
        movement_threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.01
        if dist < movement_threshold:
            onclick(press_event)

    def onclick(press_event):
        if press_event['x'] is None or press_event['y'] is None:
            return
    
        # Define thresholds based on current axes limits
        x_threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.05
        y_threshold = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05
    
        if press_event['button'] == 1:  # Left click to add points
            x_diff = np.abs(wavelength - press_event['x'])
            y_diff = np.abs(flux - press_event['y'])
            within_threshold = (x_diff < x_threshold) & (y_diff < y_threshold)
            if not np.any(within_threshold):
                return
            candidate_wavelengths = wavelength[within_threshold]
            candidate_fluxes = flux[within_threshold]
            distances = np.hypot(candidate_wavelengths - press_event['x'], candidate_fluxes - press_event['y'])
            min_idx = np.argmin(distances)
            idx = np.where(within_threshold)[0][min_idx]
            wl = wavelength[idx]
            if wl not in selected_wavelengths_tmp:
                selected_wavelengths_tmp.append(wl)
                selected_wavelengths_tmp.sort()
                update_plot()
        elif press_event['button'] == 3:  # Right click to delete points
            if not selected_wavelengths_tmp:
                return
            selected_wavelengths = np.array(selected_wavelengths_tmp)
            selected_fluxes = np.interp(selected_wavelengths, wavelength, flux)
            x_diff = np.abs(selected_wavelengths - press_event['x'])
            y_diff = np.abs(selected_fluxes - press_event['y'])
            within_threshold = (x_diff < x_threshold) & (y_diff < y_threshold)
            if not np.any(within_threshold):
                return
            candidate_wavelengths = selected_wavelengths[within_threshold]
            candidate_fluxes = selected_fluxes[within_threshold]
            distances = np.hypot(candidate_wavelengths - press_event['x'], candidate_fluxes - press_event['y'])
            min_idx = np.argmin(distances)
            idx = np.where(within_threshold)[0][min_idx]
            del selected_wavelengths_tmp[idx]
            update_plot()

    fig.canvas.mpl_connect('button_press_event', onpress)
    fig.canvas.mpl_connect('button_release_event', onrelease)

    ax_next_star = plt.axes([0.48, 0.02, 0.1, 0.05])
    btn_next_star = Button(ax_next_star, 'Next Star')

    def next_star(event):
        nonlocal skip_to_next_star
        skip_to_next_star = True
        proceed_event.set()
        plt.close(fig)
        print("Skipping to next star...")

    btn_next_star.on_clicked(next_star)

    ax_next_epoch = plt.axes([0.7, 0.02, 0.1, 0.05])
    btn_next_epoch = Button(ax_next_epoch, 'Next Epoch')

    def next_epoch(event):
        nonlocal current_epoch_idx
        if current_epoch_idx < len(epoch_numbers) - 1:
            current_epoch_idx += 1
            load_data()
            update_plot(True)
        else:
            print('Already at the last epoch.')

    btn_next_epoch.on_clicked(next_epoch)

    ax_prev_epoch = plt.axes([0.59, 0.02, 0.1, 0.05])
    btn_prev_epoch = Button(ax_prev_epoch, 'Previous Epoch')

    def prev_epoch(event):
        nonlocal current_epoch_idx
        if current_epoch_idx > 0:
            current_epoch_idx -= 1
            load_data()
            update_plot(True)
        else:
            print('Already at the first epoch.')

    btn_prev_epoch.on_clicked(prev_epoch)

    ax_finish = plt.axes([0.81, 0.02, 0.1, 0.05])
    btn_finish = Button(ax_finish, 'Finish and Save')

    def finish(event):
        proceed_event.set()
        plt.close(fig)
        print('Finished normalization.')

        for epoch_number in epoch_numbers:
            if len(selected_wavelengths_tmp) < 2:
                print(f"Not enough points selected for interpolation in epoch {epoch_number}. Skipping.")
                continue
            fits_file = star.load_observation(epoch_number, band)
            wavelength_epoch = fits_file.data['WAVE'][0]
            flux_epoch = fits_file.data['FLUX'][0]
            selected_fluxes_epoch = np.interp(selected_wavelengths_tmp, wavelength_epoch, flux_epoch)
            sorted_indices = np.argsort(selected_wavelengths_tmp)
            selected_wavelengths_epoch = np.array(selected_wavelengths_tmp)[sorted_indices]
            selected_fluxes_epoch = selected_fluxes_epoch[sorted_indices]
            interpolated_flux_epoch = np.interp(wavelength_epoch, selected_wavelengths_epoch, selected_fluxes_epoch)
            normalized_flux = flux_epoch / interpolated_flux_epoch
            star.save_property('norm_anchor_wavelengths', selected_wavelengths_epoch, epoch_number, band, overwrite=overwrite, backup=backup)
            star.save_property('normalized_flux', {'wavelengths': wavelength_epoch, 'normalized_flux': normalized_flux}, epoch_number, band, overwrite=overwrite, backup=backup)
            star.save_property('interpolated_flux', {'wavelengths': wavelength_epoch,  'interpolated_flux': interpolated_flux_epoch}, epoch_number, band, overwrite=overwrite, backup=backup)
            print(f"Saved normalization results for epoch {epoch_number}")

    btn_finish.on_clicked(finish)

    load_data()
    update_plot()
    plt.show()

    while not proceed_event.is_set():
        proceed_event.wait()
        gc.collect()

    print('Finished processing all epochs for this star.')

    return skip_to_next_star



def filter_f(wavelength, flux, batch_size=6, big_batch_size=10):
    flux_std = np.std(flux)
    flux_std = ut.robust_std(flux, 1)
    
    print(f'The whole std flux is {flux_std}')

    # Step 3: Divide the flux and wavelength arrays into batches of 'batch_size'
    num_points = len(flux)
    num_batches = int(np.ceil(num_points / batch_size))

    test_flux = []
    test_wavelengths = []
    kept_flux_means = []
    kept_wavelength_means = []
    big_flux_batch_std_list = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_points)
        flux_batch = flux[start_idx:end_idx]
        big_flux_batch_std = np.min([ut.robust_std(flux[i:i + batch_size], 2) for i in range(int(np.maximum(0, start_idx - big_batch_size * batch_size / 2)), int(end_idx + big_batch_size * batch_size / 2), batch_size)])
        wavelength_batch = wavelength[start_idx:end_idx]

        # Calculate the standard deviation of the current batch
        batch_std = np.std(flux_batch)

        # test
        test_flux.append(ut.double_robust_mean(flux_batch))
        test_wavelengths.append(np.mean(wavelength_batch))

        # Step 4: Discard batches where batch_std > flux_std
        if batch_std <= big_flux_batch_std:
            robust_mean_flux = ut.double_robust_mean(flux_batch)
            mean_wavelength = np.mean(wavelength_batch)
            kept_flux_means.append(robust_mean_flux)
            kept_wavelength_means.append(mean_wavelength)
            big_flux_batch_std_list.append(big_flux_batch_std)
        else:
            continue

    kept_flux_means = np.array(kept_flux_means)
    kept_wavelength_means = np.array(kept_wavelength_means)

    flux_diffs = np.diff(kept_flux_means)
    big_change_indices = np.where(np.abs(flux_diffs) > big_flux_batch_std_list[:-1])[0]
    small_change_indices = np.where(np.abs(flux_diffs) < big_flux_batch_std_list[:-1])[0]
    print(f'Number is close points is: {len(small_change_indices)}')
    
    kept_wavelength_means = kept_wavelength_means[small_change_indices]
    kept_flux_means = kept_flux_means[small_change_indices]
    big_flux_batch_std_list = np.array(big_flux_batch_std_list)[small_change_indices]

    flux_diffs = np.diff(kept_flux_means)
    big_change_indices = np.where(np.abs(flux_diffs) > big_flux_batch_std_list[:-1])[0]
    small_change_indices = np.where(np.abs(flux_diffs) < big_flux_batch_std_list[:-1])[0]

    if len(small_change_indices) >= 22:
        kept_wavelength_means = kept_wavelength_means[small_change_indices]
        kept_flux_means = kept_flux_means[small_change_indices]
        big_flux_batch_std_list = np.array(big_flux_batch_std_list)[small_change_indices]

    print(f'Number is close points is: {len(small_change_indices)}')

    points_to_keep = np.ones(len(kept_flux_means), dtype=bool)

    final_wavelengths = kept_wavelength_means[points_to_keep]
    final_fluxes = kept_flux_means[points_to_keep]

    return final_wavelengths, final_fluxes

def main():
    import argparse
    import re
    import specs
    from ObservationClass import ObservationManager as obsm

    parser = argparse.ArgumentParser(description="Interactive normalization of spectra.")
    parser.add_argument('--star_names', nargs='+', default=None, help='List of star names to process')
    parser.add_argument('--overwrite_flag', action='store_true', default=False, help='Flag to overwrite existing files')
    parser.add_argument('--backup_flag', action='store_true', default=True, help='Flag to create backups before overwriting')
    parser.add_argument('--skip_flag', action='store_true', default=False, help='Flag to skip if results file already exist')
    parser.add_argument('--filter_flag', action='store_true', default=False, help='Flag to use the filtering function')
    parser.add_argument('--load_saved_flag', action='store_true', default=False, help='Flag to load saved anchor points if available')
    args = parser.parse_args()

    star_names = specs.star_names if args.star_names is None else args.star_names
    obs_file_names = specs.obs_file_names
    overwrite_flag = args.overwrite_flag
    backup_flag = args.backup_flag
    skip_flag = args.skip_flag
    load_saved_flag = args.load_saved_flag
    obs = obsm()
    filter_func = filter_f if args.filter_flag else None

    for star_name in star_names:
        print(f"Processing star: {star_name}")
        if star_name not in obs_file_names:
            print(f"Star {star_name} not found in obs_file_names.")
            continue

        star = obs.load_star_instance(star_name)
        epochs_dict = obs_file_names[star_name]
        epoch_nums = [int(re.findall(r'\d+\.\d+|\d+', epoch)[0]) for epoch in epochs_dict.keys()]
        band = 'COMBINED'  # Use the combined band

        # Skip processing if skip_flag is set and results already exist
        if skip_flag:
            all_exist = True
            for epoch_num in epoch_nums:
                try:
                    result = star.load_property('norm_anchor_wavelengths', epoch_num, band)
                    if result is None:
                        all_exist = False
                        break
                except FileNotFoundError:
                    all_exist = False
                    break
            if all_exist:
                print(f"Skipping star: {star_name} as results already exist for all epochs.")
                continue

        print(f"Starting interactive normalization for {star_name}")
        skip_to_next_star = interactive_normalization(
            star, epoch_nums, band=band, filter_func=filter_func,
            overwrite=overwrite_flag, backup=backup_flag, load_saved=load_saved_flag
        )
        
        if skip_to_next_star:
            print(f"Skipped saving and moving to next star for {star_name}")
            continue

        print(f"Finished processing {star_name}")

    print("All stars have been processed.")


if __name__ == "__main__":
    main()