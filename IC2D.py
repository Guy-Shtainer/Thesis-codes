import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import argparse
import re
from ObservationClass import ObservationManager as obsm
import specs
import TwoDImage as p2D
import utils as ut

def clean_flux_and_normalize_interactive(
        image_data,
        wavelengths_2D,
        external_normalized_flux_band,
        external_wavelengths_band,
        anchor_points,
        band,
        star_name,
        epoch_num,
        load_saved_flag,
        bottom_spacial=None,
        top_spacial=None,
        include_spacial=None
    ):

    if load_saved_flag:
        saved_data = star.load_property('clean_normalized_flux',epoch_num,band)
        if saved_data:
            include_spacial = saved_data['included_spacial_coords']
            anchor_points = saved_data['anchor points']
    if bottom_spacial is None or top_spacial is None:
        if band == 'NIR':
            bottom_spacial, top_spacial = (23, 51)
        else:
            bottom_spacial, top_spacial = (21, 76)
    print(f"The top limit is: {top_spacial}, and the bottom limit is: {bottom_spacial}")

    if include_spacial is None:
        include_spacial = (bottom_spacial, top_spacial)

    # Results dictionary
    results = {
        'normalized_summed_flux_resampled': None,
        'wavelengths_2D': wavelengths_2D
    }

    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.25, wspace=0.4, hspace=0.6)

    # Initially draw full image with current bottom/top and include range
    ax_image = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    p2D.Plot2DImage_for_cleaning(
        image_data,
        wavelengths_2D,
        band,
        bottom_spacial,
        top_spacial,
        include_spacial[0],
        include_spacial[1],
        title=f"2D Flux Image for {star_name}, Band: {band}, Epoch: {epoch_num}",
        ValMin=-600,
        ValMax=600,
        ax=ax_image
    )
    ax_image.text(0.5, 1.05, "Adjust sliders below to select star region",
                  transform=ax_image.transAxes, ha='center', va='bottom', fontsize=10)

    # Subplots
    ax_summed_vertical = plt.subplot2grid((3, 2), (0, 1))
    ax_summed_horizontal = plt.subplot2grid((3, 2), (1, 1))
    ax_norm = plt.subplot2grid((3, 2), (2, 0))
    ax_diff = plt.subplot2grid((3, 2), (2, 1))

    ax_norm.set_ylim(-3, 5)
    ax_diff.set_ylim(-3, 5)
    ax_norm.set_xlim(np.min(wavelengths_2D)-10, np.max(wavelengths_2D)+10)
    ax_diff.set_xlim(np.min(wavelengths_2D)-10, np.max(wavelengths_2D)+10)

    # Slider and Button axes
    slider_ax_start = plt.axes([0.1, 0.14, 0.35, 0.03])
    slider_ax_end = plt.axes([0.55, 0.14, 0.35, 0.03])
    slider_ax_bottom = plt.axes([0.1, 0.07, 0.35, 0.03])
    slider_ax_top = plt.axes([0.55, 0.07, 0.35, 0.03])

    prev_star_ax = plt.axes([0.05, 0.02, 0.1, 0.05])
    next_star_ax = plt.axes([0.17, 0.02, 0.1, 0.05])
    prev_epoch_ax = plt.axes([0.29, 0.02, 0.1, 0.05])
    next_epoch_ax = plt.axes([0.41, 0.02, 0.1, 0.05])
    prev_band_ax = plt.axes([0.53, 0.02, 0.1, 0.05])
    next_band_ax = plt.axes([0.65, 0.02, 0.1, 0.05])
    finish_ax = plt.axes([0.77, 0.02, 0.17, 0.05])

    default_include_start = include_spacial[0]
    default_include_end = include_spacial[1]
    default_bottom_spacial = bottom_spacial
    default_top_spacial = top_spacial

    slider_start = Slider(slider_ax_start, 'Include Start', 0, image_data.shape[0]-1, valinit=default_include_start, valstep=1)
    slider_end = Slider(slider_ax_end, 'Include End', 0, image_data.shape[0], valinit=default_include_end, valstep=1)
    slider_bottom = Slider(slider_ax_bottom, 'Bottom Spacial', 0, image_data.shape[0]-2, valinit=default_bottom_spacial, valstep=1)
    slider_top = Slider(slider_ax_top, 'Top Spacial', 1, image_data.shape[0]-1, valinit=default_top_spacial, valstep=1)

    slider_start.ax.axvline(default_include_start, color='red', linestyle='-', linewidth=2)
    slider_end.ax.axvline(default_include_end, color='red', linestyle='-', linewidth=2)
    slider_bottom.ax.axvline(default_bottom_spacial, color='red', linestyle='-', linewidth=2)
    slider_top.ax.axvline(default_top_spacial, color='red', linestyle='-', linewidth=2)

    finish_button = Button(finish_ax, 'Finish and Next', color='lightgoldenrodyellow', hovercolor='0.975')
    next_band_button = Button(next_band_ax, 'Next Band', color='lightblue', hovercolor='0.875')
    prev_band_button = Button(prev_band_ax, 'Prev Band', color='lightblue', hovercolor='0.875')
    next_epoch_button = Button(next_epoch_ax, 'Next Epoch', color='lightgreen', hovercolor='0.875')
    prev_epoch_button = Button(prev_epoch_ax, 'Prev Epoch', color='lightgreen', hovercolor='0.875')
    next_star_button = Button(next_star_ax, 'Next Star', color='orchid', hovercolor='0.875')
    prev_star_button = Button(prev_star_ax, 'Prev Star', color='orchid', hovercolor='0.875')

    navigation = {
        'next_band': False,
        'prev_band': False,
        'next_star': False,
        'prev_star': False,
        'next_epoch': False,
        'prev_epoch': False,
        'finish': False
    }

    line_summed_vertical, = ax_summed_vertical.plot([], [], color='blue', label='Summed Flux (vertical)')
    scatter_anchors = ax_summed_vertical.scatter([], [], color='red', label='Anchor Points')
    line_summed_horizontal, = ax_summed_horizontal.plot([], [], color='blue', label='Summed Flux (horizontal)')
    line_norm_cleaned, = ax_norm.plot([], [], color='blue', label='Cleaned Normalized Summed Flux')
    line_norm_external, = ax_norm.plot([], [], color='orange', alpha=0.7, label='Non-Cleaned Normalized Flux')
    line_diff, = ax_diff.plot([], [], color='red', label='Flux Difference')
    line_reldiff, = ax_diff.plot([], [], color='purple', label='Relative Difference')

    ax_diff.axhline(0.1, color='red', linestyle='dashed')
    ax_diff.axhline(-0.1, color='red', linestyle='dashed')

    ax_summed_vertical.set_title(f'Vertical Summed Flux ({star_name}, {band}, Epoch {epoch_num})', fontsize=10)
    ax_summed_horizontal.set_title(f'Horizontal Summed Flux ({star_name}, {band}, Epoch {epoch_num})', fontsize=10)
    ax_norm.set_title(f'Normalized Flux Comparison ({star_name}, {band}, Epoch {epoch_num})', fontsize=10)
    ax_diff.set_title(f'Flux & Relative Difference ({star_name}, {band}, Epoch {epoch_num})', fontsize=10)

    ax_summed_vertical.set_xlabel('Wavelength (nm)')
    ax_summed_vertical.set_ylabel('Summed Flux')
    ax_summed_vertical.legend(fontsize=9)
    ax_summed_vertical.grid(True)

    ax_summed_horizontal.set_xlabel('Spatial Coordinate')
    ax_summed_horizontal.set_ylabel('Summed Flux')
    ax_summed_horizontal.legend(fontsize=9)
    ax_summed_horizontal.grid(True)

    ax_norm.set_xlabel('Wavelength (nm)')
    ax_norm.set_ylabel('Normalized Flux')
    ax_norm.legend(fontsize=9)
    ax_norm.grid(True)

    ax_diff.set_xlabel('Wavelength (nm)')
    ax_diff.set_ylabel('Difference')
    ax_diff.legend(fontsize=9)
    ax_diff.grid(True)

    def update_plots(val):
        current_bottom = int(slider_bottom.val)
        current_top = int(slider_top.val)
        if current_top <= current_bottom:
            current_top = current_bottom + 1

        abs_include_start = int(slider_start.val)
        abs_include_end = int(slider_end.val)
        if abs_include_end <= abs_include_start:
            abs_include_end = abs_include_start + 1

        # Redraw the image with updated bottom/top and include lines
        ax_image.clear()
        p2D.Plot2DImage_for_cleaning(
            image_data,
            wavelengths_2D,
            band,
            current_bottom,
            current_top,
            abs_include_start,
            abs_include_end,
            title=f"2D Flux Image for {star_name}, Band: {band}, Epoch: {epoch_num}",
            ValMin=-600,
            ValMax=600,
            ax=ax_image
        )
        ax_image.text(0.5, 1.05, "Adjust sliders below to select star region",
                      transform=ax_image.transAxes, ha='center', va='bottom', fontsize=10)

        # Compute the flux using the full image_data and the absolute include range
        # Clamp to valid array indices
        actual_start = max(0, abs_include_start)
        actual_end = min(image_data.shape[0], abs_include_end)

        if actual_end <= actual_start:
            # If the chosen region is invalid or outside the image, handle gracefully
            summed_flux = np.zeros(image_data.shape[1])
            normalized_summed_flux_resampled = np.zeros_like(external_wavelengths_band)
            selected_flux = []
        else:
            image_data_included = image_data[actual_start:actual_end, :]
            summed_flux = np.sum(image_data_included, axis=0)

            # Anchor points calculations
            anchor_points_in_range = anchor_points[(anchor_points >= wavelengths_2D.min()) & (anchor_points <= wavelengths_2D.max())]
            closest_indices = [np.abs(wavelengths_2D - ap).argmin() for ap in anchor_points_in_range]
            selected_flux = [ut.robust_mean(summed_flux[max(0, idx - 10):idx + 10], 1) for idx in closest_indices]

            if len(anchor_points_in_range) > 1:
                continuum_flux_interpolated = np.interp(wavelengths_2D, anchor_points_in_range, selected_flux)
            else:
                continuum_flux_interpolated = np.full_like(wavelengths_2D, selected_flux[0] if selected_flux else 1.0)

            normalized_summed_flux = summed_flux / continuum_flux_interpolated
            normalized_summed_flux_resampled = np.interp(external_wavelengths_band, wavelengths_2D, normalized_summed_flux)

        flux_difference = normalized_summed_flux_resampled - external_normalized_flux_band
        relative_difference = flux_difference / external_normalized_flux_band

        results['normalized_summed_flux_resampled'] = normalized_summed_flux_resampled

        line_summed_vertical.set_xdata(wavelengths_2D)
        line_summed_vertical.set_ydata(summed_flux)
        scatter_anchors.set_offsets(np.c_[anchor_points_in_range, selected_flux] if len(selected_flux) > 0 else [])
        ax_summed_vertical.relim()
        ax_summed_vertical.autoscale_view()

        # Horizontal coordinates: use the actual chosen include range for display
        # even if it's partially outside the displayed region.
        horizontal_coords = np.arange(abs_include_start, abs_include_end)
        # Adjust if out of image bounds:
        # The horizontal data must match the length of image_data_included
        # If abs_include_start/end are partly outside, actual_start/end adjusted it.
        horizontal_data_length = actual_end - actual_start
        # If invalid range, no data
        if horizontal_data_length <= 0:
            horizontal_flux = []
        else:
            horizontal_flux = np.sum(image_data[actual_start:actual_end, :], axis=1)

        line_summed_horizontal.set_xdata(horizontal_coords[:len(horizontal_flux)])
        line_summed_horizontal.set_ydata(horizontal_flux)
        ax_summed_horizontal.relim()
        ax_summed_horizontal.autoscale_view()

        line_norm_cleaned.set_xdata(external_wavelengths_band)
        line_norm_cleaned.set_ydata(normalized_summed_flux_resampled)
        line_norm_external.set_xdata(external_wavelengths_band)
        line_norm_external.set_ydata(external_normalized_flux_band)

        line_diff.set_xdata(external_wavelengths_band)
        line_diff.set_ydata(flux_difference)
        line_reldiff.set_xdata(external_wavelengths_band)
        line_reldiff.set_ydata(relative_difference)

        fig.canvas.draw_idle()

    def finish_callback(event):
        navigation['finish'] = True
        plt.close(fig)

    def next_band_callback(event):
        navigation['next_band'] = True
        plt.close(fig)

    def prev_band_callback(event):
        navigation['prev_band'] = True
        plt.close(fig)

    def next_star_callback(event):
        navigation['next_star'] = True
        plt.close(fig)

    def prev_star_callback(event):
        navigation['prev_star'] = True
        plt.close(fig)

    def next_epoch_callback(event):
        navigation['next_epoch'] = True
        plt.close(fig)

    def prev_epoch_callback(event):
        navigation['prev_epoch'] = True
        plt.close(fig)

    finish_button.on_clicked(finish_callback)
    next_band_button.on_clicked(next_band_callback)
    prev_band_button.on_clicked(prev_band_callback)
    next_star_button.on_clicked(next_star_callback)
    prev_star_button.on_clicked(prev_star_callback)
    next_epoch_button.on_clicked(next_epoch_callback)
    prev_epoch_button.on_clicked(prev_epoch_callback)

    update_plots(None)
    slider_start.on_changed(update_plots)
    slider_end.on_changed(update_plots)
    slider_bottom.on_changed(update_plots)
    slider_top.on_changed(update_plots)

    plt.show()

    final_bottom = int(slider_bottom.val)
    final_top = int(slider_top.val)
    if final_top <= final_bottom:
        final_top = final_bottom + 1

    final_include_start = int(slider_start.val)
    final_include_end = int(slider_end.val)
    if final_include_end <= final_include_start:
        final_include_end = final_include_start + 1

    # Do not clamp final_include_start/end to final_bottom/final_top.
    # The user wants to separate the functionality from the visuals.
    final_include_spacial = (final_include_start, final_include_end)

    return final_include_spacial, final_bottom, final_top, navigation, results['normalized_summed_flux_resampled'], results['wavelengths_2D']


def main():
    parser = argparse.ArgumentParser(description="Interactive flux cleaning and normalization.")
    parser.add_argument('--star_names', nargs='+', default=None, help='List of star names to process')
    parser.add_argument('--overwrite_flag', action='store_true', default=False, help='Flag to overwrite existing files')
    parser.add_argument('--backup_flag', action='store_true', default=False, help='Flag to create backups before overwriting')
    parser.add_argument('--skip_flag', action='store_true', default=False, help='Flag to skip if results file already exist')
    parser.add_argument('--load_saved_flag', action='store_true', default=False, help='Flag to load saved anchor points if available')
    args = parser.parse_args()

    star_names = specs.star_names if args.star_names is None else args.star_names
    obs_file_names = specs.obs_file_names
    overwrite_flag = args.overwrite_flag
    backup_flag = args.backup_flag
    skip_flag = args.skip_flag
    load_saved_flag = args.load_saved_flag
    obs = obsm()

    current_star_idx = 0
    while current_star_idx < len(star_names):
        star_name = star_names[current_star_idx]
        print(f"Processing star: {star_name}")

        if star_name not in obs_file_names:
            print(f"Star {star_name} not found in obs_file_names.")
            current_star_idx += 1
            continue

        star = obs.load_star_instance(star_name)
        epochs_dict = obs_file_names[star_name]
        epoch_nums = [int(re.findall(r'\d+\.\d+|\d+', epoch)[0]) for epoch in epochs_dict.keys()]
        bands = ['UVB', 'VIS', 'NIR']

        current_epoch_idx = 0
        while current_epoch_idx < len(epoch_nums):
            epoch_num = epoch_nums[current_epoch_idx]
            current_band_idx = 0

            while current_band_idx < len(bands):
                band = bands[current_band_idx]

                fits_file = star.load_observation(epoch_num, band)
                wavelengths_2D = fits_file.data['WAVE'][0]
                fits_file_2D = star.load_2D_observation(epoch_num, band)
                image_data = fits_file_2D.primary_data
                external_data = star.load_property('normalized_flux', epoch_num, 'COMBINED')
                external_normalized_flux_band = external_data['normalized_flux']
                external_wavelengths_band = external_data['wavelengths']
                anchor_points = star.load_property('norm_anchor_wavelengths', epoch_num, 'COMBINED')

                (include_spacial, bottom_spacial, top_spacial, navigation,
                 normalized_summed_flux_resampled, returned_wavelengths_2D) = clean_flux_and_normalize_interactive(
                    image_data,
                    wavelengths_2D,
                    external_normalized_flux_band,
                    external_wavelengths_band,
                    anchor_points,
                    band,
                    star_name,
                    epoch_num,
                    load_saved_flag
                )

                print(f"Processed star: {star_name}, epoch: {epoch_num}, band: {band}")
                print(f"Selected spatial range (visual): {bottom_spacial} to {top_spacial}")
                print(f"Included spatial range (functional): {include_spacial}")

                if navigation['finish']:
                    clean_normalized_flux = {
                        'normalized_flux': normalized_summed_flux_resampled,
                        'wavelengths': returned_wavelengths_2D,
                        'included_spacial_coords': include_spacial
                    }
                    star.save_property('clean_normalized_flux', clean_normalized_flux, epoch_num, band,
                                       overwrite=overwrite_flag, backup=backup_flag)

                if navigation['prev_star']:
                    current_star_idx = max(0, current_star_idx - 1)
                    break

                elif navigation['next_star']:
                    current_star_idx += 1
                    break

                elif navigation['prev_epoch']:
                    current_epoch_idx = max(0, current_epoch_idx - 1)
                    break

                elif navigation['next_epoch']:
                    current_epoch_idx += 1
                    break

                elif navigation['prev_band']:
                    current_band_idx = max(0, current_band_idx - 1)

                elif navigation['next_band']:
                    current_band_idx += 1

                elif navigation['finish']:
                    if current_band_idx < len(bands) - 1:
                        current_band_idx += 1
                    else:
                        if current_epoch_idx < len(epoch_nums) - 1:
                            current_epoch_idx += 1
                            break
                        else:
                            if current_star_idx < len(star_names) - 1:
                                current_star_idx += 1
                                break
                            else:
                                print("All stars, epochs, and bands processed.")
                                return

                else:
                    # No navigation triggered
                    pass

            else:
                current_epoch_idx += 1
                continue

            if navigation['prev_star'] or navigation['next_star']:
                break

            if navigation['prev_epoch'] or navigation['next_epoch'] or (navigation['finish'] and current_epoch_idx < len(epoch_nums)):
                continue

        else:
            current_star_idx += 1
            continue

        if navigation['prev_star'] or navigation['next_star']:
            continue

    print("All stars have been processed.")

if __name__ == "__main__":
    main()
