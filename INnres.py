#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import utils as ut
import sys
import re

from ObservationClass import ObservationManager as obsm

############################################
# Interactive NRES Normalization
############################################
def interactive_normalization_nres(star,
                                   epoch_numbers,
                                   spectra_numbers,
                                   overwrite=False,
                                   backup=True,
                                   load_saved=False):
    """
    Interactive manual normalization with:
      - One figure, 3 subplots (top=raw flux, middle=interpolated, bottom=normalized).
      - Navigation: next/prev star, epoch, spec.
      - On "click" (left or right), we add/remove anchor for the entire epoch.
      - These anchors then apply to all spectra in that epoch upon 'Finish'.

    We assume star.load_property('combined_flux', ep, sp)
    returns a dict with keys: {'wavelength': ..., 'flux': ...}.
    """

    current_epoch_idx   = 0
    current_spectra_idx = 0
    navigation_choice   = "finish"

    selected_wavelengths_tmp = []
    wave = np.array([])
    flux = np.array([])
    snr  = None

    mean_flux_half_batch_size = 10

    fig, (ax1, ax_mid, ax2) = plt.subplots(3, 1, figsize=(10, 9))
    plt.subplots_adjust(bottom=0.25, hspace=0.35)

    xlim1, ylim1 = None, None
    xlim_mid, ylim_mid = None, None
    xlim2, ylim2 = None, None

    def load_data():
        nonlocal wave, flux, snr, selected_wavelengths_tmp

        if current_epoch_idx >= len(epoch_numbers) or current_spectra_idx >= len(spectra_numbers):
            wave = np.array([])
            flux = np.array([])
            snr  = None
            return

        ep = epoch_numbers[current_epoch_idx]
        sp = spectra_numbers[current_spectra_idx]

        try:
            combined = star.load_property('combined_flux', ep, sp)
            if not combined:
                print(f"No 'combined_flux' found for epoch={ep}, spec={sp}")
                wave = np.array([])
                flux = np.array([])
                snr  = None
            else:
                wave_ = np.array(combined['wavelength'])
                flux_ = np.array(combined['flux'])
                wave  = wave_
                flux  = flux_
                snr   = None
        except Exception as e:
            print(f"Failed to load combined_flux for star={star.star_name}, epoch={ep}, spec={sp}: {e}")
            wave = np.array([])
            flux = np.array([])
            snr  = None

        if load_saved:
            try:
                saved_anchors = star.load_property('norm_anchor_wavelengths',
                                                   ep, sp,
                                                   to_print=False)
                if saved_anchors is not None and len(saved_anchors) > 0:
                    selected_wavelengths_tmp = saved_anchors.tolist()
                    print(f"Loaded {len(saved_anchors)} anchor points from disk for epoch={ep}, spec={sp}.")
                else:
                    selected_wavelengths_tmp = []
            except FileNotFoundError:
                selected_wavelengths_tmp = []
            except Exception as e:
                print(f"Error loading anchors for star={star.star_name}, epoch={ep}, spec={sp}: {e}")
                selected_wavelengths_tmp = []

    def update_plot():
        nonlocal xlim1, ylim1

        if ax1.has_data():
            xlim1 = ax1.get_xlim()
            ylim1 = ax1.get_ylim()
        ax1.clear()

        ax1.plot(wave, flux, '.', color='gray', markersize=2)

        if selected_wavelengths_tmp:
            anchor_fluxes = []
            for wl in selected_wavelengths_tmp:
                idx = np.argmin(np.abs(wave - wl))
                st  = max(0, idx - mean_flux_half_batch_size)
                en  = min(len(flux), idx + mean_flux_half_batch_size)
                anchor_fluxes.append(ut.robust_mean(flux[st:en]))
            ax1.plot(selected_wavelengths_tmp, anchor_fluxes, 'o', color='red', markersize=5)

        ep = epoch_numbers[current_epoch_idx] if current_epoch_idx < len(epoch_numbers) else None
        sp = spectra_numbers[current_spectra_idx] if current_spectra_idx < len(spectra_numbers) else None
        ax1.set_title(f"{star.star_name}  epoch={ep}, spec={sp}")
        ax1.set_ylabel("Flux")

        if xlim1 and ylim1:
            ax1.set_xlim(xlim1)
            ax1.set_ylim(ylim1)

        plot_interpolated_flux()
        plot_normalized_flux()
        fig.canvas.draw_idle()

    def plot_interpolated_flux():
        nonlocal xlim_mid, ylim_mid
        if ax_mid.has_data():
            xlim_mid = ax_mid.get_xlim()
            ylim_mid = ax_mid.get_ylim()
        ax_mid.clear()
    
        # Plot your raw wave/flux in gray
        ax_mid.plot(wave, flux, '.', color='gray', markersize=2)
    
        if len(selected_wavelengths_tmp) >= 2:
            aw = np.array(selected_wavelengths_tmp)
            af = []
            for wl in aw:
                idx = np.argmin(np.abs(wave - wl))
                st  = max(0, idx - mean_flux_half_batch_size)
                en  = min(len(flux), idx + mean_flux_half_batch_size)
                af.append(ut.robust_mean(flux[st:en]))
            srt = np.argsort(aw)
            aw  = aw[srt]
            af  = np.array(af)[srt]
    
            # Interpolate, specifying left/right to handle out-of-range wave
            interp_ = np.interp(wave, aw, af, left=af[0], right=af[-1])
    
            # ---------------------------------------------------------------
            # Filter out any wave == 0 to avoid plotting them
            # (You could do wave <= 0 if needed, or any other condition.)
            # ---------------------------------------------------------------
            nonzero_mask = (wave != 0)
            wave_nz     = wave[nonzero_mask]
            flux_nz     = flux[nonzero_mask]
            interp_nz   = interp_[nonzero_mask]
    
            # Plot only non-zero wave portion
            ax_mid.clear()  # Clear again so we only plot masked data
            ax_mid.plot(wave_nz, flux_nz, '.', color='gray', markersize=2)
            ax_mid.plot(wave_nz, interp_nz, '-', color='green')
    
            # Still plot the anchors themselves (red points)
            ax_mid.plot(aw, af, 'o', color='red')
        else:
            ax_mid.set_title("Not enough anchor points")
    
        ax_mid.set_ylabel("Interpolated Flux")
    
        if xlim_mid and ylim_mid:
            ax_mid.set_xlim(xlim_mid)
            ax_mid.set_ylim(ylim_mid)


    def plot_normalized_flux():
        nonlocal xlim2, ylim2
        if ax2.has_data():
            xlim2 = ax2.get_xlim()
            ylim2 = ax2.get_ylim()
        ax2.clear()

        if len(selected_wavelengths_tmp) >= 2:
            aw = np.array(selected_wavelengths_tmp)
            af = []
            for wl in aw:
                idx = np.argmin(np.abs(wave - wl))
                st  = max(0, idx - mean_flux_half_batch_size)
                en  = min(len(flux), idx + mean_flux_half_batch_size)
                af.append(ut.robust_mean(flux[st:en]))
            aw = np.array(aw)
            af = np.array(af)
            srt = np.argsort(aw)
            aw  = aw[srt]
            af  = af[srt]

            # Same approach for out-of-range wave:
            interp_ = np.interp(wave, aw, af, left=af[0], right=af[-1])
            norm_   = flux / interp_

            ax2.plot(wave, norm_, '-', color='blue')
            ax2.axhline(1.0, color='red', linestyle='--')
        else:
            ax2.set_title("Not enough anchor points")

        # 3) Add the Angstrom label
        ax2.set_xlabel("Wavelength [Å]")
        ax2.set_ylabel("Normalized Flux")

        # 2) Force y-limits from -40 to +40:
        ax2.set_ylim(-40, 40)

        if xlim2 and ylim2:
            ax2.set_xlim(xlim2)
            ax2.set_ylim(ylim2)

    ########################################################################
    # Click Handling: onpress/onrelease approach
    ########################################################################
    press_event = {'x': None, 'y': None, 'button': None}

    def onpress(event):
        """Record the mouse down event."""
        if event.inaxes != ax1:
            return
        press_event['x'] = event.xdata
        press_event['y'] = event.ydata
        press_event['button'] = event.button

    def onrelease(event):
        """If the mouse hasn't moved much, interpret as a click."""
        if event.inaxes != ax1:
            return
        dx = event.xdata - press_event['x'] if press_event['x'] is not None else 9999
        dy = event.ydata - press_event['y'] if press_event['y'] is not None else 9999
        dist = np.hypot(dx, dy)

        # A fraction of the total x-range as threshold for minimal movement
        movement_threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.01
        if dist < movement_threshold:
            onclick(press_event)

        # Reset the press event
        press_event['x'] = None
        press_event['y'] = None
        press_event['button'] = None

    def onclick(press_ev):
        """Handle a left-click (add anchor) or right-click (remove anchor) in data space."""
        if press_ev['x'] is None or press_ev['y'] is None:
            return

        # Data-based thresholds
        x_threshold = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.05
        y_threshold = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05

        if press_ev['button'] == 1:
            # 1) Find the single closest data point in the usual way
            x_diff = np.abs(wave - press_ev['x'])
            y_diff = np.abs(flux - press_ev['y'])
            within_threshold = (x_diff < x_threshold) & (y_diff < y_threshold)
            if not np.any(within_threshold):
                return
        
            candidate_waves = wave[within_threshold]
            candidate_fluxes= flux[within_threshold]
            distances = np.hypot(candidate_waves - press_ev['x'],
                                 candidate_fluxes - press_ev['y'])
            min_idx = np.argmin(distances)
            global_idx = np.where(within_threshold)[0][min_idx]
        
            # 2) From that single index, define a range of ±10 points
            #    so we can robust-mean the wavelength.  
            #    (Or simply keep the center wavelength if that’s preferred.)
            st_ = max(0, global_idx - 10)
            en_ = min(len(wave), global_idx + 10)
        
            # 3) Option A: Anchor wave = the exact wave of the center point
            # anchor_wl = wave[global_idx]
        
            #    Option B: Anchor wave = average wave in that region
            anchor_wl = ut.robust_mean(wave[st_:en_])
        
            # 4) Append to selected_wavelengths_tmp
            if anchor_wl not in selected_wavelengths_tmp:
                selected_wavelengths_tmp.append(anchor_wl)
                selected_wavelengths_tmp.sort()
                update_plot()


        elif press_ev['button'] == 3:
            # Right-click => remove nearest anchor
            if not selected_wavelengths_tmp:
                return
            anchor_waves = np.array(selected_wavelengths_tmp)
            anchor_fluxes = []
            for wl in anchor_waves:
                idx_ = np.argmin(np.abs(wave - wl))
                st_  = max(0, idx_ - mean_flux_half_batch_size)
                en_  = min(len(flux), idx_ + mean_flux_half_batch_size)
                anchor_fluxes.append(ut.robust_mean(flux[st_:en_]))
            anchor_waves = np.array(anchor_waves)
            anchor_fluxes= np.array(anchor_fluxes)

            x_diff = np.abs(anchor_waves - press_ev['x'])
            y_diff = np.abs(anchor_fluxes - press_ev['y'])
            within_threshold = (x_diff < x_threshold) & (y_diff < y_threshold)
            if not np.any(within_threshold):
                return

            # Among the anchors within threshold, remove the closest
            candidate_waves = anchor_waves[within_threshold]
            candidate_fluxes= anchor_fluxes[within_threshold]
            distances = np.hypot(candidate_waves - press_ev['x'],
                                 candidate_fluxes - press_ev['y'])
            min_idx = np.argmin(distances)
            global_idx = np.where(within_threshold)[0][min_idx]
            del selected_wavelengths_tmp[global_idx]
            update_plot()

    # Connect onpress/onrelease to the figure
    fig.canvas.mpl_connect('button_press_event', onpress)
    fig.canvas.mpl_connect('button_release_event', onrelease)

    ########################################################################
    # Navigation Buttons
    ########################################################################
    # (1) PrevStar
    ax_prev_star = plt.axes([0.05, 0.02, 0.1, 0.05])
    btn_prev_star= Button(ax_prev_star, 'PrevStar')
    def prev_star_cb(evt):
        nonlocal navigation_choice
        navigation_choice = "prev_star"
        plt.close(fig)
    btn_prev_star.on_clicked(prev_star_cb)

    # (2) NextStar
    ax_next_star = plt.axes([0.17, 0.02, 0.1, 0.05])
    btn_next_star= Button(ax_next_star, 'NextStar')
    def next_star_cb(evt):
        nonlocal navigation_choice
        navigation_choice = "next_star"
        plt.close(fig)
    btn_next_star.on_clicked(next_star_cb)

    # (3) PrevEpoch
    ax_prev_ep = plt.axes([0.29, 0.02, 0.1, 0.05])
    btn_prev_ep= Button(ax_prev_ep, 'PrevEpoch')
    def prev_epoch_cb(evt):
        nonlocal current_epoch_idx, current_spectra_idx, selected_wavelengths_tmp
        if current_epoch_idx > 0:
            current_epoch_idx -= 1
            current_spectra_idx = 0

            # Load the new epoch's data & anchors
            load_data()
            update_plot()
        else:
            print("Already at first epoch.")
    btn_prev_ep.on_clicked(prev_epoch_cb)

    # (4) NextEpoch
    ax_next_ep = plt.axes([0.41, 0.02, 0.1, 0.05])
    btn_next_ep= Button(ax_next_ep, 'NextEpoch')
    def next_epoch_cb(evt):
        nonlocal current_epoch_idx, current_spectra_idx, selected_wavelengths_tmp
        if current_epoch_idx < len(epoch_numbers) - 1:
            current_epoch_idx += 1
            current_spectra_idx = 0

            # Load the new epoch's data & anchors
            load_data()
            update_plot()
        else:
            print("Already at last epoch.")
    btn_next_ep.on_clicked(next_epoch_cb)

    # (5) PrevSpec
    ax_prev_sp = plt.axes([0.53, 0.02, 0.1, 0.05])
    btn_prev_sp= Button(ax_prev_sp, 'PrevSpec')
    def prev_spec_cb(evt):
        nonlocal current_spectra_idx
        if current_spectra_idx > 0:
            # If you want to save each spec individually, you could call save_current_epoch_anchors() here,
            # but typically anchors are saved epoch-wide upon "Finish" for NRES. Adjust as needed.
            current_spectra_idx -= 1

            load_data()
            update_plot()
        else:
            print("At first spectrum of this epoch.")
    btn_prev_sp.on_clicked(prev_spec_cb)

    # (6) NextSpec
    ax_next_sp = plt.axes([0.65, 0.02, 0.1, 0.05])
    btn_next_sp= Button(ax_next_sp, 'NextSpec')
    def next_spec_cb(evt):
        nonlocal current_spectra_idx
        if current_spectra_idx < len(spectra_numbers) - 1:
            # If you want to save each spec individually, you could call save_current_epoch_anchors() here,
            # but typically anchors are saved epoch-wide upon "Finish" for NRES. Adjust as needed.
            current_spectra_idx += 1

            load_data()
            update_plot()
        else:
            print("At last spectrum of this epoch.")
    btn_next_sp.on_clicked(next_spec_cb)

    # (7) Finish
    ax_finish = plt.axes([0.77, 0.02, 0.1, 0.05])
    btn_finish= Button(ax_finish, 'Finish')

    def save_current_epoch_anchors():
        """
        Apply anchors to all spectra in the current epoch if >=2 anchors.
        For each spec, load 'combined_flux', build interpolation, save
        'norm_anchor_wavelengths', 'normalized_flux', 'interpolated_flux'.
        """
        if current_epoch_idx >= len(epoch_numbers):
            return
        ep = epoch_numbers[current_epoch_idx]
        if len(selected_wavelengths_tmp) < 2:
            print(f"Not enough anchor points in epoch={ep}. No save.")
            return

        for sp in spectra_numbers:
            try:
                combined = star.load_property('combined_flux', ep, sp)
                if not combined:
                    print(f"No 'combined_flux' found for star={star.star_name}, ep={ep}, sp={sp}")
                    continue
                w_ = np.array(combined['wavelength'])
                f_ = np.array(combined['flux'])
            except Exception as e:
                print(f"Failed star={star.star_name}, epoch={ep}, spec={sp}: {e}")
                continue

            # Build anchor flux array
            aw = np.array(selected_wavelengths_tmp)
            af = []
            for wl in aw:
                idx_ = np.argmin(np.abs(w_ - wl))
                st_  = max(0, idx_ - mean_flux_half_batch_size)
                en_  = min(len(f_), idx_ + mean_flux_half_batch_size)
                af.append(ut.robust_mean(f_[st_:en_]))
            aw = np.array(aw)
            af = np.array(af)
            srt_ = np.argsort(aw)
            aw   = aw[srt_]
            af   = af[srt_]

            interp_ = np.interp(w_, aw, af)
            norm_   = f_ / interp_

            # Save
            star.save_property('norm_anchor_wavelengths', aw, ep, sp,
                               overwrite=overwrite, backup=backup)
            star.save_property('normalized_flux',
                               {'wavelengths': w_, 'normalized_flux': norm_},
                               ep, sp, overwrite=overwrite, backup=backup)
            star.save_property('interpolated_flux',
                               {'wavelengths': w_, 'interpolated_flux': interp_},
                               ep, sp, overwrite=overwrite, backup=backup)

        print(f"Saved anchor-based normalization for epoch={ep} on {len(spectra_numbers)} spectra.")

    def finish_cb(evt):
        nonlocal navigation_choice
        # Save for the current epoch one last time
        save_current_epoch_anchors()
        navigation_choice = "finish"
        plt.close(fig)

    btn_finish.on_clicked(finish_cb)

    # ------------------------------------------------------------
    # Initial load & show
    # ------------------------------------------------------------
    load_data()      # loads wave/flux & any saved anchors if load_saved=True
    update_plot()
    plt.show()
    return navigation_choice


############################################
# MAIN
############################################
def main():
    parser = argparse.ArgumentParser(
        description="Interactive normalization of NRES data (single figure)."
    )
    parser.add_argument('--star_names', nargs='+', default=['WR 52','WR17'],
                        help='Which star(s) to process')
    parser.add_argument('--overwrite_flag', action='store_true', default=False)
    parser.add_argument('--backup_flag', action='store_true', default=False)
    parser.add_argument('--skip_flag', action='store_true', default=False)
    parser.add_argument('--load_saved_flag', action='store_true', default=False)
    args = parser.parse_args()

    star_names     = args.star_names
    overwrite_flag = args.overwrite_flag
    backup_flag    = args.backup_flag
    skip_flag      = args.skip_flag
    load_saved_flag= args.load_saved_flag

    obs = obsm()
    current_star_idx = 0

    while current_star_idx < len(star_names):
        star_name = star_names[current_star_idx]
        star = obs.load_star_instance(star_name)
        if not star:
            print(f"Failed loading star={star_name}. Skipping.")
            current_star_idx += 1
            continue

        epoch_numbers = star.get_all_epoch_numbers()
        if not epoch_numbers:
            print(f"No epochs for star={star_name}. Skipping.")
            current_star_idx += 1
            continue

        first_ep = epoch_numbers[0]
        spectra_numbers = star.get_all_spectra_in_epoch(first_ep)
        if not spectra_numbers:
            print(f"No spectra in epoch={first_ep}. Skipping star.")
            current_star_idx += 1
            continue

        print(f"\nStarting interactive normalization: star={star_name}, "
              f"epochs={epoch_numbers}, e.g. spectra={spectra_numbers}")

        nav_choice = interactive_normalization_nres(
            star,
            epoch_numbers,
            spectra_numbers,
            overwrite=overwrite_flag,
            backup=backup_flag,
            load_saved=load_saved_flag
        )

        # Navigate to next or previous star
        if nav_choice in ("finish","next_star"):
            current_star_idx += 1
        elif nav_choice == "prev_star" and current_star_idx > 0:
            current_star_idx -= 1
        else:
            current_star_idx += 1

    print("All done. Exiting.")


if __name__=="__main__":
    main()
