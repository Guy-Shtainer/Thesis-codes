#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import utils as ut
import re

from ObservationClass import ObservationManager as obsm

def interactive_normalization_nres(star,
                                   epoch_numbers,
                                   spectra_numbers,
                                   overwrite=False,
                                   backup=True,
                                   load_saved=False):
    """
    Interactive manual normalization for NRES data with:
      - One figure, 3 subplots (top = raw flux, middle = interpolated, bottom = normalized).
      - Navigation buttons for star, epoch, and spectrum.
      - On left-click, add an anchor for the entire epoch (store only its x-value).
        The y-value is computed on the fly as a robust mean over ±10 points.
      - On right-click, remove the nearest anchor based on 2D distance (using a 5% threshold on both axes).
      - When finished, anchors are applied to all spectra in that epoch.

    Assumes star.load_property('combined_flux', ep, sp) returns a dict with keys:
        {'wavelength': ..., 'flux': ...}.
    """

    current_epoch_idx   = 0
    current_spectra_idx = 0
    navigation_choice   = "finish"  # When finished with this star

    # List holding the anchor x-values for the current epoch.
    selected_wavelengths_tmp = []
    # Cache anchors per epoch so they persist when switching spectra.
    epoch_anchors = {}

    wave = np.array([])
    flux = np.array([])
    snr  = None

    mean_flux_half_batch_size = 10

    fig, (ax1, ax_mid, ax2) = plt.subplots(3, 1, figsize=(10, 9))
    plt.subplots_adjust(bottom=0.25, hspace=0.35)

    xlim1, ylim1 = None, None
    xlim_mid, ylim_mid = None, None
    xlim2, ylim2 = None, None

    ########################################################################
    # Data loading and anchor caching
    ########################################################################
    def load_data():
        nonlocal wave, flux, snr, selected_wavelengths_tmp, epoch_anchors

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
                wave = np.array(combined['wavelength'])
                flux = np.array(combined['flux'])
                snr  = None
        except Exception as e:
            print(f"Failed to load combined_flux for star={star.star_name}, epoch={ep}, spec={sp}: {e}")
            wave = np.array([])
            flux = np.array([])
            snr  = None

        if ep in epoch_anchors:
            selected_wavelengths_tmp[:] = epoch_anchors[ep][:]
        else:
            if load_saved:
                try:
                    saved_anchors = star.load_property('norm_anchor_wavelengths', ep, sp, to_print=False)
                    if saved_anchors is not None and len(saved_anchors) > 0:
                        selected_wavelengths_tmp[:] = saved_anchors.tolist()
                        print(f"Loaded {len(saved_anchors)} anchor points from disk for epoch={ep}, spec={sp}.")
                    else:
                        selected_wavelengths_tmp[:] = []
                except Exception as e:
                    print(f"Error loading anchors for star={star.star_name}, epoch={ep}, spec={sp}: {e}")
                    selected_wavelengths_tmp[:] = []
            else:
                selected_wavelengths_tmp[:] = []
            epoch_anchors[ep] = selected_wavelengths_tmp[:]

    ########################################################################
    # Plotting
    ########################################################################
    def update_plot():
        nonlocal xlim1, ylim1
        if ax1.has_data():
            xlim1 = ax1.get_xlim()
            ylim1 = ax1.get_ylim()
        ax1.clear()
        ax1.plot(wave, flux, '.', color='gray', markersize=2)

        # Plot anchors using robust mean for y-values.
        if selected_wavelengths_tmp:
            anchor_fluxes = []
            for wl in selected_wavelengths_tmp:
                idx = np.argmin(np.abs(wave - wl))
                st = max(0, idx - mean_flux_half_batch_size)
                en = min(len(flux), idx + mean_flux_half_batch_size)
                anchor_fluxes.append(ut.robust_mean(flux[st:en], 3))
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
        ax_mid.plot(wave, flux, '.', color='gray', markersize=2)
        if len(selected_wavelengths_tmp) >= 2:
            aw = np.array(selected_wavelengths_tmp)
            af = []
            for wl in aw:
                idx = np.argmin(np.abs(wave - wl))
                st = max(0, idx - mean_flux_half_batch_size)
                en = min(len(flux), idx + mean_flux_half_batch_size)
                af.append(ut.robust_mean(flux[st:en], 3))
            srt = np.argsort(aw)
            aw = aw[srt]
            af = np.array(af)[srt]
            interp_ = np.interp(wave, aw, af, left=af[0], right=af[-1])
            ax_mid.clear()
            ax_mid.plot(wave, flux, '.', color='gray', markersize=2)
            ax_mid.plot(wave, interp_, '-', color='green')
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
                st = max(0, idx - mean_flux_half_batch_size)
                en = min(len(flux), idx + mean_flux_half_batch_size)
                af.append(ut.robust_mean(flux[st:en], 3))
            aw = np.array(aw)
            af = np.array(af)
            srt = np.argsort(aw)
            aw = aw[srt]
            af = af[srt]
            interp_ = np.interp(wave, aw, af, left=af[0], right=af[-1])
            norm_ = flux / interp_
            ax2.plot(wave, norm_, '-', color='blue')
            ax2.axhline(1.0, color='red', linestyle='--')
        else:
            ax2.set_title("Not enough anchor points")
        ax2.set_xlabel("Wavelength [Å]")
        ax2.set_ylabel("Normalized Flux")
        ax2.set_ylim(-40, 40)
        if xlim2 and ylim2:
            ax2.set_xlim(xlim2)
            ax2.set_ylim(ylim2)

    ########################################################################
    # Click Handling
    ########################################################################
    press_event = {'x': None, 'y': None, 'button': None}

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
        # press_event['x'] = None
        # press_event['y'] = None
        # press_event['button'] = None

    def onclick(press_event):
        if press_event['x'] is None or press_event['y'] is None:
            return

        # Get the current epoch identifier.
        current_ep = epoch_numbers[current_epoch_idx]

        if press_event['button'] == 1:
            # Left-click: add an anchor at the clicked x position.
            anchor = press_event['x']
            if anchor not in selected_wavelengths_tmp:
                selected_wavelengths_tmp.append(anchor)
                selected_wavelengths_tmp.sort()
                epoch_anchors[current_ep] = selected_wavelengths_tmp[:]
                update_plot()
        elif press_event['button'] == 3:
            # Right-click: delete the nearest anchor using a dynamic tolerance.
            if not selected_wavelengths_tmp:
                return
            selected_wavelengths = np.array(selected_wavelengths_tmp)
            differences = np.abs(selected_wavelengths - press_event['x'])
            min_idx = np.argmin(differences)
            # Compute a dynamic threshold based on 5% of the current x-axis range.
            x_range = ax1.get_xlim()[1] - ax1.get_xlim()[0]
            tol = x_range * 0.05  # Adjust this factor as needed.
            if differences[min_idx] < tol:
                del selected_wavelengths_tmp[min_idx]
                epoch_anchors[current_ep] = selected_wavelengths_tmp[:]
                update_plot()
            else:
                print("No anchor near the click to delete.")

    fig.canvas.mpl_connect('button_press_event', onpress)
    fig.canvas.mpl_connect('button_release_event', onrelease)

    ########################################################################
    # Navigation Buttons
    ########################################################################
    ax_prev_star = plt.axes([0.05, 0.02, 0.1, 0.05])
    btn_prev_star = Button(ax_prev_star, 'PrevStar')
    def prev_star_cb(evt):
        nonlocal navigation_choice
        navigation_choice = "prev_star"
        plt.close(fig)
    btn_prev_star.on_clicked(prev_star_cb)

    ax_next_star = plt.axes([0.17, 0.02, 0.1, 0.05])
    btn_next_star = Button(ax_next_star, 'NextStar')
    def next_star_cb(evt):
        nonlocal navigation_choice
        navigation_choice = "next_star"
        plt.close(fig)
    btn_next_star.on_clicked(next_star_cb)

    ax_prev_ep = plt.axes([0.29, 0.02, 0.1, 0.05])
    btn_prev_ep = Button(ax_prev_ep, 'PrevEpoch')
    def prev_epoch_cb(evt):
        nonlocal current_epoch_idx, current_spectra_idx
        if current_epoch_idx > 0:
            current_epoch_idx -= 1
            current_spectra_idx = 0
            load_data()
            update_plot()
        else:
            print("Already at first epoch.")
    btn_prev_ep.on_clicked(prev_epoch_cb)

    ax_next_ep = plt.axes([0.41, 0.02, 0.1, 0.05])
    btn_next_ep = Button(ax_next_ep, 'NextEpoch')
    def next_epoch_cb(evt):
        nonlocal current_epoch_idx, current_spectra_idx
        if current_epoch_idx < len(epoch_numbers) - 1:
            current_epoch_idx += 1
            current_spectra_idx = 0
            load_data()
            update_plot()
        else:
            print("Already at last epoch.")
    btn_next_ep.on_clicked(next_epoch_cb)

    ax_prev_sp = plt.axes([0.53, 0.02, 0.1, 0.05])
    btn_prev_sp = Button(ax_prev_sp, 'PrevSpec')
    def prev_spec_cb(evt):
        nonlocal current_spectra_idx
        if current_spectra_idx > 0:
            current_spectra_idx -= 1
            load_data()
            update_plot()
        else:
            print("At first spectrum of this epoch.")
    btn_prev_sp.on_clicked(prev_spec_cb)

    ax_next_sp = plt.axes([0.65, 0.02, 0.1, 0.05])
    btn_next_sp = Button(ax_next_sp, 'NextSpec')
    def next_spec_cb(evt):
        nonlocal current_spectra_idx
        if current_spectra_idx < len(spectra_numbers) - 1:
            current_spectra_idx += 1
            load_data()
            update_plot()
        else:
            print("At last spectrum of this epoch.")
    btn_next_sp.on_clicked(next_spec_cb)

    ax_finish = plt.axes([0.77, 0.02, 0.1, 0.05])
    btn_finish = Button(ax_finish, 'Finish')
    def save_current_epoch_anchors():
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
            aw = np.array(selected_wavelengths_tmp)
            af = []
            for wl in aw:
                idx_ = np.argmin(np.abs(w_ - wl))
                st_ = max(0, idx_ - mean_flux_half_batch_size)
                en_ = min(len(f_), idx_ + mean_flux_half_batch_size)
                af.append(ut.robust_mean(f_[st_:en_], 3))
            aw = np.array(aw)
            af = np.array(af)
            srt_ = np.argsort(aw)
            aw = aw[srt_]
            af = af[srt_]
            interp_ = np.interp(w_, aw, af)
            norm_ = f_ / interp_
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
        nonlocal navigation_choice, current_epoch_idx, current_spectra_idx
        save_current_epoch_anchors()
        if current_epoch_idx < len(epoch_numbers) - 1:
            current_epoch_idx += 1
            current_spectra_idx = 0
            load_data()
            update_plot()
        else:
            navigation_choice = "finish"
            plt.close(fig)
    btn_finish.on_clicked(finish_cb)

    ########################################################################
    # Initial load & show
    ########################################################################
    load_data()
    update_plot()
    plt.show()
    return navigation_choice

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

    star_names = args.star_names
    overwrite_flag = args.overwrite_flag
    backup_flag = args.backup_flag
    skip_flag = args.skip_flag
    load_saved_flag = args.load_saved_flag

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
        print(f"\nStarting interactive normalization: star={star_name}, epochs={epoch_numbers}, e.g. spectra={spectra_numbers}")
        nav_choice = interactive_normalization_nres(
            star,
            epoch_numbers,
            spectra_numbers,
            overwrite=overwrite_flag,
            backup=backup_flag,
            load_saved=load_saved_flag
        )
        if nav_choice in ("finish", "next_star"):
            current_star_idx += 1
        elif nav_choice == "prev_star" and current_star_idx > 0:
            current_star_idx -= 1
        else:
            current_star_idx += 1
    print("All done. Exiting.")

if __name__=="__main__":
    main()
