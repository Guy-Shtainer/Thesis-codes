import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider, CheckButtons
import numpy as np
import TwoDImage as p2D
import utils as ut

# GPT 23/7/25 17:49
# at the top of clean_flux_and_normalize_interactive (or near other constants)
SNR_PROP_NAME = 'snr_bounds'          # dict saved to star file:
# {'red': [x1,x2], 'blue': [x3,x4]}
NEG_MASK_STATE = {'on': False}


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
        star=None,  # NEW: pass in the star instance
        bottom_spacial=None,
        top_spacial=None,
        include_spacial=None,
        snr_cache=None
    ):
    # -------------------------------------------------------------------------
    #  If the load flag is on and a star is provided, try to load saved ranges.
    #  These will override passed defaults (only for stars/epochs already processed).
    # -------------------------------------------------------------------------
    if load_saved_flag and star is not None:
        saved_inc = star.load_property('include_range', epoch_num, band)
        if saved_inc is not None:
            include_spacial = (saved_inc['bottom_include'], saved_inc['top_include'])
        saved_spat = star.load_property('spacial_range', epoch_num, band)
        if saved_spat is not None:
            bottom_spacial = saved_spat['bottom_spacial']
            top_spacial = saved_spat['top_spacial']

    # Set default spatial limits if not provided:
    if bottom_spacial is None or top_spacial is None:
        if band == 'NIR':
            bottom_spacial, top_spacial = (23, 51)
        else:
            bottom_spacial, top_spacial = (21, 76)
    print(f"The top limit is: {top_spacial}, and the bottom limit is: {bottom_spacial}")

    if include_spacial is None:
        include_spacial = (bottom_spacial, top_spacial)

    # Determine the "cleaned" status: if a saved include_range exists then show a check mark.
    cleaned_status = "V" if (load_saved_flag and star is not None and star.load_property('include_range', epoch_num, band) is not None) else "X"

    # Results dictionary and extra values to be returned (we add clean_flux and continuum flux)
    results = {
        'normalized_summed_flux_resampled': None,
        # 'wavelengths_2D': wavelengths_2D,
        'wavelengths_2D': wavelengths_2D,
        'clean_flux': None,              # summed flux before normalization
        'interpolated_flux': None          # continuum flux used for normalization
    }

    # Make a mutable copy of anchors
    selected_anchors_list = list(anchor_points) if anchor_points is not None else []
    press_event = {'x': None, 'y': None, 'button': None}
    mean_flux_half_batch_size = 10  # as before

    # Create figure and subplots
    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.25, wspace=0.4, hspace=0.6)

    # Draw the 2D image with the current ranges.
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
    # Instruction text
    ax_image.text(0.5, 1.05, "Adjust sliders below to select star region",
                  transform=ax_image.transAxes, ha='center', va='bottom', fontsize=10)
    # NEW: Display cleaned status in the upper right corner of the image
    ax_image.text(0.98, 0.98, f"Cleaned: {cleaned_status}", transform=ax_image.transAxes,
                  ha='right', va='top', fontsize=12, color='green' if cleaned_status=="V" else 'red')

    # GPT 6/9/25 16:13
    # --- Checkbox to highlight negative pixels (session-sticky) ---
    # Place it near the 2D image: to the right of ax_image.
    img_pos = ax_image.get_position()  # figure fraction coords
    # Try to put the box just to the right of the image; clamp to figure bounds
    cb_left = min(img_pos.x1 + 0.01, 0.88)
    cb_bottom = max(img_pos.y1 - 0.075, 0.02)
    cb_ax = fig.add_axes([cb_left, cb_bottom, 0.10, 0.06])
    # create with styling that hides the checkbox frame (the little square)
    neg_check = CheckButtons(
        cb_ax, ['<0'], [NEG_MASK_STATE['on']],
        label_props={'bbox': [dict(facecolor='none', edgecolor='none', pad=0)]}  # no label box
    )

    # also hide the axes border so *nothing* frames the control
    cb_ax.set_frame_on(False)
    for s in cb_ax.spines.values():
        s.set_visible(False)
    cb_ax.set_xticks([])
    cb_ax.set_yticks([])

    def _on_neg_toggle(_):
        NEG_MASK_STATE["on"] = not NEG_MASK_STATE["on"]
        update_plots(None)  # redraw with/without overlay

    neg_check.on_clicked(_on_neg_toggle)

    ### End GPT 6/9/25 16:13

    # Create additional subplots (same as before)
    ax_summed_vertical = plt.subplot2grid((3, 2), (0, 1))
    ax_summed_horizontal = plt.subplot2grid((3, 2), (1, 1))
    ax_norm = plt.subplot2grid((3, 2), (2, 0))

    ### GPT 23/7/25 17:50
    # -------------------------------------------------------------
    # --- S N R   w i n d o w   i n i t i a l   p o s i t i o n s
    # -------------------------------------------------------------
    centre_idx = len(wavelengths_2D) // 2
    half_red = 20  # red window = ±20 pixels
    half_blue = 60  # blue window starts wider

    def idx_to_wave(idx):
        return wavelengths_2D[np.clip(idx, 0, len(wavelengths_2D) - 1)]

    # Try to reload saved bounds for this band/epoch:
    # Try to reload from the in‑memory cache
    cache_key = (star_name, epoch_num, band)
    saved_bounds = snr_cache.get(cache_key) if snr_cache is not None else None

    if saved_bounds:  # pulled from cache
        snr_bounds_red = saved_bounds['red']
        snr_bounds_blue = saved_bounds['blue']
    else:  # default positions
        snr_bounds_red = [idx_to_wave(centre_idx - half_red),
                          idx_to_wave(centre_idx + half_red)]
        snr_bounds_blue = [idx_to_wave(centre_idx - half_blue),
                           idx_to_wave(centre_idx + half_blue)]

    # Two red dashed lines  (continuum window)
    snr_red_left = ax_norm.axvline(snr_bounds_red[0], color='red', ls='--', lw=1.2,
                                   picker=5)
    snr_red_right = ax_norm.axvline(snr_bounds_red[1], color='red', ls='--', lw=1.2,
                                    picker=5)

    # Two blue dashed lines (emission‑line window)
    snr_blue_left = ax_norm.axvline(snr_bounds_blue[0], color='blue', ls='--', lw=1.2,
                                    picker=5)
    snr_blue_right = ax_norm.axvline(snr_bounds_blue[1], color='blue', ls='--', lw=1.2,
                                     picker=5)

    # Add legend once
    ax_norm.legend(fontsize=9, loc='upper right')

    ax_diff = plt.subplot2grid((3, 2), (2, 1))

    ax_norm.set_ylim(-3, 5)
    ax_diff.set_ylim(-3, 5)
    ax_norm.set_xlim(np.min(wavelengths_2D)-10, np.max(wavelengths_2D)+10)
    ax_diff.set_xlim(np.min(wavelengths_2D)-10, np.max(wavelengths_2D)+10)

    # Set up slider/button axes (same as before)
    slider_ax_range = plt.axes([0.1, 0.14, 0.8, 0.03])
    slider_ax_range_bottom = plt.axes([0.1, 0.07, 0.8, 0.03])
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

    slider_range = RangeSlider(slider_ax_range, 'Include Range', 0, image_data.shape[0],
                               valinit=(default_include_start, default_include_end), valstep=1)
    slider_range._handles[0].set_color('red')
    slider_range._handles[1].set_color('red')
    slider_range_bottom = RangeSlider(slider_ax_range_bottom, 'Spacial Range', 0, image_data.shape[0]-1,
                                      valinit=(default_bottom_spacial, default_top_spacial), valstep=1)
    slider_range_bottom._handles[0].set_color('red')
    slider_range_bottom._handles[1].set_color('red')

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

    # Set up lines and scatter objects in the subplots (same as before)
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

    # -------------------------
    # Mouse callback functions (same as before)
    # -------------------------
    def onpress(event):
        if event.inaxes != ax_summed_vertical:
            return
        press_event['x'] = event.xdata
        press_event['y'] = event.ydata
        press_event['button'] = event.button

    def onrelease(event):
        if event.inaxes != ax_summed_vertical:
            return
        dx = event.xdata - press_event['x'] if press_event['x'] is not None else 0
        dy = event.ydata - press_event['y'] if press_event['y'] is not None else 0
        dist = np.hypot(dx, dy)
        movement_threshold = (ax_summed_vertical.get_xlim()[1] - ax_summed_vertical.get_xlim()[0]) * 0.01
        if dist < movement_threshold:
            onclick(press_event)

    # GPT 23/7/25 17:50
    # ------------------------------------------------------------------
    # Draggable SNR‑window lines (works only inside ax_norm)
    # ------------------------------------------------------------------
    dragging = {'artist': None}

    def on_pick(event):
        if event.artist in (snr_red_left, snr_red_right,
                            snr_blue_left, snr_blue_right):
            dragging['artist'] = event.artist

    def snap_to_grid(x):
        # optional: snap to nearest wavelength sample
        idx = np.abs(wavelengths_2D - x).argmin()
        return wavelengths_2D[idx]

    def on_motion(event):
        art = dragging['artist']
        if art is None or event.inaxes != ax_norm or event.xdata is None:
            return
        new_x = snap_to_grid(event.xdata)

        # move the picked line and update the corresponding bound list
        art.set_xdata([new_x, new_x])
        if art is snr_red_left:
            snr_bounds_red[0] = new_x
        elif art is snr_red_right:
            snr_bounds_red[1] = new_x
        elif art is snr_blue_left:
            snr_bounds_blue[0] = new_x
        else:
            snr_bounds_blue[1] = new_x

        update_plots(None)  # live refresh

    def on_release(event):
        dragging['artist'] = None

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    def onclick(event_press):
        if event_press['x'] is None or event_press['y'] is None:
            return
        x_threshold = (ax_summed_vertical.get_xlim()[1] - ax_summed_vertical.get_xlim()[0]) * 0.01
        click_x = event_press['x']
        button = event_press['button']
        if button == 1:
            idx = np.abs(wavelengths_2D - click_x).argmin()
            anchor_x = wavelengths_2D[idx]
            if not any([abs(a - anchor_x) < x_threshold for a in selected_anchors_list]):
                selected_anchors_list.append(anchor_x)
                selected_anchors_list.sort()
                update_plots(None)
        elif button == 3:
            if len(selected_anchors_list) == 0:
                return
            anchor_array = np.array(selected_anchors_list)
            distances = np.abs(anchor_array - click_x)
            min_idx = np.argmin(distances)
            if distances[min_idx] < x_threshold:
                selected_anchors_list.pop(min_idx)
                update_plots(None)

    fig.canvas.mpl_connect('button_press_event', onpress)
    fig.canvas.mpl_connect('button_release_event', onrelease)

    # -------------------------
    # Update plots function: now also records the summed flux and continuum flux
    # -------------------------
    def update_plots(val):
        current_bottom, current_top = map(int, slider_range_bottom.val)
        if current_top <= current_bottom:
            current_top = current_bottom + 1
        abs_include_start, abs_include_end = map(int, slider_range.val)
        if abs_include_end <= abs_include_start:
            abs_include_end = abs_include_start + 1
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
        # Redraw the cleaned-status annotation
        ax_image.text(0.98, 0.98, f"Cleaned: {cleaned_status}", transform=ax_image.transAxes,
                      ha='right', va='top', fontsize=12, color='green' if cleaned_status=="V" else 'red')

        # GPT 6/9/25 16:14
        # --- Optional pink overlay where image_data < 0 (high contrast vs. green) ---
        if NEG_MASK_STATE["on"]:
            # 1) Remember the current view limits that p2D set (incl. your spatial window)
            xlim = ax_image.get_xlim()
            ylim = ax_image.get_ylim()

            # 2) Build an RGBA array: fully opaque pink on negatives, transparent elsewhere
            neg_mask = image_data < 0
            overlay = np.zeros(
                (image_data.shape[0], image_data.shape[1], 4), dtype=float
            )
            overlay[..., 0] = 1.0  # R
            overlay[..., 2] = 1.0  # B   -> (1, 0, 1) pink
            overlay[..., 3] = neg_mask.astype(
                float
            )  # alpha: 1 for negatives, 0 otherwise

            # 3) Draw it exactly on top, but don't let imshow autoscale the axes
            ax_image.imshow(
                overlay,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[
                    np.min(wavelengths_2D),
                    np.max(wavelengths_2D),
                    0,
                    image_data.shape[0],
                ],
                zorder=10,
                clip_on=True,
            )

            # 4) Restore the view limits so your spatial frame/zoom never changes
            ax_image.set_xlim(xlim)
            ax_image.set_ylim(ylim)
        ### End GPT 6/9/25 16:14

        actual_start = max(0, abs_include_start)
        actual_end = min(image_data.shape[0], abs_include_end)
        if actual_end <= actual_start:
            summed_flux = np.zeros(image_data.shape[1])
            normalized_summed_flux_resampled = np.zeros_like(external_wavelengths_band)
            all_anchors = []
            all_fluxes = []
        else:
            image_data_included = image_data[actual_start:actual_end, :]
            summed_flux = np.sum(image_data_included, axis=0)
            left_edge = wavelengths_2D.min() + 10
            right_edge = wavelengths_2D.max() - 10
            user_anchors = np.array(selected_anchors_list)
            anchor_points_in_range = user_anchors[(user_anchors >= left_edge) & (user_anchors <= right_edge)]
            tolerance = 15
            has_left_edge_anchor = np.any(np.isclose(anchor_points_in_range, left_edge, atol=tolerance))
            has_right_edge_anchor = np.any(np.isclose(anchor_points_in_range, right_edge, atol=tolerance))
            def compute_anchor_flux(wave_val):
                i_closest = np.abs(wavelengths_2D - wave_val).argmin()
                i_min = max(0, i_closest - mean_flux_half_batch_size)
                i_max = min(len(summed_flux), i_closest + mean_flux_half_batch_size)
                return ut.robust_mean(summed_flux[i_min:i_max], 1)
            added_anchors = []
            added_fluxes = []
            if not has_left_edge_anchor:
                left_edge_flux = compute_anchor_flux(left_edge)
                added_anchors.append(left_edge)
                added_fluxes.append(left_edge_flux)
            if not has_right_edge_anchor:
                right_edge_flux = compute_anchor_flux(right_edge)
                added_anchors.append(right_edge)
                added_fluxes.append(right_edge_flux)
            closest_indices = [np.abs(wavelengths_2D - ap).argmin() for ap in anchor_points_in_range]
            selected_flux = [
                ut.robust_mean(summed_flux[max(0, idx-mean_flux_half_batch_size): idx+mean_flux_half_batch_size], 1)
                for idx in closest_indices
            ]
            all_anchors = np.concatenate([anchor_points_in_range, added_anchors])
            all_fluxes  = np.concatenate([selected_flux, added_fluxes])
            sort_indices = np.argsort(all_anchors)
            all_anchors = all_anchors[sort_indices]
            all_fluxes  = all_fluxes[sort_indices]
            if len(all_anchors) > 1:
                continuum_flux_interpolated = np.interp(wavelengths_2D, all_anchors, all_fluxes)
            else:
                default_flux = all_fluxes[0] if len(all_fluxes) > 0 else 1.0
                continuum_flux_interpolated = np.full_like(wavelengths_2D, default_flux)
            # -------------- NORMALISE on native band grid ---------------------
            normalized_summed_flux = summed_flux / continuum_flux_interpolated

            # GPT 23/7/25 17:50
            # # -------------- 1) Continuum SNR (red window) ---------------------
            # low_r, high_r = sorted(snr_bounds_red)
            # mask_r = (wavelengths_2D >= low_r) & (wavelengths_2D <= high_r)
            # cont_seg = normalized_summed_flux[mask_r]
            # snr_cont = np.nan
            # if cont_seg.size > 1 and np.all(np.isfinite(cont_seg)):
            #     snr_cont = np.average(cont_seg) / np.std(cont_seg)

            # # -------------- 2) Integrated line SNR (blue window) --------------
            # low_b, high_b = sorted(snr_bounds_blue)
            # mask_b = (wavelengths_2D >= low_b) & (wavelengths_2D <= high_b)
            # line_seg = normalized_summed_flux[mask_b]
            # snr_line = np.nan
            # if line_seg.size > 1 and np.isfinite(line_seg).all() and not np.isnan(snr_cont):
            #     snr_line = np.average(line_seg) / np.std(line_seg)

            # -------------- 1) Continuum SNR (red window) ---------------------
            low_r, high_r = sorted(snr_bounds_red)
            mask_r = (wavelengths_2D >= low_r) & (wavelengths_2D <= high_r)
            cont_seg = normalized_summed_flux[mask_r]

            snr_cont = np.nan
            sigma_red = np.nan
            if cont_seg.size > 1 and np.all(np.isfinite(cont_seg)):
                mu_red = np.nanmean(cont_seg)
                sigma_red = np.nanstd(cont_seg, ddof=1) if cont_seg.size > 2 else np.nan
                # keep RED SNR unchanged: mean/std inside the red window
                snr_cont = mu_red / sigma_red if (sigma_red > 0 and np.isfinite(sigma_red)) else np.nan

            # -------------- 2) Line SNR (blue window, new definition) ----------
            # SNR_blue = (mean(line) - 1) / std(red)
            low_b, high_b = sorted(snr_bounds_blue)
            mask_b = (wavelengths_2D >= low_b) & (wavelengths_2D <= high_b)
            line_seg = normalized_summed_flux[mask_b]

            snr_line = np.nan
            if line_seg.size > 1 and np.all(np.isfinite(line_seg)) and np.isfinite(sigma_red) and (sigma_red > 0):
                mu_blue = np.nanmean(line_seg)
                snr_line = (mu_blue - 1.0) / sigma_red


            # -------------- Resample to combined grid (unchanged) -------------
            normalized_summed_flux_resampled = np.interp(
                external_wavelengths_band, wavelengths_2D, normalized_summed_flux)

            # -------------- Annotation (clear then draw) ----------------------
            for t in ax_norm.texts:
                t.remove()
            ax_norm.text(
                0.02, 0.95,
                f"SNR_cont (red): {snr_cont:.0f}\nSNR_line (blue): {snr_line:.0f}",
                transform=ax_norm.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                fontsize=9
            )

        flux_difference = normalized_summed_flux_resampled - external_normalized_flux_band
        relative_difference = flux_difference / external_normalized_flux_band
        results['normalized_summed_flux'] = normalized_summed_flux
        results['clean_flux'] = summed_flux
        results['interpolated_flux'] = continuum_flux_interpolated

        line_summed_vertical.set_xdata(wavelengths_2D)
        line_summed_vertical.set_ydata(summed_flux)
        if len(all_anchors) > 0:
            scatter_anchors.set_offsets(np.c_[all_anchors, all_fluxes])
        else:
            scatter_anchors.set_offsets([])
        ax_summed_vertical.relim()
        ax_summed_vertical.autoscale_view()

        horizontal_coords = np.arange(abs_include_start, abs_include_end)
        if actual_end - actual_start <= 0:
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

    # -------------------------
    # Button callbacks (same as before)
    # -------------------------
    def finish_callback(event):
        navigation['finish'] = True
        # store updated bounds in the cache (in memory only)
        if snr_cache is not None:
            snr_cache[cache_key] = {
                'red': list(sorted(snr_bounds_red)),
                'blue': list(sorted(snr_bounds_blue))
            }
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

    # Initialize plots
    update_plots(None)
    slider_range.on_changed(update_plots)
    slider_range_bottom.on_changed(update_plots)
    plt.show()

    final_bottom, final_top = map(int, slider_range_bottom.val)
    if final_top <= final_bottom:
        final_top = final_bottom + 1
    final_include_start, final_include_end = map(int, slider_range.val)
    if final_include_end <= final_include_start:
        final_include_end = final_include_start + 1
    final_include_spacial = (final_include_start, final_include_end)

    # Return also the clean flux and interpolated continuum so these can be saved.
    return (final_include_spacial, final_bottom, final_top, navigation, 
            results['normalized_summed_flux'], results['wavelengths_2D'],
            results['clean_flux'], results['interpolated_flux'])


def main():
    import argparse
    import re
    from ObservationClass import ObservationManager as obsm
    import specs

    # GPT 23/7/25 18:05
    # ------------ in‑memory cache for red / blue SNR lines -------------
    snr_bounds_cache = {}  # key = (star_name, epoch_num, band)
    # val = {'red':[x1,x2], 'blue':[x3,x4]}

    parser = argparse.ArgumentParser(description="Interactive flux cleaning and normalization.")
    parser.add_argument('--star_names', nargs='+', default=None, help='List of star names to process')
    parser.add_argument('--overwrite_flag', action='store_true', default=False, help='Flag to overwrite existing files')
    parser.add_argument('--backup_flag', action='store_true', default=False, help='Flag to create backups before overwriting')
    parser.add_argument('--skip_flag', action='store_true', default=False, help='Flag to skip if results file already exist')
    parser.add_argument('--load_saved_flag', action='store_true', default=False, help='Flag to load saved anchor points and ranges if available')
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

        # For star-level navigation:
        star_navigation_triggered = False

        current_epoch_idx = 0
        while current_epoch_idx < len(epoch_nums):
            epoch_num = epoch_nums[current_epoch_idx]
            # For a new epoch, reset defaults and the previous_band_height
            default_include_spacial = None  
            default_bottom_spacial = None
            default_top_spacial = None
            previous_band_height = None

            current_band_idx = 0
            epoch_navigation_triggered = False
            while current_band_idx < len(bands):
                band = bands[current_band_idx]

                # Load observation data for this band.
                fits_file = star.load_observation(epoch_num, band)
                wavelengths_2D = fits_file.data['WAVE'][0]
                fits_file_2D = star.load_2D_observation(epoch_num, band)
                image_data = fits_file_2D.primary_data

                external_data = star.load_property('normalized_flux', epoch_num, 'COMBINED')
                external_normalized_flux_band = external_data['normalized_flux']
                external_wavelengths_band = external_data['wavelengths']

                anchor_points = star.load_property('norm_anchor_wavelengths', epoch_num, 'COMBINED')
                if anchor_points is None:
                    anchor_points = []

                # --- NEW: If defaults exist from a previous band, rescale them ---
                new_image_height = image_data.shape[0]
                if previous_band_height is not None and default_bottom_spacial is not None:
                    # Compute the ratio between the new image height and the previous band's image height.
                    scale_factor = new_image_height / previous_band_height
                    # Scale the previously stored defaults.
                    default_bottom_spacial = int(default_bottom_spacial * scale_factor)
                    default_top_spacial = int(default_top_spacial * scale_factor)
                    default_include_spacial = (
                        int(default_include_spacial[0] * scale_factor),
                        int(default_include_spacial[1] * scale_factor)
                    )
                # Update previous_band_height for future rescaling.
                previous_band_height = new_image_height
                # --- End NEW rescaling ---

                # Call the interactive cleaning function
                (include_spacial,
                 bottom_spacial,
                 top_spacial,
                 navigation,
                 normalized_summed_flux,
                 returned_wavelengths_2D,
                 clean_flux_before_norm,
                 continuum_flux_interpolated) = clean_flux_and_normalize_interactive(
                    image_data,
                    wavelengths_2D,
                    external_normalized_flux_band,
                    external_wavelengths_band,
                    anchor_points,
                    band,
                    star_name,
                    epoch_num,
                    load_saved_flag,
                    star=star,
                    bottom_spacial=default_bottom_spacial,
                    top_spacial=default_top_spacial,
                    include_spacial=default_include_spacial,
                    snr_cache=snr_bounds_cache
                )

                print(f"Processed star: {star_name}, epoch: {epoch_num}, band: {band}")
                print(f"Selected spacial range (visual): {bottom_spacial} to {top_spacial}")
                print(f"Included spacial range (functional): {include_spacial}")

                # Save properties if "Finish and Next" was triggered.
                if navigation['finish']:
                    cleaned_normalized_flux = {
                        'wavelengths': returned_wavelengths_2D,
                        'normalized_flux': normalized_summed_flux
                    }
                    star.save_property('cleaned_normalized_flux', cleaned_normalized_flux, 
                                       epoch_num, band, overwrite=overwrite_flag, backup=backup_flag)
                    include_range_dict = {'bottom_include': include_spacial[0], 'top_include': include_spacial[1]}
                    star.save_property('include_range', include_range_dict, epoch_num, band,
                                       overwrite=overwrite_flag, backup=backup_flag)
                    spacial_range_dict = {'bottom_spacial': bottom_spacial, 'top_spacial': top_spacial}
                    star.save_property('spacial_range', spacial_range_dict, epoch_num, band,
                                       overwrite=overwrite_flag, backup=backup_flag)
                    star.save_property('clean_flux', {'clean_flux': clean_flux_before_norm,'wavelengths': returned_wavelengths_2D}, epoch_num, band,
                                       overwrite=overwrite_flag, backup=backup_flag)
                    star.save_property('interpolated_flux', {'interpolated_flux': continuum_flux_interpolated}, epoch_num, band,
                                       overwrite=overwrite_flag, backup=backup_flag)

                # Update the defaults for the next band with the current values
                default_include_spacial = include_spacial
                default_bottom_spacial = bottom_spacial
                default_top_spacial = top_spacial

                # Handle navigation flags.
                if navigation['prev_star'] or navigation['next_star']:
                    star_navigation_triggered = True
                    break  # Break out of band loop immediately

                if navigation['prev_band']:
                    current_band_idx = max(0, current_band_idx - 1)
                elif navigation['next_band'] or navigation['finish']:
                    current_band_idx += 1
                elif navigation['prev_epoch']:
                    epoch_navigation_triggered = True
                    break
                elif navigation['next_epoch']:
                    epoch_navigation_triggered = True
                    break
                else:
                    # Default: move to next band
                    current_band_idx += 1

            # End band loop

            # Break epoch loop if star-level navigation was triggered.
            if star_navigation_triggered:
                break

            if epoch_navigation_triggered:
                if navigation['prev_epoch']:
                    current_epoch_idx = max(0, current_epoch_idx - 1)
                elif navigation['next_epoch']:
                    current_epoch_idx += 1
                continue

            current_epoch_idx += 1

        # End epoch loop

        # Process star navigation based on the flag
        if star_navigation_triggered:
            if navigation['prev_star']:
                current_star_idx = max(0, current_star_idx - 1)
            elif navigation['next_star']:
                current_star_idx += 1
            continue
        else:
            current_star_idx += 1

    print("All stars have been processed.")


if __name__ == "__main__":
    main()
