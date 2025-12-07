import numpy as np
from scipy.interpolate import interp1d
import traceback
import multiprocessing
from ObservationClass import ObservationManager as obsm
from CCF import CCFclass
import datetime
import os

obs = obsm()

# Global constants
c_kms = 299792.458
DEFAULT_CROSS_VELO = 2000


def log_msg(filepath, msg):
    """Writes a message to the log file immediately."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    pid = multiprocessing.current_process().name
    formatted_msg = f"[{timestamp}] [{pid}] {msg}\n"

    # Opening in 'a' mode with immediate flush
    with open(filepath, "a") as f:
        f.write(formatted_msg)


def process_single_star(star_name, star_cfg, global_settings, run_ts, log_file):
    """
    Worker function with file logging.
    """
    # Unpack settings
    LINES_DEFAULT = global_settings["emission_lines_default"]
    FIT_FRAC_DEFAULT = global_settings.get("fit_fraction_default", 0.95)

    try:
        log_msg(log_file, f"STARTING {star_name}")

        perstar_lines = dict(LINES_DEFAULT)
        perstar_lines.update(star_cfg.get("emission_lines", {}))

        skip_eps_star = set(star_cfg.get("skip_epochs", []))
        skip_lines_star = star_cfg.get("skip_emission_lines", {})
        skip_cleaning_cfg = star_cfg.get("skip_cleaning", {})
        perstar_fit = star_cfg.get("fit_fraction", {})
        CrossVelo = star_cfg.get("cross_velo", DEFAULT_CROSS_VELO)

        if star_name == "HD 269891":
            CrossVelo = 1000

        # Load star
        log_msg(log_file, f"{star_name}: Loading star instance...")
        star = obs.load_star_instance(star_name, to_print=False)

        d0 = star.load_property(
            "cleaned_normalized_flux", 1, "COMBINED"
        ) or star.load_property("normalized_flux", 1, "COMBINED")
        if d0 is None:
            log_msg(log_file, f"{star_name}: [Skip] No epoch 1 template")
            return f"{star_name}: Skipped"

        bary = 0.0
        tpl_wave = d0["wavelengths"] * (1 + bary / c_kms)
        tpl_flux = d0["normalized_flux"]
        common_wavegrid = tpl_wave

        # Gather epochs
        all_epochs = star.get_all_epoch_numbers()
        epoch_nums = [ep for ep in all_epochs if ep not in skip_eps_star]

        log_msg(log_file, f"{star_name}: Interpolating {len(epoch_nums)} epochs...")

        obs_data_all = []
        for ep in epoch_nums:
            d = star.load_property(
                "cleaned_normalized_flux", ep, "COMBINED"
            ) or star.load_property("normalized_flux", ep, "COMBINED")
            if d is None:
                continue

            w_corr = d["wavelengths"]
            f_corr = d["normalized_flux"]
            mask = np.isfinite(w_corr) & np.isfinite(f_corr)
            w_corr, f_corr = w_corr[mask], f_corr[mask]

            if w_corr.size == 0:
                continue

            interp_flux = interp1d(
                w_corr, f_corr, kind="cubic", bounds_error=False, fill_value=1.0
            )(common_wavegrid)
            obs_data_all.append((ep, common_wavegrid, interp_flux))

        if not obs_data_all:
            log_msg(log_file, f"{star_name}: [Skip] No valid spectra")
            return f"{star_name}: Skipped"

        # Iterate emission lines
        total_lines = len(perstar_lines)
        for i, (line_tag, rng) in enumerate(perstar_lines.items()):
            log_msg(
                log_file,
                f"{star_name}: Processing line '{line_tag}' ({i+1}/{total_lines})...",
            )

            # 1. Filtering Logic
            skip_eps_for_line = skip_lines_star.get(line_tag, [])
            if isinstance(skip_eps_for_line, (int, np.integer)):
                skip_eps_for_line = [skip_eps_for_line]

            if 0 in skip_eps_for_line:
                log_msg(log_file, f"{star_name}: Skipped line {line_tag} (config)")
                continue

            skip_set = set(skip_eps_for_line)
            obs_data_line = [t for t in obs_data_all if t[0] not in skip_set]
            if not obs_data_line:
                continue

            # 2. Cleaning Logic
            clean_skip_list = skip_cleaning_cfg.get(line_tag, [])
            skip_cleaning_set = set()
            if 0 in clean_skip_list:
                skip_cleaning_set = {ep for (ep, _, _) in obs_data_line}
            else:
                skip_cleaning_set = set(clean_skip_list)

            fit_frac = perstar_fit.get(line_tag, FIT_FRAC_DEFAULT)

            # 3. Heavy Calculation: Double CCF
            log_msg(log_file, f"{star_name} | {line_tag}: Running Double CCF...")
            CCF_base = CCFclass(
                PlotAll=False,
                CrossVeloMin=-CrossVelo,
                CrossVeloMax=CrossVelo,
                Fit_Range_in_fraction=fit_frac,
                CrossCorRangeA=[rng],
                star_name=star_name,
                epoch=0,
                line_tag=line_tag,
                run_ts=run_ts,
                nm=False,
            )

            r1, r2, (co_wave, co_flux), failed_idx, _ew_meta = CCF_base.double_ccf(
                obs_data_line,
                tpl_wave,
                tpl_flux,
                return_coadd=True,
                return_meta=True,
                skip_clean_epochs=skip_cleaning_set,
            )

            # 4. Saving/Plotting
            log_msg(log_file, f"{star_name} | {line_tag}: Saving RVs and plotting...")

            for idx, ((ep, w, f), (RV_val, RV_err)) in enumerate(
                zip(obs_data_line, r2)
            ):
                EW = _ew_meta[idx]["EW"]
                sigEW = _ew_meta[idx]["sigma_EW"]
                SNR = _ew_meta[idx]["SNR"]
                detected = _ew_meta[idx]["detected"]

                # NOTE: IF IT HANGS, IT IS LIKELY HERE IN THE PLOTTING CLASS
                CCF_plot = CCFclass(
                    PlotAll=False,
                    CrossVeloMin=-CrossVelo,
                    CrossVeloMax=CrossVelo,
                    Fit_Range_in_fraction=fit_frac,
                    CrossCorRangeA=[rng],
                    star_name=star_name,
                    epoch=ep,
                    line_tag=line_tag,
                    savePlot=True,
                    run_ts=run_ts,
                    nm=False,
                )

                if co_wave is not None:
                    should_clean = ep not in skip_cleaning_set
                    _ = CCF_plot.compute_RV(w, f, co_wave, co_flux, clean=should_clean)

                RVs = star.load_property("RVs", ep, "COMBINED") or {}
                EWs = star.load_property("EWs", ep, "COMBINED") or {}

                RVs[line_tag] = {"full_RV": RV_val, "full_RV_err": RV_err}
                EWs[line_tag] = {
                    "EW": EW,
                    "sigma_EW": sigEW,
                    "SNR": SNR,
                    "detected": detected,
                }

                star.save_property("RVs", RVs, ep, overwrite=True, band="COMBINED")
                star.save_property("EWs", EWs, ep, overwrite=True, band="COMBINED")

            log_msg(log_file, f"{star_name} | {line_tag}: DONE")

        log_msg(log_file, f"FINISHED {star_name}")
        return f"{star_name}: Success"

    except Exception as e:
        log_msg(log_file, f"ERROR {star_name}: {e}")
        traceback.print_exc()
        return f"{star_name}: Failed ({e})"


def worker_wrapper(args):
    return process_single_star(*args)
