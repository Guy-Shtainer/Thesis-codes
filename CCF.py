"""
CCFclass
========
A minimal, array-only implementation of the Zucker & Mazeh (1994) /
Zucker et al. (2003) 1-D cross-correlation algorithm for stellar
radial-velocity work.

Public API
----------
compute_RV(obs_wave, obs_flux, tpl_wave, tpl_flux) -> (RV, σ)
double_ccf(obs_list, tpl_wave, tpl_flux)           -> (round1, round2)
    where   obs_list = [(wave1, flux1), (wave2, flux2), …]

No file I/O, no FITS header logic – just NumPy arrays in / out.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
from datetime import datetime
import re

clight = 2.9979e5  # km s⁻¹


class CCFclass:
    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        intr_kind: str = "cubic",
        Fit_Range_in_fraction: float = 0.95,
        CrossCorRangeA=((4000.0, 4500.0),),
        CrossVeloMin: float = -400.0,
        CrossVeloMax: float = 400.0,
        PlotFirst: bool = False,
        PlotAll: bool = False,
        star_name: str | None = None,
        epoch: str | int | None = None,
        spectrum: str | int | None = None,
        line_tag: str = "",
        savePlot: bool = False,
        run_ts: str = "",
        nm: bool = True,
    ):
        # ---- original parameters --------------------------------------
        self.intr_kind = intr_kind
        self.Fit_Range_in_fraction = Fit_Range_in_fraction
        self.CrossCorRangeA = np.asarray(CrossCorRangeA, float)
        self.S2Nrange = [[445.0, 445.5]]
        self.CrossVeloMin = CrossVeloMin
        self.CrossVeloMax = CrossVeloMax
        self.PlotFirst = PlotFirst
        self.PlotAll = PlotAll
        self.spectrum = spectrum
        self._first_done = not PlotFirst
        self.savePlot = savePlot

        # ---- new contextual metadata ----------------------------------
        self.star_name = star_name or "unknown‑star"
        self.epoch = epoch
        self.line_tag = line_tag
        self.run_ts = run_ts
        self.nm = nm

    # ------------------------------------------------------------------ #
    # static helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _CCF(f1, f2, N):
        """Normalised dot-product."""
        return np.sum(f1 * f2) / np.std(f1) / np.std(f2) / N

    # ------------------------------------------------------------------ #
    # internal core: parabola-fit cross-correlation                       #
    # ------------------------------------------------------------------ #
    def _crosscorreal(
        self,
        Observation,
        Mask,
        CrossCorInds,
        sRange,
        N,
        veloRange,
        wavegridlog,
        obs_plot_clean=None,  # cleaned, NOT mean-subtracted
        obs_plot_raw=None,  # raw,    NOT mean-subtracted
    ):
        wavegridlog = wavegridlog*10
        CCFarr = np.array(
            [
                self._CCF(np.copy(Observation), (np.roll(Mask, s))[CrossCorInds], N)
                for s in sRange
            ]
        )
        IndMax = np.argmax(CCFarr)
        # CCFarr = CCFarr[:IndMax].concatenate(CCFarr[IndMax+1:])
        # print(f'CCFMax is {CCFarr[IndMax]} at index {IndMax}')
        # CCFarr[IndMax] = np.average([CCFarr[IndMax-3:IndMax-1],CCFarr[IndMax+2:IndMax+4]])
        CCFMAX1 = np.average(
            [CCFarr[IndMax - 3 : IndMax - 1], CCFarr[IndMax + 2 : IndMax + 4]]
        )
        # print(f"After removing local max, CCFMax is {CCFarr[IndMax]} at index {IndMax}")
        # vmax = veloRange[IndMax]
        # CCFMAX1  = CCFarr[IndMax]
        # CCFarr = CCFarr[:IndMax - 1].concatenate(CCFarr[IndMax + 2 :])
        # IndMax = np.argmax(CCFarr)

        # edges at fitfac·CCFMAX1
        LeftEdgeArr = np.abs(self.Fit_Range_in_fraction * CCFMAX1 - CCFarr[:IndMax])
        RightEdgeArr = np.abs(
            self.Fit_Range_in_fraction * CCFMAX1 - CCFarr[IndMax + 1 :]
        )

        if len(LeftEdgeArr) == 0 or len(RightEdgeArr) == 0:
            print("Can't find local maximum in CCF\n")
            fig1, ax1 = plt.subplots()
            ax1.plot(veloRange, CCFarr, color='C0', label=f'obs {self.star_name}')
            plt.show()
            return np.array([None, None, None, None])

        IndFit1 = np.argmin(LeftEdgeArr)
        IndFit2 = np.argmin(RightEdgeArr) + IndMax + 1
        a, b, c = np.polyfit(
            np.concatenate(
                (veloRange[IndFit1:IndMax], veloRange[IndMax + 1 : IndFit2 + 1])
            ),
            np.concatenate((CCFarr[IndFit1:IndMax], CCFarr[IndMax + 1 : IndFit2 + 1])),
            2,
        )
        vmax = -b / (2 * a)
        CCFAtMax = min(1 - 1e-20, c - b**2 / 4.0 / a)
        FineVeloGrid = np.arange(veloRange[IndFit1], veloRange[IndFit2], 0.1)
        parable = a * FineVeloGrid**2 + b * FineVeloGrid + c
        sigma = np.sqrt(-1.0 / (N * 2 * a * CCFAtMax / (1 - CCFAtMax**2)))

        if self.PlotFirst or self.PlotAll or self.savePlot:

            # -------- 0.  Gather & format metadata --------------------------
            RV = vmax
            RV_error = sigma
            star_name = getattr(self, "star_name", "unknown").strip()
            epoch = getattr(self, "epoch", None)
            spectrum = getattr(self, "spectrum", None)
            line_rng = self.CrossCorRangeA[0]  # first interval
            line_tag = getattr(self, "line_tag", "")

            # units for labels based on nm flag
            wave_units = "nm" if self.nm else "Å"
            line_txt = (
                f"{line_tag}  ({line_rng[0]*10:.0f}–{line_rng[1]*10:.0f} {wave_units})"
                if line_tag
                else f"{line_rng[0]*10:.0f}–{line_rng[1]*10:.0f} {wave_units}"
            )

            # epoch/spectrum label parts
            epoch_txt = f"Epoch {epoch}" if epoch is not None else "Epoch ?"
            spec_txt = f"  |  Spec {spectrum}" if spectrum is not None else ""

            # safe strings for filenames
            clean_star = re.sub(r"[^A-Za-z0-9_-]", "_", star_name)
            epoch_str = str(epoch) if epoch is not None else "NA"
            spec_str = (
                f"_S{int(spectrum)}"
                if isinstance(spectrum, (int, np.integer))
                else (f"_S{spectrum}" if spectrum is not None else "")
            )
            rv_tag = f"{RV:+.1f}".replace("+", "p").replace("-", "m")

            # -------- 1.  CCF figure ----------------------------------------
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(veloRange, CCFarr, label="CCF", color="C0")
            # Add both horizontal lines for original and averaged max
            ax1.axhline(
                y=CCFarr[IndMax],
                color="red",
                linestyle=":",
                label=f"Original Max ({CCFarr[IndMax]:.3f})",
                alpha=0.7,
            )
            ax1.axhline(
                y=CCFMAX1,
                color="gray",
                linestyle="--",
                label=f"Averaged Max ({CCFMAX1:.3f})",
                alpha=0.7,
            )
            ax1.axhline(
                y=self.Fit_Range_in_fraction * CCFMAX1,
                color="blue",
                linestyle="-.",
                label=f"Fit Range ({self.Fit_Range_in_fraction:.2f}×avg_max)",
                alpha=0.7,
            )
            # Add horizontal line at fit range fraction
            ax1.axhline(
                y=self.Fit_Range_in_fraction * CCFMAX1,
                color="gray",
                linestyle="--",
                label=f"Fit Range ({self.Fit_Range_in_fraction:.2f}×max)",
                alpha=0.7,
            )

            # Mark fit range points
            ax1.plot(
                [veloRange[IndFit1], veloRange[IndFit2]],
                [CCFarr[IndFit1], CCFarr[IndFit2]],
                "go",
                label="Fit edges",
            )
            ax1.plot(FineVeloGrid, parable, label="Fit (parabola)", color="C1", lw=1.5)
            ax1.axvline(
                RV, ls="--", color="r", label=f"RV = {RV:.2f} ± {RV_error:.2f} km/s"
            )

            # ax1.set_title(f"CCF  |  {star_name}  |  Epoch {epoch_str}  |  {line_txt}",
            #               fontsize=14, weight='bold')
            ax1.set_title(
                f"CCF  |  {star_name}  |  {epoch_txt}{spec_txt}  |  {line_txt}",
                fontsize=14,
                weight="bold",
            )
            ax1.set_xlabel("Radial Velocity [km/s]")
            ax1.set_ylabel("Normalized CCF")
            ax1.grid(ls="--", alpha=0.4)
            ax1.legend()
            plt.tight_layout()
            if self.PlotFirst or self.PlotAll:
                plt.show()
            else:
                plt.close(fig1)

            # -------- 2.  Spectrum vs template (overlay; zero-mean all) -----------
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # zero-mean helper
            def _zm(y):
                if y is None:
                    return None
                m = np.nanmean(y)
                return y - (0.0 if (m is None or not np.isfinite(m)) else m)

            # zero-mean versions for a fair visual comparison
            y_raw   = _zm(obs_plot_raw)
            y_clean = _zm(obs_plot_clean)
            y_mask  = Mask - np.mean(Mask)  # already zero-mean

            # Raw first (blue), then cleaned (green)
            if y_raw is not None:
                ax2.plot(
                    wavegridlog[CrossCorInds], y_raw,
                    label="Observation (raw)", color="steelblue", alpha=0.70,
                )
            if y_clean is not None:
                ax2.plot(
                    wavegridlog[CrossCorInds], y_clean,
                    label="Observation (cleaned)", color="forestgreen", alpha=0.95,
                )
            else:
                # fallback: what CCF used (zero-mean)
                ax2.plot(
                    wavegridlog[CrossCorInds], Observation,
                    label="Observation (cleaned, zero-mean)", color="forestgreen", alpha=0.95,
                )

            # Templates (zero-mean)
            ax2.plot(
                wavegridlog, y_mask,
                label="Template (unshifted)", color="orchid", alpha=0.9,
            )
            ax2.plot(
                wavegridlog * (1 + RV / clight), y_mask,
                label="Template (shifted)", color="turquoise", alpha=0.9,
            )

            ax2.set_title(
                f"Spectra  |  {star_name}  |  {epoch_txt}{spec_txt}  |  {line_txt}",
                fontsize=14, weight="bold",
            )
            ax2.set_xlabel(rf"Wavelength [{wave_units}]")
            ax2.set_ylabel("Normalized Flux (zero-mean)")
            ax2.grid(ls="--", alpha=0.4)
            ax2.legend()
            plt.tight_layout()

            # -------- 3.  Optional saving -----------------------------------
            if getattr(self, "savePlot", False):
                ts_str = self.run_ts
                out_dir = Path("../output") / clean_star / "CCF" / ts_str / line_tag
                out_dir.mkdir(parents=True, exist_ok=True)

                # fig1.savefig(out_dir / f"{clean_star}_MJD{epoch_str}_RV{rv_tag}_CCF.png", dpi=150)
                # fig2.savefig(out_dir / f"{clean_star}_MJD{epoch_str}_RV{rv_tag}_SPEC.png", dpi=150)
                fig1.savefig(
                    out_dir
                    / f"{clean_star}_MJD{epoch_str}{spec_str}_RV{rv_tag}_CCF.png",
                    dpi=150,
                )
                fig2.savefig(
                    out_dir
                    / f"{clean_star}_MJD{epoch_str}{spec_str}_RV{rv_tag}_SPEC.png",
                    dpi=150,
                )

                print(f"[saved] plots to {out_dir}")

            if self.PlotFirst or self.PlotAll:
                plt.show()
            else:
                plt.close(fig2)
            self.PlotFirst = False

        if CCFAtMax > 1:
            print("Failed to cross-correlate: template probably sucks!")
            print("Check cross-correlation function + parable fit.")
            return None, None

        CFFdvdvAtMax = 2 * a
        return np.array(
            [vmax, np.sqrt(-1.0 / (N * CFFdvdvAtMax * CCFAtMax / (1 - CCFAtMax**2)))]
        )

        # ------------------------------------------------------------------ #

    # EW + SNR helpers                                                    #
    # ------------------------------------------------------------------ #
    def _estimate_snr_robust(self, w, f, for_ew: bool = False):
        """
        SNR = 1/std.

        If for_ew=True:
            Calculates noise in the immediate continuum 'shoulders' of the line.
            It looks at a 5 Å (or 0.5 nm) window to the LEFT and to the RIGHT
            of each interval in self.CrossCorRangeA.

            Windows: [λ_start - Δ, λ_start]  AND  [λ_end, λ_end + Δ]

        Otherwise:
            Falls back to self.S2Nrange (user defined fixed continuum).
        """
        # --- choose where to measure the noise ---
        if for_ew:
            width = 0.5 if self.nm else 5.0  # 0.5 nm ≡ 5 Å
            search_ranges = []

            # Create a Left and Right window for every line range
            for r in np.atleast_2d(self.CrossCorRangeA):
                # Left shoulder: [Start - width, Start]
                search_ranges.append([r[0] - width, r[0]])
                # Right shoulder: [End, End + width]
                search_ranges.append([r[1], r[1] + width])
        else:
            search_ranges = self.S2Nrange

        # --- measure noise as std in each available window ---
        noises = []
        for lo, hi in search_ranges:
            # Ensure order
            if hi <= lo:
                lo, hi = hi, lo

            # Create mask for this small window
            m = (w > lo) & (w < hi)

            # Only calculate if we have enough points (e.g. > 3 pixels)
            if np.count_nonzero(m) > 3:
                noises.append(float(np.std(f[m])))

        # --- Aggregation ---
        if len(noises) == 0:
            # Fallback 1: Try to use everything NOT in the line ranges
            line_mask = np.zeros_like(w, dtype=bool)
            for r in np.atleast_2d(self.CrossCorRangeA):
                line_mask |= (w > r[0]) & (w < r[1])
            cont = ~line_mask

            if np.count_nonzero(cont) > 10:
                noise = float(np.std(f[cont]))
            else:
                # Fallback 2: Global STD (Desperation mode)
                noise = float(np.std(f))
        else:
            # We use Median instead of Mean.
            # Why? If the Left window is clean but the Right window hits a cosmic ray,
            # the Mean would be skewed high. The Median will pick the cleaner value.
            # (If there are only 2 values, Median equals Mean, which is also fine).
            noise = float(np.min(noises))

        # Avoid division by zero
        snr = (1.0 / noise) if (noise > 1e-9 and np.isfinite(noise)) else 0.0
        return snr, noise

    def _ew_sigma_rule_of_thumb(self, w, f):
        """
        Compute emission EW over CrossCorRangeA with continuum=1,
        and the rule-of-thumb sigma(EW) using SNR from a 5 Å window
        immediately LEFT of each line interval.
        Returns (EW, sigma_EW, SNR).
        """
        # ensure increasing wavelength for integration
        if w[0] > w[-1]:
            idx = np.argsort(w)
            w = w[idx]
            f = f[idx]

        total_EW = 0.0
        sum_dlam2N = 0.0  # for sigma(EW) = sqrt(sum(dλ^2 N)) / SNR

        for r in np.atleast_2d(self.CrossCorRangeA):
            m = (w > r[0]) & (w < r[1])
            if np.count_nonzero(m) < 2:
                continue
            dlam = np.median(np.diff(w[m]))
            total_EW += np.trapz(f[m] - 1.0, w[m], dx=dlam)
            Nk = np.count_nonzero(m)
            sum_dlam2N += (dlam * dlam) * Nk

        # *** NEW: SNR from left-of-line, 5 Å (0.5 nm if self.nm=True) ***
        snr, _ = self._estimate_snr_robust(w, f, for_ew=True)

        sigma = (
            (np.sqrt(sum_dlam2N) / snr)
            if (sum_dlam2N > 0 and np.isfinite(snr))
            else np.nan
        )
        return total_EW, sigma, snr

    def _ew_gate(self, w, f, ksig=10.0):
        """
        Returns a dict with EW gate results for one epoch.
        """
        EW, sigEW, SNR = self._ew_sigma_rule_of_thumb(w, f)
        detected = bool(
            np.isfinite(EW) and np.isfinite(sigEW) and (EW - ksig * sigEW) > 0.0
        )
        return {"EW": EW, "sigma_EW": sigEW, "SNR": SNR, "detected": detected}

    # ------------------------------------------------------------------ #
    # public: single-spectrum RV                                          #
    # ------------------------------------------------------------------ #
    def compute_RV(
        self,
        obs_wave,
        obs_flux,
        tpl_wave,
        tpl_flux,
        clean=True,  # << NEW: clean inside compute_RV
        clean_kwargs=None,  # << NEW: overrides for the cleaner
    ):

        """
        Parameters
        ----------
        obs_wave, obs_flux : arrays – observation (λ in Å, normalised flux)
        tpl_wave, tpl_flux : arrays – template / mask

        Returns
        -------
        (RV_km_s, σ_km_s)
        """

        # ----- build common logarithmic grid (match instructor logic) --------
        LambdaRangeUser = self.CrossCorRangeA * np.array(
            [1 - 1.1 * self.CrossVeloMax / clight, 1 - 1.1 * self.CrossVeloMin / clight]
        )

        LamRangeB = LambdaRangeUser[0, 0]
        LamRangeR = LambdaRangeUser[-1, 1]

        Dlam = obs_wave[1] - obs_wave[0]
        Resolution = obs_wave[1] / Dlam  # instructor: Resolution (λ/Δλ)
        vbin = clight / Resolution  # identical formula

        Nwaves = int(np.log(LamRangeR / LamRangeB) / np.log(1.0 + vbin / clight))

        wavegridlog = LamRangeB * (1.0 + vbin / clight) ** np.arange(Nwaves)

        IntIs = np.array(
            [
                np.argmin(np.abs(wavegridlog - self.CrossCorRangeA[i][0]))
                for i in np.arange(len(self.CrossCorRangeA))
            ]
        )
        IntFs = np.array(
            [
                np.argmin(np.abs(wavegridlog - self.CrossCorRangeA[i][1]))
                for i in np.arange(len(self.CrossCorRangeA))
            ]
        )

        Ns = (
            IntFs - IntIs
        )  # number of points in range. if there are several ranges at once it accounts for them
        N = np.sum(Ns)  # relevant in case i pass several emission lines ranges

        CrossCorInds = np.concatenate(
            ([np.arange(IntIs[i], IntFs[i]) for i in np.arange(len(IntFs))])
        )  # Find the indices which are the emission line
        sRange = np.arange(
            int(self.CrossVeloMin / vbin), int(self.CrossVeloMax / vbin) + 1, 1
        )
        veloRange = vbin * sRange

        MaskAll = np.array([tpl_wave, tpl_flux]).T
        Mask = interp1d(
            MaskAll[:, 0],
            np.nan_to_num(MaskAll[:, 1]),
            bounds_error=False,
            fill_value=1.0,
            kind=self.intr_kind,
        )(wavegridlog)
        # print(f'plotting new mask')
        # plt.plot(wavegridlog, Mask, 'k')

        # Clean spikes using rolling window
        # window_size = 20
        # sigma_thresh = 3

        # # Clean Mask
        # for i in range(len(Mask) - window_size):
        #     window = Mask[i : i + window_size]
        #     window_mean = np.mean(window)
        #     window_std = np.std(window)
        #     # if self.epoch == 6:
        #     # print(f'window mean is {window_mean}, window std is {window_std}')

        #     # Check each point in the window
        #     for j in range(window_size):
        #         # if self.epoch == 6:
        #         # print(f'checking Mask at index {i+j}, value {Mask[i+j]}')
        #         if i + j >= len(Mask):
        #             break
        #         if (
        #             Mask[i + j] > window_mean + sigma_thresh * window_std
        #             or Mask[i + j] < window_mean - sigma_thresh * window_std
        #         ):
        #             Mask[i + j] = (
        #                 Mask[max(0, i + j - 1)] + Mask[min(len(Mask) - 1, i + j + 1)]
        #             ) / 2

        # # Clean flux
        # for i in range(len(flux) - window_size):
        #     window = flux[i : i + window_size]
        #     window_mean = np.mean(window)
        #     window_std = np.std(window)

        #     # Check each point in the window
        #     for j in range(window_size):
        #         if i + j >= len(flux):
        #             break
        #         if (
        #             flux[i + j] > window_mean + sigma_thresh * window_std
        #             or flux[i + j] < window_mean - sigma_thresh * window_std
        #         ):
        #             flux[i + j] = (
        #                 flux[max(0, i + j - 1)] + flux[min(len(flux) - 1, i + j + 1)]
        #             ) / 2

        # ================= CLEANING STEP (NEW) =================
        # Keep the raw flux as provided to this method
        obs_flux_raw = np.asarray(obs_flux, dtype=float)

        if ('C IV 5801-5812' in self.line_tag) and True:
            parts_to_split = 8
            deg_to_use = 6
            sigma_pos = 7
            sigma_neg = 8
        else:
            parts_to_split = 8
            deg_to_use = 6
            sigma_neg=4
            sigma_pos=5

        # Defaults that mimic your double_ccf cleaner setup
        default_clean_kwargs = dict(
            focus_range=None,  # uses self.CrossCorRangeA internally
            n_iter=100,
            n_stages=20,
            n_split=parts_to_split,
            sample_frac=0.9,
            deg=deg_to_use,
            sigma_clip_neg=sigma_neg,
            sigma_clip_pos=sigma_pos,
            random_state=42,
            plot=False,  # avoid extra cleaning plots here
            add_noise=True,
            noise_source="leftwin",
        )
        if clean_kwargs is not None:
            default_clean_kwargs.update(clean_kwargs)

        if clean:
            obs_flux_clean, _, _ = self.clean_line_with_iterative_poly(
                wave=obs_wave,
                flux=obs_flux_raw,
                **default_clean_kwargs,
            )
        else:
            obs_flux_clean = obs_flux_raw.copy()
        # ======================================================

        # Interpolate CLEANED flux to the log grid segment used for CCF
        flux_ccf = interp1d(
            obs_wave,
            np.nan_to_num(obs_flux_clean),
            bounds_error=False,
            fill_value=1.0,
            kind="cubic",
        )(wavegridlog[CrossCorInds])

        # Build plotting copies (not mean-subtracted)
        obs_plot_clean = interp1d(
            obs_wave,
            np.nan_to_num(obs_flux_clean),
            bounds_error=False,
            fill_value=1.0,
            kind="cubic",
        )(wavegridlog[CrossCorInds])

        obs_plot_raw = interp1d(
            obs_wave,
            np.nan_to_num(obs_flux_raw),
            bounds_error=False,
            fill_value=1.0,
            kind="cubic",
        )(wavegridlog[CrossCorInds])

        # ---- run CCF on zero-mean CLEANED segment ----
        CCFeval = self._crosscorreal(
            np.copy(flux_ccf - np.mean(flux_ccf)),
            np.copy(Mask - np.mean(Mask)),
            CrossCorInds,
            sRange,
            N,
            veloRange,
            wavegridlog,
            obs_plot_clean=obs_plot_clean,  # << show cleaned
            obs_plot_raw=obs_plot_raw,  # << also show raw
        )
        return CCFeval[0], CCFeval[1]

    def clean_line_with_iterative_poly(
        self,
        wave,
        flux,
        focus_range=None,
        n_iter=300,
        n_stages=3,
        sample_frac=0.7,
        deg=5,
        sigma_clip_pos=3.0,
        sigma_clip_neg=2.0,
        random_state=None,
        plot=True,
        ax=None,
        add_noise=True,
        noise_source="residual",
        noise_floor=1e-6,
        n_split=1,  # <--- NEW ARGUMENT
    ):
        """
        Staged cleaning with asymmetric σ-clipping.
        Supports splitting the range into n_split segments for piecewise fitting.
        """
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)

        # 1. Determine base ranges (usually the whole CrossCorRangeA)
        if focus_range is None:
            ranges = [
                tuple(
                    [
                        r[0] * (1 - 1.1 * self.CrossVeloMax / clight),
                        r[1] * (1 - 1.1 * self.CrossVeloMin / clight),
                    ]
                )
                for r in np.atleast_2d(self.CrossCorRangeA)
            ]
        else:
            lo, hi = focus_range
            if hi <= lo:
                raise ValueError("focus_range must satisfy lo < hi.")
            ranges = [(lo, hi)]

        # 2. NEW: Apply Splitting Logic
        # We subdivide the base ranges into n_split smaller chunks
        processing_ranges = []
        for start, end in ranges:
            if n_split > 1:
                # Calculate split points (e.g., 5 parts = 6 boundaries)
                boundaries = np.linspace(start, end, n_split + 1)
                for i in range(n_split):
                    # Create segment (b_i, b_{i+1})
                    # Note: Adjacent segments share a boundary point.
                    # This ensures no data gaps.
                    processing_ranges.append((boundaries[i], boundaries[i + 1]))
            else:
                processing_ranges.append((start, end))

        # Setup outputs
        combined_model = np.full_like(flux, np.nan, dtype=float)
        combined_repl = np.zeros_like(flux, dtype=bool)
        cleaned_flux = flux.copy()
        rng = np.random.default_rng(random_state)

        # --- Internal Worker Function (Unchanged) ---
        def _clean_one_range_staged(w, f_in, lo, hi):
            in_mask = (w >= lo) & (w <= hi)
            # Safety check for too few points
            if np.count_nonzero(in_mask) < (deg + 2):
                return (
                    f_in.copy(),
                    np.full_like(f_in, np.nan, dtype=float),
                    np.zeros_like(f_in, dtype=bool),
                )

            xi = w[in_mask]
            yi = f_in[in_mask].copy()
            M = xi.size

            # per-iteration sample size
            k = max(3 * deg + 1, int(np.ceil(sample_frac * M)))
            k = min(k, M)

            # normalize x
            xc = xi.mean()
            xs = xi.std() if xi.std() > 0 else 1.0
            xi0 = (xi - xc) / xs

            n_stages_eff = max(1, int(n_stages))
            it_per_stage = max(1, int(np.ceil(n_iter / n_stages_eff)))

            replaced_union = np.zeros_like(f_in, dtype=bool)
            model_full_final = np.full_like(f_in, np.nan, dtype=float)

            for stage in range(n_stages_eff):
                # --- average model for this stage ---
                preds = np.empty((it_per_stage, M), dtype=float)
                for t in range(it_per_stage):
                    idx = rng.choice(M, size=k, replace=False)
                    coeffs = np.polyfit(xi0[idx], yi[idx], deg=deg)
                    preds[t] = np.polyval(coeffs, xi0)
                model_i = preds.mean(axis=0)

                # --- robust residual σ & asymmetric clipping ---
                res = yi - model_i
                med = np.median(res)
                mad = np.median(np.abs(res - med))
                sigma = (
                    1.4826 * mad
                    if mad > 0
                    else (np.std(res) if np.std(res) > 0 else 0.0)
                )
                sigma = max(float(sigma), float(noise_floor))

                r = res - med
                outliers_local = (r > sigma_clip_pos * sigma) | (
                    r < -sigma_clip_neg * sigma
                )

                # --- choose noise scale for inserts ---
                if noise_source == "residual":
                    noise_sigma = sigma
                elif noise_source == "leftwin":
                    _, noise_sigma = self._estimate_snr_robust(w, f_in, for_ew=True)
                elif noise_source == "local":
                    noise_sigma = float(np.std(f_in[in_mask]))
                elif noise_source == "global":
                    noise_sigma = float(np.std(f_in))
                else:
                    noise_sigma = sigma
                noise_sigma = max(float(noise_sigma), float(noise_floor))

                # --- replace and continue ---
                if np.any(outliers_local):
                    if add_noise:
                        yi[outliers_local] = model_i[outliers_local] + rng.normal(
                            0.0, noise_sigma, size=int(outliers_local.sum())
                        )
                    else:
                        yi[outliers_local] = model_i[outliers_local]

                # accumulate to full-length arrays
                tmp_mask = np.zeros_like(f_in, dtype=bool)
                tmp_mask[in_mask] = outliers_local
                replaced_union |= tmp_mask

                if stage == n_stages_eff - 1:
                    model_full_final[in_mask] = model_i

                if not np.any(outliers_local) and stage < n_stages_eff - 1:
                    model_full_final[in_mask] = model_i
                    break

            f_out = f_in.copy()
            f_out[in_mask] = yi
            return f_out, model_full_final, replaced_union

        # 3. Clean all ranges sequentially on evolving flux
        for lo, hi in processing_ranges:
            cleaned_flux, model_full, replaced_full = _clean_one_range_staged(
                wave, cleaned_flux, lo, hi
            )
            # Merge models into the combined array
            msel = ~np.isnan(model_full)
            combined_model[msel] = model_full[msel]
            combined_repl |= replaced_full

        # --- plotting (focused; always when plot=True) ---
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 5))
            units = "nm" if self.nm else "Å"

            # For plotting, we just show the full extent
            full_extent_ranges = (
                [tuple(r) for r in np.atleast_2d(self.CrossCorRangeA)]
                if focus_range is None
                else [
                    (
                        min(focus_range[0], focus_range[1]),
                        max(focus_range[0], focus_range[1]),
                    )
                ]
            )

            mask_total = np.zeros_like(wave, dtype=bool)
            for lo, hi in full_extent_ranges:
                mask_total |= (wave >= lo) & (wave <= hi)

            f_plot = flux.copy()
            f_plot[~mask_total] = np.nan
            c_plot = cleaned_flux.copy()
            c_plot[~mask_total] = np.nan
            m_plot = combined_model.copy()

            ax.plot(wave, f_plot, label="original", alpha=0.8, lw=1)
            if np.any(~np.isnan(m_plot)):
                ax.plot(
                    wave,
                    m_plot,
                    ls="--",
                    color="black",
                    alpha=0.6,
                    label=f"model (split={n_split}, deg={deg})",
                )
            ax.plot(wave, c_plot, alpha=0.9, label="cleaned", lw=1.2)

            repl_in = combined_repl & mask_total
            if np.any(repl_in):
                ax.scatter(
                    wave[repl_in],
                    flux[repl_in],
                    s=18,
                    color="red",
                    label="replaced",
                    zorder=5,
                )

            # Draw vertical lines to show split points if split > 1
            if n_split > 1:
                for start, end in ranges:
                    boundaries = np.linspace(start, end, n_split + 1)
                    for b in boundaries[1:-1]:  # Skip start/end
                        ax.axvline(b, color="gray", linestyle=":", alpha=0.5)

            lo_all = min(r[0] for r in full_extent_ranges)
            hi_all = max(r[1] for r in full_extent_ranges)
            pad = (
                0.05 * (hi_all - lo_all)
                if hi_all > lo_all
                else (0.5 if self.nm else 5.0)
            )
            ax.set_xlim(lo_all - pad, hi_all + pad)
            ax.set_xlabel(f"Wavelength [{units}]")
            ax.set_ylabel("Normalized flux")
            ax.set_title(
                f"Cleaning {self.star_name} | {self.line_tag or 'emission line'}"
            )
            ax.legend()
            ax.grid(ls="--", alpha=0.35)
            plt.tight_layout()

            if self.savePlot:
                clean_star = re.sub(
                    r"[^A-Za-z0-9_-]", "_", (self.star_name or "unknown")
                )
                epoch_str = "NA" if self.epoch is None else str(self.epoch)
                spec_str = "" if self.spectrum is None else (f"_S{self.spectrum}")
                out_dir = Path("../output") / clean_star / "CLEAN" / (self.run_ts or "")
                out_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    out_dir
                    / f"{clean_star}_MJD{epoch_str}{spec_str}_{self.line_tag or 'line'}_CLEAN.png",
                    dpi=150,
                )
            plt.show()
            try:
                plt.close(fig)
            except NameError:
                pass

        return cleaned_flux, combined_model, combined_repl

    # ------------------------------------------------------------------ #
    # public: two-pass RV + coadd                                         #
    # ------------------------------------------------------------------ #
    def double_ccf(
        self,
        obs_list,
        tpl_wave,
        tpl_flux,
        return_coadd=False,
        return_meta=False,
        skip_clean_epochs=None,
    ):
        """
        Two-pass CCF with EW-based epoch gating.

        Gate rule (emission): keep epoch if EW - 5*sigma(EW) > 0,
        where sigma(EW) ≈ sqrt(sum_k N_k dλ_k^2) / SNR and SNR = 1/std
        in self.S2Nrange.

        Returns (backwards compatible by default):
            if not return_coadd and not return_meta:
                -> (round1, round2)
            if return_coadd and not return_meta:
                -> (round1, round2, (wave_coadd, flux_coadd))
            if not return_coadd and return_meta:
                -> (round1, round2, failed_indices)
            if return_coadd and return_meta:
                -> (round1, round2, (wave_coadd, flux_coadd), failed_indices, ew_meta)
        """
        # Handle default None
        if skip_clean_epochs is None:
            skip_clean_epochs = set()
        else:
            skip_clean_epochs = set(skip_clean_epochs)

        cleaned_obs_list = []
        for i, (ep, w, f) in enumerate(obs_list):
            if ep in skip_clean_epochs:
                print(f"Info: Skipping cleaning for epoch {ep}")
                f_clean = f
            else:
                # --- NEW: Check if this is the problematic line ---
                # Note: 'self.line_tag' stores the current line being processed
                if ('C IV 5801-5812' in self.line_tag):
                    parts_to_split = 8
                    deg_to_use = 6
                    sigma_pos = 7
                    sigma_neg = 8
                else:
                    parts_to_split = 8
                    deg_to_use = 6
                    sigma_neg=4
                    sigma_pos=5

                f_clean, _, _ = self.clean_line_with_iterative_poly(
                    wave=w,
                    flux=f,
                    n_split=parts_to_split,  # <--- Pass the new argument here
                    
                    # Your other robust settings from before:
                    n_iter=100,          
                    n_stages=20,
                    sample_frac=0.9,    
                    deg=deg_to_use, # You might be able to LOWER this if you are splitting!
                            # e.g., deg=5 might be enough if you split into 5 small chunks
                    sigma_clip_neg=sigma_neg,  
                    sigma_clip_pos=sigma_pos,  
                    random_state=42,
                    noise_source="leftwin",
                    plot=False,
                )
                # print(f"cleaned epoch {ep} of line {self.line_tag}")

            cleaned_obs_list.append((ep, w, f_clean))

        obs_list = cleaned_obs_list
        # ---------- 0) EW gate per epoch ----------
        ew_meta = []
        include_mask = []
        failed_indices = []
        for ep, w, f in obs_list:
            info = self._ew_gate(w, f, ksig=10.0)
            ew_meta.append(info)
            ok = bool(info["detected"])
            include_mask.append(ok)
            if not ok:
                failed_indices.append(ep)

        # keep for later access even if caller doesn't request it
        self.last_failed_indices = failed_indices

        # ---------- 1) First pass (RV) ----------
        r1 = []
        S2N_all = []
        print("Calculating first-pass RVs (EW-gated)…")
        for i, (ep, w, f) in enumerate(obs_list):
            if include_mask[i]:
                rv, sig = self.compute_RV(w, f, tpl_wave, tpl_flux, clean=False)
            else:
                rv, sig = (None, None)
            r1.append((rv, sig))

            # SNR for weights (re-use same definition SNR=1/std)
            snr, _ = self._estimate_snr_robust(w, f)
            S2N_all.append(snr)

        # ---------- 2) Coadd (only included epochs) ----------
        idx_keep = [i for i, ok in enumerate(include_mask) if ok]
        if len(idx_keep) == 0:
            print("[EW gate] No usable epochs: coadd and round-2 RVs not computed.")
            r2 = [(None, None) for _ in obs_list]
            coadd_pair = (None, None)
            if return_coadd and return_meta:
                return (r1, r2, coadd_pair, failed_indices, ew_meta)
            elif return_coadd:
                return (r1, r2, coadd_pair)
            elif return_meta:
                return (r1, r2, failed_indices)
            else:
                return (r1, r2)

        # choose reference wavelength grid from the first included epoch
        w_ref = obs_list[idx_keep[0]][1]
        w_common = w_ref * 10 if self.nm else w_ref
        coadd = np.zeros_like(w_common)

        # weights from included-only SNRs, normalized to sum=1 over included
        S2N_included = np.asarray([S2N_all[i] for i in idx_keep])
        wts = S2N_included**2
        wts /= np.sum(wts)

        print("Building coadded template (included epochs only)…")
        for wgt, i in zip(wts, idx_keep):
            (ep, w, f), (rv, _) = obs_list[i], r1[i]
            # rv should not be None here, but guard anyway
            if rv is None:
                continue
            shifted = interp1d(
                w * (1 - rv / clight),
                f,
                kind=self.intr_kind,
                bounds_error=False,
                fill_value=1.0,
            )(w_common)
            coadd += wgt * shifted

        # ---------- 3) Second pass (only included epochs) ----------
        print("Finally calculating CCF using coadded template (EW-gated)…")
        r2_full = [None] * len(obs_list)
        for i in idx_keep:
            ep, w, f = obs_list[i]
            r2_full[i] = self.compute_RV(w, f, w_common, coadd, clean=False)

        # Fill excluded ones with (None, None) so the output lines up with obs_list
        r2 = [val if val is not None else (None, None) for val in r2_full]

        # ---------- 4) Returns ----------
        coadd_pair = (w_common, coadd)
        if return_coadd and return_meta:
            return (r1, r2, coadd_pair, failed_indices, ew_meta)
        elif return_coadd:
            return (r1, r2, coadd_pair)
        elif return_meta:
            return (r1, r2, failed_indices)
        else:
            return (r1, r2)
