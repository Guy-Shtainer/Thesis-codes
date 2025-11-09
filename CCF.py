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
        self, Observation, Mask, CrossCorInds, sRange, N, veloRange, wavegridlog
    ):
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

        # if len(LeftEdgeArr) == 0 or len(RightEdgeArr) == 0:
        #     print("Can't find local maximum in CCF\n")
        #     fig1, ax1 = plt.subplots()
        #     ax1.plot(veloRange, CCFarr, color='C0', label=f'obs {self.star_name}')
        #     plt.show()
        #     return np.array([None, None, None, None])

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
                f"{line_tag}  ({line_rng[0]:.0f}–{line_rng[1]:.0f} {wave_units})"
                if line_tag
                else f"{line_rng[0]:.0f}–{line_rng[1]:.0f} {wave_units}"
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

            # -------- 2.  Spectrum vs template ------------------------------
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(
                wavegridlog[CrossCorInds],
                Observation,
                label="Observation",
                color="k",
                alpha=0.8,
            )
            ax2.plot(
                wavegridlog,
                Mask - np.mean(Mask),
                label="Template (unshifted)",
                color="orchid",
                alpha=0.9,
            )
            ax2.plot(
                wavegridlog * (1 + RV / clight),
                Mask - np.mean(Mask),
                label="Template (shifted)",
                color="turquoise",
                alpha=0.9,
            )

            # ax2.set_title(f"Spectra  |  {star_name}  |  Epoch {epoch_str}  |  {line_txt}",
            #               fontsize=14, weight='bold')
            # ax2.set_xlabel(r'Wavelength [nm]')
            ax2.set_title(
                f"Spectra  |  {star_name}  |  {epoch_txt}{spec_txt}  |  {line_txt}",
                fontsize=14,
                weight="bold",
            )
            ax2.set_xlabel(rf"Wavelength [{wave_units}]")  # nm or Å

            ax2.set_ylabel("Normalized Flux")
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
        SNR = 1/std. If for_ew=True, estimate the noise in near-line windows:
        a 5 Å (or 0.5 nm when self.nm=True) window immediately LEFT of each
        emission-line interval in self.CrossCorRangeA (i.e., [λ_left-Δ, λ_left]).
        Otherwise fall back to self.S2Nrange (old behavior).
        If no points are available in the chosen windows, fall back to the
        continuum-outside-lines std, then to global std.
        """
        # --- choose where to measure the noise ---
        if for_ew:
            width = 0.5 if self.nm else 5.0  # 0.5 nm ≡ 5 Å
            # one window per line range: [left-edge - width, left-edge]
            search_ranges = [[r[0] - width, r[0]] for r in np.atleast_2d(self.CrossCorRangeA)]
        else:
            search_ranges = self.S2Nrange

        # --- measure noise as std in each available window; take median ---
        noises = []
        for lo, hi in np.atleast_2d(search_ranges):
            if hi <= lo:
                lo, hi = hi, lo
            m = (w > lo) & (w < hi)
            if np.any(m):
                noises.append(float(np.std(f[m])))

        if len(noises) == 0:
            # exclude the line windows to approximate continuum
            line_mask = np.zeros_like(w, dtype=bool)
            for r in np.atleast_2d(self.CrossCorRangeA):
                line_mask |= (w > r[0]) & (w < r[1])
            cont = ~line_mask
            if np.any(cont):
                noise = float(np.std(f[cont]))
            else:
                noise = float(np.std(f))
        else:
            noise = float(np.median(noises))

        snr = (1.0 / noise) if (noise > 0 and np.isfinite(noise)) else np.inf
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

        sigma = (np.sqrt(sum_dlam2N) / snr) if (sum_dlam2N > 0 and np.isfinite(snr)) else np.nan
        return total_EW, sigma, snr


    def _ew_gate(self, w, f, ksig=10.0):
        """
        Returns a dict with EW gate results for one epoch.
        """
        EW, sigEW, SNR = self._ew_sigma_rule_of_thumb(w, f)
        detected = bool(np.isfinite(EW) and np.isfinite(sigEW) and (EW - ksig * sigEW) > 0.0)
        return {"EW": EW, "sigma_EW": sigEW, "SNR": SNR, "detected": detected}

    # ------------------------------------------------------------------ #
    # public: single-spectrum RV                                          #
    # ------------------------------------------------------------------ #
    def compute_RV(self, obs_wave, obs_flux, tpl_wave, tpl_flux):
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
        flux = interp1d(
            obs_wave,
            np.nan_to_num(obs_flux),
            bounds_error=False,
            fill_value=1.0,
            kind="cubic",
        )(wavegridlog[CrossCorInds])
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

        CCFeval = self._crosscorreal(
            np.copy(flux - np.mean(flux)),
            np.copy(Mask - np.mean(Mask)),
            CrossCorInds,
            sRange,
            N,
            veloRange,
            wavegridlog,
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
            sigma_clip=3.0,
            random_state=None,
            plot=True,
            ax=None,
            add_noise=True,                 # << NEW
            noise_source='residual',        # 'residual' | 'leftwin' | 'local' | 'global'
            noise_floor=1e-6,               # << NEW
        ):
        """
        Multi-stage cleaner:
        - Split n_iter into n_stages chunks.
        - After each chunk, average the random deg-degree poly fits to a model,
            σ-clip residuals within the focus range(s), replace outliers with the model,
            and proceed to the next chunk using the UPDATED flux.
        - Final outputs are the cumulatively cleaned flux, the FINAL stage model,
            and a union mask of all replaced points across stages.

        If focus_range is None, applies to all ranges in self.CrossCorRangeA,
        sequentially on the same flux. Plot shows only the emission-line range(s)
        and highlights all replaced points in red.
        """

        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)

        # Which ranges to clean?
        if focus_range is None:
            ranges = [tuple(r) for r in np.atleast_2d(self.CrossCorRangeA)]
        else:
            lo, hi = focus_range
            if hi <= lo:
                raise ValueError("focus_range must satisfy lo < hi.")
            ranges = [(lo, hi)]

        # Accumulators for a single combined plot across all ranges
        combined_model = np.full_like(flux, np.nan, dtype=float)  # final stage model per pixel
        combined_repl  = np.zeros_like(flux, dtype=bool)          # union of all replacements
        cleaned_flux   = flux.copy()

        rng = np.random.default_rng(random_state)

        def _clean_one_range_staged(w, f_in, lo, hi):
            """Run staged cleaning on a single [lo,hi] range, return updated flux and masks."""
            in_mask = (w >= lo) & (w <= hi)
            if np.count_nonzero(in_mask) < (deg + 1):
                # Not enough points to fit the polynomial – skip safely
                return f_in.copy(), np.full_like(f_in, np.nan, dtype=float), np.zeros_like(f_in, dtype=bool)

            # local views for convenience
            xi = w[in_mask]
            # will mutate yi as we replace outliers between stages
            yi = f_in[in_mask].copy()
            M  = xi.size

            # sample size per iteration
            k = max(deg + 1, int(np.ceil(sample_frac * M)))
            k = min(k, M)

            # normalize x for numerical stability
            xc  = xi.mean()
            xs  = xi.std() if xi.std() > 0 else 1.0
            xi0 = (xi - xc) / xs

            # staging setup
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

                # --- robust residual sigma & clip ---
                res   = yi - model_i
                med   = np.median(res)
                mad   = np.median(np.abs(res - med))
                sigma = 1.4826 * mad if mad > 0 else (np.std(res) if np.std(res) > 0 else 0.0)

                outliers = np.abs(res - med) > (sigma_clip * max(sigma, noise_floor))

                # --- choose noise scale ---------------------------------------
                if noise_source == 'residual':
                    noise_sigma = sigma
                elif noise_source == 'leftwin':
                    # use the same near-line continuum window as your EW SNR
                    # (left side of the line range)
                    _, noise_sigma = self._estimate_snr_robust(w, f_in, for_ew=True)
                elif noise_source == 'local':
                    # plain std of current segment (robustness not guaranteed)
                    yseg = f_in[in_mask]
                    noise_sigma = float(np.std(yseg)) if yseg.size else 0.0
                elif noise_source == 'global':
                    noise_sigma = float(np.std(f_in))
                else:
                    noise_sigma = sigma

                noise_sigma = max(float(noise_sigma), noise_floor)

                # --- replace outliers (optionally with noise) ------------------
                if np.any(outliers):
                    if add_noise:
                        yi[outliers] = model_i[outliers] + rng.normal(
                            0.0, noise_sigma, size=int(outliers.sum())
                        )
                    else:
                        yi[outliers] = model_i[outliers]

                # Accumulate replacements etc. (unchanged)
                tmp_mask = np.zeros_like(f_in, dtype=bool)
                tmp_mask[in_mask] = outliers
                replaced_union |= tmp_mask

                if stage == n_stages_eff - 1:
                    model_full_final[in_mask] = model_i

                if not np.any(outliers) and stage < n_stages_eff - 1:
                    model_full_final[in_mask] = model_i
                    break

            # write back cleaned segment
            f_out = f_in.copy()
            f_out[in_mask] = yi

            return f_out, model_full_final, replaced_union

        # Clean all ranges sequentially on the evolving cleaned_flux
        for (lo, hi) in ranges:
            cleaned_flux, model_full, replaced_full = _clean_one_range_staged(wave, cleaned_flux, lo, hi)
            # accumulate model & replacements
            msel = ~np.isnan(model_full)
            combined_model[msel] = model_full[msel]
            combined_repl |= replaced_full

        # --- plotting (once) ---
        if plot:  # << only this flag matters for showing the plot
            if ax is None:
                fig, ax = plt.subplots(figsize=(9, 5))

            units = "nm" if self.nm else "Å"

            # Determine plotted ranges
            if focus_range is None:
                _ranges = [tuple(r) for r in np.atleast_2d(self.CrossCorRangeA)]
            else:
                lo, hi = focus_range
                _ranges = [(min(lo, hi), max(lo, hi))]

            # Union mask for the plotted region(s)
            mask_total = np.zeros_like(wave, dtype=bool)
            for lo, hi in _ranges:
                mask_total |= (wave >= lo) & (wave <= hi)

            # Mask outside-range points for a clean view
            f_plot = flux.copy()
            f_plot[~mask_total] = np.nan
            c_plot = cleaned_flux.copy()
            c_plot[~mask_total] = np.nan
            m_plot = combined_model.copy()  # NaN outside cleaned regions already

            # Plot original, final-stage model, cleaned, and replaced points (union)
            ax.plot(wave, f_plot, label="original", alpha=0.8)
            if np.any(~np.isnan(m_plot)):
                ax.plot(wave, m_plot, ls="--", label=f"avg deg-{deg} model (final stage)")
            ax.plot(wave, c_plot, alpha=0.9, label="cleaned")

            repl_in = combined_repl & mask_total
            if np.any(repl_in):
                ax.scatter(wave[repl_in], flux[repl_in], s=18, color="red", label="replaced")

            # Tight x-limits around union of ranges (+5% padding)
            lo_all = min(r[0] for r in _ranges)
            hi_all = max(r[1] for r in _ranges)
            pad = 0.05 * (hi_all - lo_all) if hi_all > lo_all else (0.5 if self.nm else 5.0)
            ax.set_xlim(lo_all - pad, hi_all + pad)

            ax.set_xlabel(f"Wavelength [{units}]")
            ax.set_ylabel("Normalized flux")
            title_tag = self.line_tag or "emission line"
            ax.set_title(f"Cleaning {self.star_name} | {title_tag}")
            ax.legend()
            ax.grid(ls="--", alpha=0.35)
            plt.tight_layout()

            if self.savePlot:
                clean_star = re.sub(r"[^A-Za-z0-9_-]", "_", (self.star_name or "unknown"))
                epoch_str  = "NA" if self.epoch is None else str(self.epoch)
                spec_str   = "" if self.spectrum is None else (f"_S{self.spectrum}")
                out_dir = Path("../output") / clean_star / "CLEAN" / (self.run_ts or "")
                out_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_dir / f"{clean_star}_MJD{epoch_str}{spec_str}_{title_tag}_CLEAN.png", dpi=150)

            plt.show()
            try:
                plt.close(fig)
            except NameError:
                pass

        return cleaned_flux, combined_model, combined_repl




    # ------------------------------------------------------------------ #
    # public: two-pass RV + coadd                                         #
    # ------------------------------------------------------------------ #
    def double_ccf(self, obs_list, tpl_wave, tpl_flux, return_coadd=False, return_meta=False):
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
        cleaned_obs_list = []
        for i, (ep, w, f) in enumerate(obs_list):
            # Plot first, or all, according to class flags; otherwise suppress
            do_plot = self.PlotAll or (self.PlotFirst and (i == 0)) or self.savePlot
            f_clean, _, _ = self.clean_line_with_iterative_poly(
                wave=w,
                flux=f,
                focus_range=None,     # << uses self.CrossCorRangeA internally
                n_iter=3000,
                n_stages=30,
                sample_frac=0.7,
                deg=18,
                sigma_clip=4.0,
                random_state=42,
                plot=True
            )
            print(f'cleaned epoch {ep} of line {self.line_tag}')
            cleaned_obs_list.append((ep, w, f_clean))

        obs_list = cleaned_obs_list
        # ---------- 0) EW gate per epoch ----------
        ew_meta = []
        include_mask = []
        failed_indices = []
        for ep, w, f in obs_list:
            info = self._ew_gate(w, f, ksig=5.0)
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
                rv, sig = self.compute_RV(w, f, tpl_wave, tpl_flux)
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
            ep,w, f = obs_list[i]
            r2_full[i] = self.compute_RV(w, f, w_common, coadd)

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
