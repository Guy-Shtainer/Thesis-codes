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
        window_size = 20
        sigma_thresh = 1

        # Clean Mask
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

    # ------------------------------------------------------------------ #
    # public: two-pass RV + coadd                                         #
    # ------------------------------------------------------------------ #
    def double_ccf(self, obs_list, tpl_wave, tpl_flux, return_coadd=False):
        """
        Parameters
        ----------
        obs_list : list[(wave, flux), …]  – multiple observations
        tpl_wave, tpl_flux : arrays       – initial template

        Returns
        -------
        round1, round2 : lists of (RV, σ)
        (optionally) (wave_coadd, flux_coadd)
        """
        # --- first pass -------------------------------------------------
        # --- first pass: RVs + per‑spectrum S/N from the CCF window ----------
        r1, S2N = [], []
        print(
            f"Calculating for first time to get shifts and create the coadded template"
        )
        for w, f in obs_list:
            rv, sig = self.compute_RV(w, f, tpl_wave, tpl_flux)
            r1.append((rv, sig))

            # ---- S/N from the same wavelength slice(s) ----------------------
            noises = []
            for rng in self.S2Nrange:
                m = (w > rng[0]) & (w < rng[1])  # boolean mask
                if m.any():
                    noises.append(np.std(f[m]))
            # fallback: if none of the ranges overlap this spectrum
            noise = np.mean(noises)
            S2N.append(1.0 / noise if noise > 0 else 1.0)

        S2N = np.asarray(S2N)

        # --- co-add in rest frame --------------------------------------
        w_common = obs_list[0][0] * 10 if self.nm else obs_list[0][0]
        coadd = np.zeros_like(w_common)
        weights = (S2N**2) / np.sum(S2N**2)
        # print(f'weights = {weights}')
        # Iterate over observation list, radial velocities, and weights
        print(f"Now calculating coadded template")
        for (w, f), (rv, _), wgt in zip(obs_list, r1, weights):
            if rv is None:
                print(
                    f"Check out star {self.star_name}, epoch {self.epoch} line {self.line_tag}"
                )
                continue

            # print(f'rv is {rv}, wgt is {wgt} for epoch {self.epoch}')
            # print(f'f is {f}, w is {w}')
            # Perform wavelength shift and interpolation
            try:
                shifted = interp1d(
                    w * (1 - rv / clight),
                    f,
                    kind=self.intr_kind,
                    bounds_error=False,
                    fill_value=1.0,
                )(w_common)
            except:
                print(f"what failed is:")
                print(f"rv: {rv}")
                print(f"w: {w * (1 - rv / clight)}")
                print(f"f: {shifted}")
            coadd += wgt * shifted

            # --- second pass ----------------------------------------------
        # print(f'coadd is {coadd}')
        print(f"Finally calculating CCF using coadded template")
        r2 = [self.compute_RV(w, f, w_common, coadd) for w, f in obs_list]

        return (r1, r2, (w_common, coadd)) if return_coadd else (r1, r2)
