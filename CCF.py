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

clight = 2.9979e5  # km s⁻¹


class CCFclass:
    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(self,
                 intr_kind: str = "cubic",
                 Fit_Range_in_fraction: float = 0.95,
                 CrossCorRangeA=((4000., 4500.),),
                 CrossVeloMin: float = -400.,
                 CrossVeloMax: float =  400.,
                 PlotFirst: bool = False,
                 PlotAll:   bool = False,
                 star_name: str | None = None,
                 epoch: str | int | None = None,
                 line_tag: str = ""):
        # ---- original parameters --------------------------------------
        self.intr_kind    = intr_kind
        self.Fit_Range_in_fraction  = Fit_Range_in_fraction
        self.CrossCorRangeA  = np.asarray(CrossCorRangeA, float)
        self.S2Nrange = [[445.0, 445.5]]
        self.CrossVeloMin    = CrossVeloMin
        self.CrossVeloMax    = CrossVeloMax
        self.PlotFirst   = PlotFirst
        self.PlotAll = PlotAll
        self._first_done = not PlotFirst

        # ---- new contextual metadata ----------------------------------
        self.star_name = star_name or "unknown‑star"
        self.epoch     = epoch
        self.line_tag  = line_tag


    # ------------------------------------------------------------------ #
    # static helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _CCF(f1, f2, N):
        """Normalised dot-product."""
        return np.sum(f1 * f2) / np.std(f1) / np.std(f2) / N


    # remove duplicates **after** log-transform, keep first occurrence
    @staticmethod
    def _prep_spectrum(wave, flux):
        logw = np.log(wave)
        order = np.argsort(logw)
        logw, flux = logw[order], flux[order]
        _, keep = np.unique(logw, return_index=True)
        return logw[keep], flux[keep]

    # ------------------------------------------------------------------ #
    # internal core: parabola-fit cross-correlation                       #
    # ------------------------------------------------------------------ #
    def _crosscorreal(self, Observation, Mask, CrossCorInds, sRange, N, veloRange, wavegridlog):
        CCFarr = np.array([self._CCF(np.copy(Observation),
                               (np.roll(Mask, s))[CrossCorInds],N) for s in sRange])

        IndMax  = np.argmax(CCFarr)
        vmax = veloRange[IndMax]
        CCFMAX1  = CCFarr[IndMax]

        # edges at fitfac·CCFMAX1
        LeftEdgeArr  = np.abs(self.Fit_Range_in_fraction*CCFMAX1 - CCFarr[:IndMax])
        RightEdgeArr = np.abs(self.Fit_Range_in_fraction*CCFMAX1 - CCFarr[IndMax+1:])
        if len(LeftEdgeArr) == 0 or len(RightEdgeArr) == 0:
            print("Can't find local maximum in CCF\n")
            return np.array([np.nan, np.nan])
        IndFit1 = np.argmin(LeftEdgeArr)
        IndFit2 = np.argmin(RightEdgeArr) + IndMax + 1
        a, b, c = np.polyfit(veloRange[IndFit1:IndFit2 + 1],
                             CCFarr[IndFit1:IndFit2 + 1], 2)
        vmax   = -b / (2*a)
        CCFAtMax = min(1-1E-20, c - b**2/4./a)
        FineVeloGrid = np.arange(veloRange[IndFit1], veloRange[IndFit2], .1)
        parable = (a * FineVeloGrid ** 2 + b * FineVeloGrid + c)
        sigma = np.sqrt(-1. / (N * 2 * a * CCFAtMax / (1 - CCFAtMax ** 2)))

        if self.PlotFirst is True or self.PlotAll is True:
            # Radial Velocity and Error
            RV = vmax  # Example Radial Velocity in km/s (replace with computed value)
            RV_error = sigma  # Example error in km/s (replace with computed value)
            star_name = self.star_name
            epoch = self.epoch

            # CCF Plot
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(veloRange, CCFarr, color='C0', label='CCF')
            ax1.plot(FineVeloGrid, parable, color='C1', linewidth=1.5, label='Fit (Parabola)')
            ax1.set_title(f"CCF for {star_name} - Epoch: {epoch}", fontsize=14, weight='bold')
            ax1.set_xlabel('Radial Velocity [km/s]', fontsize=12)
            ax1.set_ylabel('Normalized CCF', fontsize=12)
            ax1.axvline(RV, color='red', linestyle='--', alpha=0.7, label=f"RV = {RV:.4f} ± {RV_error:.4f} km/s")
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # Annotating RV and error directly on the plot
            ax1.annotate(f"RV = {RV:.2f} ± {RV_error:.2f} km/s",
                         xy=(RV, np.max(CCFarr) * 0.9),
                         xytext=(RV + 2, np.max(CCFarr) * 0.8),
                         arrowprops=dict(facecolor='black', arrowstyle="->", lw=0.8),
                         fontsize=10)

            # Save the plot with a meaningful name
            cutname = f"{star_name}_{epoch}"
            # fig1.savefig(f"CCFPlots/CCF_parabola_{cutname}.pdf")
            plt.show()

            # Spectrum and Template Plot
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(wavegridlog[CrossCorInds], Observation, color='k', label='Observation', alpha=0.8)
            ax2.plot(wavegridlog, (Mask - np.mean(Mask)), color='orchid', label='Template (Unshifted)', alpha=0.9)
            ax2.plot((wavegridlog * (1 + RV / clight)), (Mask - np.mean(Mask)), color='turquoise',
                     label='Template (Shifted by RV)', alpha=0.9)

            ax2.set_title(f"Spectra for {star_name} - Epoch: {epoch}", fontsize=14, weight='bold')
            ax2.set_xlabel(r'Wavelength [$\AA$]', fontsize=12)
            ax2.set_ylabel("Normalized Flux", fontsize=12)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.5)

            # Emphasizing RV shift on wavelength grid
            # ax2.annotate(f"Wavelength Shift by RV = {RV:.2f} km/s",
            #              xy=(wavegridlog[-100], np.min(Mask) * 1.1),
            #              xytext=(wavegridlog[-300], np.min(Mask) * 1.2),
            #              arrowprops=dict(facecolor='blue', arrowstyle="->", lw=0.8),
            #              fontsize=10, color='blue')

            # Save the plot with a meaningful name
            # fig2.savefig(f"SpectraPlots/Spectra_{cutname}.pdf")
            plt.show()

            # Turn off PlotFirst after plotting
            self.PlotFirst = False

        if CCFAtMax > 1:
            print("Failed to cross-correlate: template probably sucks!")
            print("Check cross-correlation function + parable fit.")
            return np.nan, np.nan
    
        CFFdvdvAtMax = 2*a
        return np.array([vmax, np.sqrt(-1./(N * CFFdvdvAtMax *
                                            CCFAtMax / (1 - CCFAtMax**2)))])

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
        LambdaRangeUser = self.CrossCorRangeA * np.array([1 - 1.1 * self.CrossVeloMax / clight,
                                       1 - 1.1 * self.CrossVeloMin / clight])

        LamRangeB = LambdaRangeUser[0, 0]
        LamRangeR = LambdaRangeUser[-1, 1]

        Dlam = obs_wave[1] - obs_wave[0]
        Resolution = obs_wave[1] / Dlam  # instructor: Resolution (λ/Δλ)
        vbin = clight / Resolution  # identical formula

        Nwaves = int(np.log(LamRangeR / LamRangeB) / np.log(1. + vbin / clight))
        
        wavegridlog = LamRangeB * (1. + vbin / clight) ** np.arange(Nwaves)

        IntIs = np.array([np.argmin(np.abs(wavegridlog - self.CrossCorRangeA[i][0]))
                          for i in np.arange(len(self.CrossCorRangeA))])
        IntFs = np.array([np.argmin(np.abs(wavegridlog - self.CrossCorRangeA[i][1]))
                          for i in np.arange(len(self.CrossCorRangeA))])

        Ns = IntFs - IntIs # number of points in range. if there are several ranges at once it accounts for them
        N = np.sum(Ns) # relevant in case i pass several emission lines ranges


        CrossCorInds = np.concatenate(([np.arange(IntIs[i], IntFs[i])
                                        for i in np.arange(len(IntFs))])) # Find the indices which are the emission line
        sRange = np.arange(int(self.CrossVeloMin / vbin), int(self.CrossVeloMax / vbin) + 1, 1)
        veloRange = vbin * sRange

        MaskAll = np.array([tpl_wave, tpl_flux]).T
        Mask = interp1d(MaskAll[:, 0], np.nan_to_num(MaskAll[:, 1]), bounds_error=False,
                        fill_value=1., kind=self.intr_kind)(wavegridlog)

        flux = interp1d(obs_wave, np.nan_to_num(
            obs_flux), bounds_error=False, fill_value=1.,
                          kind='cubic')(wavegridlog[CrossCorInds])
        CCFeval = self._crosscorreal(np.copy(flux - np.mean(flux)),
                               np.copy(Mask - np.mean(Mask)),CrossCorInds,sRange,N,veloRange,wavegridlog)
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
            noise = np.mean(noises) if noises else np.std(f)
            S2N.append(1.0 / noise if noise > 0 else 1.0)

        S2N = np.asarray(S2N)

        # --- co-add in rest frame --------------------------------------
        w_common = obs_list[0][0]
        coadd = np.zeros_like(w_common)
        weights = (S2N**2) / np.sum(S2N**2)
        for (w, f), (rv, _), wgt in zip(obs_list, r1, weights):
            shifted = interp1d(w*(1-rv/clight), f, kind=self.intr_kind,
                               bounds_error=False, fill_value=1.0)(w_common)
            coadd += wgt * shifted

        # --- second pass ----------------------------------------------
        r2 = [self.compute_RV(w, f, w_common, coadd) for w, f in obs_list]

        return (r1, r2, (w_common, coadd)) if return_coadd else (r1, r2)
