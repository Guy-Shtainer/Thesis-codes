import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

clight = 2.9979E5  # [km/s]
eps = 1E-10

class CCFclass:
    def __init__(self, intr_kind='cubic', Fit_Range_in_fraction=0.95, CrossCorRangeA=[[4000., 5000.]],
                 CrossVeloMin=-400., CrossVeloMax=400., PlotFirst=False, PlotAll=False):
        # Determines type of interpolation when reading spectra: should not be changed
        self.intr_kind = intr_kind

        # Determines whether diagnostic plots are shown (first / all)
        self.PlotFirst = PlotFirst
        self.PlotAll = PlotAll

        # Determines the range in which a parable is fit to the CCF function.
        self.Fit_Range_in_fraction = Fit_Range_in_fraction

        self.CrossCorRangeA = np.array(CrossCorRangeA)

        # Minimum and maximum RVs to be searched
        self.CrossVeloMin = CrossVeloMin
        self.CrossVeloMax = CrossVeloMax

    # Returns CCF of two functions f1(x) and f2(x) [f1 = observation, f2 = mask]
    def CCF(self, f1, f2, N):
        f1 = f1
        f2 = f2
        return np.sum(f1 * f2) / np.std(f1) / np.std(f2) / N

    # Returns RV and error following Zucker+ 2003
    def crosscorreal(self, Observation, Mask, CrossCorInds, sRange, N, veloRange, wavegridlog):
        CCFarr = np.array([self.CCF(Observation[CrossCorInds], np.roll(Mask, s)[CrossCorInds], N) for s in sRange])
        IndMax = np.argmax(CCFarr)
        vmax = veloRange[IndMax]
        CCFMAX1 = CCFarr[IndMax]

        LeftEdgeArr = np.abs(self.Fit_Range_in_fraction * CCFMAX1 - CCFarr[:IndMax])
        RightEdgeArr = np.abs(self.Fit_Range_in_fraction * CCFMAX1 - CCFarr[IndMax+1:])

        if len(LeftEdgeArr) == 0 or len(RightEdgeArr) == 0:
            print("Can't find local maximum in CCF\n")
            return np.array([np.nan, np.nan])

        IndFit1 = np.argmin(LeftEdgeArr)
        IndFit2 = np.argmin(RightEdgeArr) + IndMax + 1

        a, b, c = np.polyfit(veloRange[IndFit1:IndFit2+1], CCFarr[IndFit1:IndFit2+1], 2)
        vmax = -b / (2 * a)
        CCFAtMax = min(1 - 1E-20, c - b**2 / (4 * a))

        if self.PlotFirst or self.PlotAll:
            # plot the ccf
            fig1, ax1 = plt.subplots()
            ax1.plot(veloRange, CCFarr, color='C0')
            # ax1.scatter(veloRange, CCFarr, color='C0')
            FineVeloGrid = np.arange(veloRange[IndFit1], veloRange[IndFit2], 0.1)
            parable = a * FineVeloGrid**2 + b * FineVeloGrid + c
            ax1.plot(FineVeloGrid, parable, color='C1', linewidth=1.5)
            ax1.set_xlabel('Radial velocity [km/s]')
            ax1.set_ylabel('Normalized CCF')
            plt.show()

            # plot the spectrum and the template
            fig2, ax2 = plt.subplots()
            ax2.plot(np.exp(wavegridlog[CrossCorInds]), Observation[CrossCorInds], color='k',
                     label='observation', alpha=0.8)
            ax2.plot(np.exp(wavegridlog), Mask - np.mean(Mask), color='orchid',
                     label='template, unshifted', alpha=0.9)
            shifted_wavegridlog = wavegridlog + np.log(1 + vmax / clight)
            ax2.plot(np.exp(shifted_wavegridlog), Mask - np.mean(Mask),
                     color='turquoise', label='template, shifted', alpha=0.9)
            ax2.set_xlabel(r'Wavelength [$\AA$]')
            ax2.set_ylabel('Normalized flux')
            ax2.legend(loc='best')
            plt.show()
            self.PlotFirst = False

        if CCFAtMax > 1:
            print("Failed to cross-correlate: template probably not suitable!")
            print("Check cross-correlation function and parabolic fit.")
            return np.array([np.nan, np.nan])

        CFFdvdvAtMax = 2 * a
        RV_error = np.sqrt(-1. / (N * CFFdvdvAtMax * CCFAtMax / (1 - CCFAtMax**2)))

        return np.array([vmax, RV_error])

    def compute_RV(self, observation_wave, observation_flux, template_wave, template_flux):
        # Convert wavelengths to logarithmic scale
        log_obs_wave = np.log(observation_wave)
        log_temp_wave = np.log(template_wave)

        # Adjust CrossCorRangeA for velocities
        LambdaRangeUser = self.CrossCorRangeA * np.array([1. - 1.1*self.CrossVeloMax/clight, 1 - 1.1*self.CrossVeloMin/clight])

        LamRangeB = LambdaRangeUser[0, 0]
        LamRangeR = LambdaRangeUser[-1, 1]

        log_LamRangeB = np.log(LamRangeB)
        log_LamRangeR = np.log(LamRangeR)

        # Determine overlapping wavelength range
        log_wave_min = max(log_LamRangeB, np.min(log_obs_wave), np.min(log_temp_wave))
        log_wave_max = min(log_LamRangeR, np.max(log_obs_wave), np.max(log_temp_wave))

        # Create logarithmic wavelength grid
        num_points = 10000  # Adjust as needed
        log_wavegrid = np.linspace(log_wave_min, log_wave_max, num_points)

        delta_log_lambda = log_wavegrid[1] - log_wavegrid[0]
        vbin = delta_log_lambda * clight

        # Interpolate observation and template onto log_wavegrid
        interp_obs_flux = interp1d(log_obs_wave, observation_flux, bounds_error=False, fill_value=1.0, kind=self.intr_kind)(log_wavegrid)
        interp_temp_flux = interp1d(log_temp_wave, template_flux, bounds_error=False, fill_value=1.0, kind=self.intr_kind)(log_wavegrid)

        # print(f'interp_obs_flux len is: {len(interp_obs_flux)} and interp_temp_flux len is {len(interp_temp_flux)}')
        # Subtract mean
        obs_flux_norm = interp_obs_flux - np.mean(interp_obs_flux)
        temp_flux_norm = interp_temp_flux - np.mean(interp_temp_flux)
        # obs_flux_norm = interp_obs_flux
        # temp_flux_norm = interp_temp_flux

        # Define CrossCorInds
        CrossCorInds_list = []
        Ns = []
        for i in range(len(self.CrossCorRangeA)):
            CrossCorRange = self.CrossCorRangeA[i]
            log_CrossCorRange = np.log(CrossCorRange)
            IntI = np.argmin(np.abs(log_wavegrid - log_CrossCorRange[0]))
            IntF = np.argmin(np.abs(log_wavegrid - log_CrossCorRange[1]))
            Ns.append(IntF - IntI)
            CrossCorInds_list.append(np.arange(IntI, IntF))
        CrossCorInds = np.concatenate(CrossCorInds_list)
        N = np.sum(Ns)

        # Define sRange and veloRange
        sRange = np.arange(int(self.CrossVeloMin/vbin), int(self.CrossVeloMax/vbin)+1, 1)
        veloRange = vbin * sRange

        # print(f'obs_flux_norm len is: {len(obs_flux_norm)} and temp_flux_norm len is {len(temp_flux_norm)}')

        # Perform cross-correlation
        CCFeval = self.crosscorreal(obs_flux_norm, temp_flux_norm, CrossCorInds, sRange, N, veloRange, log_wavegrid)

        RV = CCFeval[0]
        RV_error = CCFeval[1]

        return RV, RV_error
