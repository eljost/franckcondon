import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from numpy.polynomial.hermite import Hermite
from scipy.integrate import simpson
from scipy.special import factorial


np.set_printoptions(suppress=True, precision=4)


def get_ho(k=1, m=1, nu=2, q0=0, offset=0.0, v_max=5):
    # Hermite polynomials
    hermites = list()
    vs = v_max + 1
    for v in range(vs):
        coef = np.zeros(vs)
        coef[v] = 1
        hermites.append(Hermite(coef))

    # w = (k / m) ** 0.5

    def ho_pot(q):
        """Harmonic oscillator potential."""
        return 1 / 2 * (q - q0) ** 2 + offset

    def ho_pot_inv(pot):
        return np.sqrt(2 * (pot - offset)) + q0

    def ho_levels(v):
        return nu * (v + 1 / 2) + offset

    def ho_wf(q, v):
        N = ((1 / np.pi) ** 0.5 / (2 ** v * factorial(v))) ** 0.5
        H = hermites[v](q)
        return N * H * np.exp(-(q ** 2) / 2)

    return ho_pot, ho_pot_inv, ho_levels, ho_wf


class HarmonicOscillator:
    def __init__(self, k=1, m=1, nu=2, q0=0.0, offset=0.0, v_max=5):
        self.k = k
        self.m = m
        self.nu = nu
        self.q0 = q0
        self.offset = offset
        self.v_max = v_max

        # Hermite polynomials
        self.hermites = list()
        vs = v_max + 1
        for v in range(vs):
            coef = np.zeros(vs)
            coef[v] = 1
            self.hermites.append(Hermite(coef))

    def pot(self, q):
        """Harmonic oscillator potential."""
        return 1 / 2 * (q - self.q0) ** 2 + self.offset

    def pot_inv(self, pot):
        return np.sqrt(2 * (pot - self.offset)) + self.q0

    def level(self, v):
        return self.nu * (v + 1 / 2) + self.offset

    def wf(self, q, v):
        N = ((1 / np.pi) ** 0.5 / (2 ** v * factorial(v))) ** 0.5
        H = self.hermites[v](q - self.q0)
        return N * H * np.exp(-((q - self.q0) ** 2) / 2)


def run():
    v_max = 5
    trunc_pot = 15
    HO_init = HarmonicOscillator()

    thresh = 7
    qs = np.linspace(-thresh, thresh, 100)
    wf_gs_init = HO_init.wf(qs, 0)  # Ground state WF

    vs = np.arange(v_max + 1)

    fig, (ax, ax_ovlp) = plt.subplots(
        nrows=2, gridspec_kw={"height_ratios": [3, 1]}, figsize=(5, 8)
    )

    def update_plot(q0):
        # Clear axes
        for ax_ in (ax, ax_ovlp):
            ax_.cla()

        HO_fin = HarmonicOscillator(offset=15, q0=q0)
        HOs = (HO_init, HO_fin)

        # Plot potentials
        for HO in HOs:
            ys = HO.pot(qs)
            # Truncate potentials
            ys[ys >= HO.offset + trunc_pot] = np.nan
            ax.plot(qs, ys)

        # Calculate and plot wavefunctions
        for v in vs:
            for HO in HOs:
                lvl = HO.level(v)
                ax.axhline(lvl, color="black", ls="--", lw=0.5)
                wf = HO.wf(qs, v)
                ax.plot(qs, wf + lvl)

        # Calculate overlaps between GS WF and ES wavefunctions
        overlaps = np.zeros_like(vs, dtype=float)
        for v in vs:
            wf = HO_fin.wf(qs, v)
            ovlp = simpson(wf_gs_init * wf, qs) ** 2
            overlaps[v] = ovlp
        ax_ovlp.stem(overlaps)
        # Label
        for v in vs:
            xy = (v, min(45, overlaps[v] + 5))
            ax_ovlp.annotate(f"0-{v}", xy, ha="center")
        ax_ovlp.set_ylim(0, 1.0)
        ax_ovlp.set_title("Wavefunction overlaps")
        ax_ovlp.set_ylabel("Overlap")

        sqrt2 = 2 ** 0.5
        ax.axvline(-sqrt2)
        ax.axvline(sqrt2)
        ax.set_ylim(0, 30)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\Delta E$")
        ax.set_title("Franck-Condon principle")

    ax_q0 = plt.axes((0.2, 0.025, 0.5, 0.025))
    slider_q0 = Slider(ax_q0, r"$\Delta q$", -3, 3, valinit=0, valfmt="%f")
    slider_q0.on_changed(update_plot)
    update_plot(0.0)

    plt.show()


if __name__ == "__main__":
    run()
