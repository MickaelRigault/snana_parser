
import os
from glob import glob
import pandas

from snana_parser import fitres, wfit, m0dif

class BBCOutput():

    def __init__(self, fitres_, m0dif_, wfit_):
        """ """
        self.set_fitres(fitres_)
        self.set_m0dif(m0dif_)
        self.set_wfit(wfit_)

    # ================= #
    #      I/O          #
    # ================= #
    @classmethod
    def from_directory(cls, outputdir, muopt=0, fitopt=0, use_dask=True, client=None):
        """ """
        base = os.path.join(outputdir,"OUTPUT_BBCFIT-*",
                            f"FITOPT{muopt:03d}_MUOPT{muopt:03d}")
        fitres_files = glob(base+".FITRES")
        m0dir_files = glob(base+".M0DIF")
        wfit_files = glob(base.replace("FITOPT","wfit_FITOPT")+".YAML")

        propload = dict(use_dask=use_dask, client=client)
        return cls(fitres_=fitres.FITRES.from_fitres_files(fitres_files,
                                                               **propload),
                   m0dif_=m0dif.M0DIF.from_m0dif_files(m0dir_files, **propload),
                   wfit_ = wfit.read_wfit_files(wfit_files, **propload))
                       
        
    # ================= #
    #     Methods       #
    # ================= #
    # -------- #
    #  GETTER  #
    # -------- #
    def set_fitres(self, fitres):
        """ """
        self._fitres = fitres
        self._fittedparams = None
        
    def set_m0dif(self, m0dif):
        """ """
        self._m0dif = m0dif

    def set_wfit(self, wfit):
        """ """
        self._wfit = wfit
        self._fittedparams = None


    def show_contours(self, xkey, ykey,
                     scatter=True, conf_ellipses=[2,3],
                     ell_fc="tab:grey", ell_ec=None, ell_alpha=0.2, ell_prop={},
                     marker="o", mfc="C0", mec="0.5", ms=None, 
                     ell_autoscale=True, 
                      set_label=True, label_fontsize="large",**kwargs):
        """ """
        fig = mpl.figure()
        ax = fig.add_subplot(111)


        x, y = self.fittedparams.loc[[xkey, ykey]].astype(float).values

        if conf_ellipses is not None:
            from snana_parser.utils import ellipse
            for n_std in conf_ellipses:
                ell_2 = ellipse.confidence_ellipse(x, y, 
                                                   n_std=n_std, alpha=ell_alpha, 
                                                  facecolor=ell_fc, edgecolor=ell_ec, 
                                                   **ell_prop)
                ax.add_patch(ell_2)

        if scatter:
            ax.scatter(x, y, marker=marker, facecolor=mfc, edgecolor=mec, s=ms, **kwargs)
        elif ell_autoscale:
            xrange = np.asarray([x.min(), x.max()])
            delt_x = xrange[1]-xrange[0]
            yrange = np.asarray([y.min(), y.max()])
            delt_y = yrange[1]-yrange[0]

            ax.set_xlim(*(xrange+np.asarray([-0.1,0.1])*delt_x))
            ax.set_ylim(*(yrange+np.asarray([-0.1,0.1])*delt_y))

        if set_label:
            ax.set_xlabel(xkey, fontsize=label_fontsize)
            ax.set_ylabel(ykey, fontsize=label_fontsize)        

        return fig
    # ================= #
    #     Propeties     #
    # ================= #
    @property
    def fitres(self):
        """ """
        return self._fitres

    @property
    def m0dif(self):
        """ """
        return self._m0dif
    
    @property
    def wfit(self):
        """ """
        return self._wfit

    @property
    def fittedparams(self):
        """ concat of wfit and fitres.fitparams """
        if not hasattr(self,"_fittedparams") or self._fittedparams is None:
            self._fittedparams = pandas.concat([self.fitres.fitparam, self.wfit])
            
        return self._fittedparams
