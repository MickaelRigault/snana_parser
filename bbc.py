
import os
import glob


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
    def set_fitres(self, fitres):
        """ """
        self._fitres = fitres

    def set_m0dif(self, m0dif):
        """ """
        self._m0dif = m0dif

    def set_wfit(self, wfit):
        """ """
        self._wfit = wfit
        
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
