""" Parse FITRES files """

import pandas
import numpy as np

from scipy import stats
import re


def parse_fitresfile(filename, lochost=10, scale=0.1):
    """ """
    fitparam = parse_fitres_fitparam(filename)
    data = parse_fitres_data(filename, hostgamma=fitparam["gamma"])
    return data, fitparam

def parse_fitres_data(filename, hostgamma=None, lochost=10, scale=0.1):
    """ """
    sndatares = pandas.read_csv(filename, comment="#", delim_whitespace=True).drop(columns="VARNAMES:")
    if hostgamma is not None:
        sndatares["DELTA_HOST"] = float(hostgamma)*( stats.logistic.cdf(sndatares["HOST_LOGMASS"].astype(float), 
                                                                         loc=lochost, scale=scale) - 0.5)
    return sndatares

def parse_fitres_fitparam(filename):
    """ """
    comments = [re.split("[:=]",l.replace("#","").strip()) for l in open(filename).read().splitlines() 
                if l.startswith("#") 
                and not l.startswith("# WARN") and not "\t" in l]
    
    d_ = pandas.DataFrame(comments, columns=["parameter", "value","comments"])
    d_["parameter"] = d_["parameter"].str.strip()
    d_ = d_.set_index("parameter")
    params = ["ISDATA_REAL","SNANA_VERSION","BBC_VERSION", "NCALL_FCN","CPU", "NSNFIT",
              "-2log(L)", "chi2(Ia)/dof", "sigint", "alpha0","beta0","gamma0"]
    d_ = d_.loc[params]["value"]

    fitvalues = np.asarray(d_[["alpha0","beta0","gamma0"]
                             ].astype("string").str.split(expand=True)[[0,2]].values,
                           dtype="float").flatten()

    d_["sigmaint"], sigmaint_iter = d_["sigint"].split("  ")
    d_["sigmaint_iter"] = int(sigmaint_iter.replace("(","").split()[0])

    fitdata = pandas.concat([d_, 
                  pandas.Series(fitvalues,index=["alpha","e_alpha",
                                                 "beta","e_beta",
                                                 "gamma","e_gamma"])
                            ])
    _ = [fitdata.pop(k) for k in ["alpha0","beta0","gamma0","sigint"]]    

    return fitdata.astype("string").str.strip()


############################
#                          #
#                          #
#       FITRES             #
#                          #
#                          #
############################
class FITRES():

    def __init__(self, dataset, fitparam=None):
        """ """
        self.set_data(dataset)
        self.set_fitparam(fitparam)
        
    # ================= #
    #      I/O          #
    # ================= #
    @classmethod
    def from_fitres_files(cls, filenames, use_dask=True, filekeys=None, client=None):
        """ combine a list of FITRES file into a single FITES FILE Object """
        # Get the ID
        if filekeys is None:
            # vectorized version of int( file.split('/')[-2].split('-')[-1] )
            filekeys = np.asarray([file_.split("/")[-2].split("-")[-1]
                                   for file_ in filenames],
                      dtype="int")
        
        if use_dask:
            fileout = [dask.delayed(fitres.parse_fitresfile)(f_) for f_ in filenames]
            if client is None:
                alls = dask.delayed(list)(fileout).compute()
            else:
                alls = client.gather(client.compute(fileout))
        else:
            alls = [fitres.parse_fitresfile(f_) for f_ in filenames]
            
        data_all = pandas.concat([d_ for d_,i_ in alls], keys=filekeys)
        fitparams = pandas.concat([i_ for d_,i_ in alls], axis=1, keys=filekeys)
        return cls(data_all, fitparams)
    
    @classmethod
    def from_fitres_file(cls, filename):
        """ """
        dataset, fitparam = parse_fitresfile(filename)
        return cls(dataset=dataset, fitparam=fitparam)

    # ================= #
    #    Methods        #
    # ================= #
    # --------- #
    #  SETTER   #
    # --------- #    
    def set_data(self, data):
        """ """
        self._data = data
        
    def set_fitparam(self, fitparam):
        """ """
        self._fitparam = fitparam    
        
    # --------- #
    #  GETTER   #
    # --------- #
    # - get data
    def get_data(self, index=None, columns=None):
        """  """
        data = self.data.copy() if index is None else self.data.loc[index]
        if columns is not None:
            return data[columns]
        
        return data
            
    def get_splitindex(self, keys, split, method="qcut", **kwargs):
        """ """
        return getattr(pandas,method)(self.data[keys], split, **kwargs)
    
    def get_massstep(self, masscut=10, index=None):
        """ """
        from .utils.stepfit import StepFitter
        mu, dmu, hostmass,dhostmass, deltahost = np.asarray(self.get_data(index=index, 
                                                     columns=["MURES","MUERR_RAW",
                                                              "HOST_LOGMASS", "HOST_LOGMASS_ERR",
                                                              "DELTA_HOST"]).values, dtype="float").T
        mu -= deltahost
        return StepFitter.derive_step(hostmass, mu, dx=dhostmass, dy=dmu, xcut=masscut)
        
    # - get parameters
    def get_standardisation_param(self):
        """ """
        return self.fitparam.loc[["alpha","e_alpha","beta","e_beta","gamma","e_gamma"]]
    
    def get_sigmaint(self):
        """ """
        return float(self.fitparam["sigmaint"])

    # ================= #
    #   Properties      #
    # ================= #
    @property
    def data(self):
        """ """
        return self._data
    
    @property
    def fitparam(self):
        """ """
        return self._fitparam
    
        
