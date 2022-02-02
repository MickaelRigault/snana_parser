""" Tools: fitting step fit. """

import pandas
import numpy as np
from scipy import stats
import warnings


class StepFitter():
    
    PARAMNAMES = ["mu_u","sigma_u","mu_d","sigma_d"]
    
    def __init__(self, data, meta=None):
        """ see from_xyvariables() """
        self.set_data(data)
        self.set_meta(meta)
        
    @classmethod
    def from_xyvariables(cls, x, y, dx=None, dy=None, p_up=None,
                        xcut=None, index=None):
        """ """
        meta = {}
        if p_up is None:
            if xcut is None:
                raise ValueError(f"neither probaup nor xcut are given. One must.")
            if dx is not None:
                p_up = 1-stats.norm.cdf(xcut, loc=x, scale=dx)
            else:
                p_up = np.asarray(x>xcut, dtype="float")
                
            meta["xcut"] = xcut
            
        if dy is None:
            dy = np.zeros(len(x))
            
        data = pandas.DataFrame({"x":x, "y":y, "dy":dy, "p_up":p_up})        
        return cls(data, pandas.Series(meta))

    @classmethod
    def derive_step(cls, x, y, dx=None, dy=None, p_up=None,
                        xcut=None, **kwargs):
        """ """
        this = cls.from_xyvariables(x, y, dx=dx, dy=dy, p_up=p_up,
                                        xcut=xcut, **kwargs)
        _ = this.fit(**kwargs)
        return this.get_step()
    
    # ============== #
    #   Methods      #
    # ============== #
    # --------- #
    #  GETTER   #
    # --------- #
    def fit(self, fit_mlogprior=True, guess=None, maxcall=1000, **kwargs):
        """ """
        import iminuit        
        if guess is None:
            guess = {}
        
        guess_ = self.get_guess(**guess)
        if fit_mlogprior:
            func = self._get_minus_logposterior_
        else:
            func = self.get_chi2
        
        m = iminuit.Minuit(func,
                           name=self.PARAMNAMES,
                           **guess_, 
                          **kwargs)
        m.errordef = 1.
        #m.limits = [(None, None), (0, None), (None, None), (0, None)]
        res = m.migrad(ncall=maxcall)
        
        data, cols_ = res.params.to_table()
        results = pandas.DataFrame(data, columns=cols_).set_index("name")
        covmatrix = pandas.DataFrame(res.covariance, columns= res.parameters, index=res.parameters)
        self.set_results(results, covmatrix)
        
        return res
        
    def _get_minus_logposterior_(self, *parameters, **kwargs):
        """ """
        return -self.get_logposterior(*parameters, **kwargs)
    
    def get_logposterior(self, *parameters, index=None):
        """ """
        loglikelihood = self.get_loglikelihood(*parameters, index=index)
        logprior = self.get_logprior(*parameters)
        
        return loglikelihood + logprior
    
    def get_loglikelihood(self, *parameters, index=None):
        """ sum_i log(pdf_i) """
        return np.sum( np.log(self.get_pdf(*parameters, index=index)) )
        
    def get_chi2(self, *parameters):
        """ """
        return -2*self.get_loglikelihood(*parameters)
        
    def get_pdf(self, *parameters, index=None):
        """ """
        # convert into dict if needed:
        if parameters is not None and len(parameters)>0:
            parameters = self._read_parameters_(parameters)
        else:
            parameters = self.parameters
            
        p_up, data_, err_ = self.get_fittedvalues(index=index)
        
        pdf_u = stats.norm.pdf( data_, 
                                loc=parameters["mu_u"],
                                scale=np.sqrt(parameters["sigma_u"]**2 + err_**2)
                              )
        
        pdf_d = stats.norm.pdf( data_, 
                                loc=parameters["mu_d"],
                                scale=np.sqrt(parameters["sigma_u"]**2 + err_**2)
                              )
        return p_up*pdf_u + (1-p_up)*pdf_d
    
    def get_guess(self, index=None, **kwargs):
        """ """
        p_up, data_, err_ = self.get_fittedvalues(index=index)
        
        mu_u = data_[p_up>0.5].mean()
        var_u = data_[p_up>0.5].std()**2-err_[p_up>0.5].mean()**2
        
        mu_d = data_[p_up<=0.5].mean()
        var_d = data_[p_up<=0.5].std()**2-err_[p_up<=0.5].mean()**2
        
        # make sure var is defined positive.
        guess = {"mu_u": mu_u,
                 "sigma_u": np.sqrt(np.max([var_u, 0.0001])),
                 "mu_d": mu_d,
                 "sigma_d": np.sqrt(np.max([var_d, 0.0001])),
                }
        return {**guess, **kwargs}
        
    def get_fittedvalues(self, index=None):
        """ """
        data = self.data if index is None else self.data.loc[index]
        return np.asarray(data[["p_up","y","dy"]].values, dtype="float").T
        
    def get_logprior(self, *parameters, 
                 sigma_logistic={"loc":0.001, "scale":0.0001}):
        """z """
        if parameters is not None and len(parameters)>0:
            parameters = self._read_parameters_(parameters)
        else:
            parameters = self.parameters
            
        prior_sigma_u = stats.logistic.cdf(parameters["sigma_u"], 
                                           **sigma_logistic)
        prior_sigma_d = stats.logistic.cdf(parameters["sigma_d"], 
                                           **sigma_logistic)
        
        prior = prior_sigma_u*prior_sigma_d
        return np.log(prior)
    
    def get_step(self):
        """ mu_d - mu_u"""
        step = float(self.results.value.mu_d)-float(self.results.value.mu_u)
        err = np.sqrt(self.covariance.loc["mu_u","mu_u"] + self.covariance.loc["mu_d","mu_d"]
                     -2*self.covariance.loc["mu_u","mu_d"])
        return step, err
    
    # --------- #
    #  SETTER   #
    # --------- #                  
    def set_results(self, results, covariance=None):
        """ """
        self._results = results
        self._covariance = covariance
        
    def set_data(self, data):
        """ """
        self._data = data
        
    def set_meta(self, meta):
        """ """
        self._meta = meta

    # ============== #
    #   INTERNAL     #
    # ============== #
    @classmethod
    def _read_parameters_(cls, parameters):
        """ """
        if type(parameters) == dict:
            return parameters

        return {k:v for k,v in zip(cls.PARAMNAMES, np.squeeze(parameters))}
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def data(self):
        """ """
        return self._data
        
    @property
    def meta(self):
        """ """
        return self._meta
        
    @property
    def results(self):
        """ dictionary of the current parameter values """
        if not hasattr(self, "_results"):
            return None

        return self._results
    
    @property
    def covariance(self):
        """ dictionary of the current parameter values """
        if not hasattr(self, "_covariance"):
            return None

        return self._covariance
    
