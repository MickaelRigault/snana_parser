

import numpy as np
import pandas


def parse_m0dif_data(filename):
    """ """
    return pandas.read_csv(filename, comment="#", delim_whitespace=True).drop(columns="VARNAMES:")

def parse_m0dif_params(filename):
    """ """
    params = [[l_.strip() for l_ in l.replace("#","").split(": ")] for l in open(filename).read().splitlines() 
              if l.startswith("#") if len(l)>5 and "Reference" not in l]
    warnings = [[p[0].replace("WARNING(","").replace(")",""),*p[1:]] for p in params if "WARNING" in p[0]]
    warnings = pandas.DataFrame(warnings, columns=["warning","message"]).set_index("warning")
    parameter = [p for p in params if not "WARNING" in p[0]]
    parameter= pandas.DataFrame(parameter, columns=["parameter","value"]).set_index("parameter")
    return parameter, warnings



class M0DIF():
    
    def __init__(self, data, params, warnings):
        """ """
        self.set_data(data)
        self.set_params(params)
        self.set_warnings(warnings)

    @classmethod
    def from_m0dif_files(cls, filenames, use_dask=True, filekeys=None, client=None):
        """ combine a list of FITRES file into a single FITES FILE Object """
        # Get the ID
        if filekeys is None:
            filekeys = np.asarray([file_.split("/")[-2].split("-")[-1]
                                   for file_ in filenames],
                      dtype="int")
        
        if use_dask:
            import dask
            outputs = []
            d_data = [dask.delayed(parse_m0dif_data)(f_) for f_ in filenames]
            d_params_wargs = [dask.delayed(parse_m0dif_data)(f_) for f_ in filenames]
            d_alls = dask.delayed(list)([d_data,d_params_wargs])
            if client is None:
                alls = dask.delayed(list)(d_alls).compute()
            else:
                alls = client.gather(client.compute(d_alls))
            data, params_warnings = alls
        else:
            data = [parse_m0dif_data(f_) for f_ in filenames]
            params_warnings = [parse_m0dif_params(f_) for f_ in filenames]
            
        data_all = pandas.concat(data, keys=filekeys)
        params = pandas.concat([d_ for d_,i_ in alls], axis=1, keys=filekeys)
        warnings = pandas.concat([i_ for d_,i_ in alls], axis=1, keys=filekeys)
        return cls(data, params, warnings)

    @classmethod
    def from_m0dif_file(cls, filename):
        """ """
        data = parse_m0dif_data(filename)
        params, warnings = parse_m0dif_params(filename)
        return cls(data, params, warnings)

    # ================= #
    #    Methods        #
    # ================= #
    # --------- #
    #  GETTER   #
    # --------- #
    
    # --------- #
    #  SETTER   #
    # --------- #
    def set_data(self, data):
        """ """
        self._data = data

    def set_params(self, params):
        """ """
        self._params = params

    def set_warnings(self, warnings):
        """ """
        self._warnings = warnings
        
    # ================= #
    #    Properties     #
    # ================= #
    @property
    def data(self):
        """ """
        return self._data

    @property
    def params(self):
        """ """
        return self._params

    @property
    def warnings(self):
        """ """
        return self._warnings
