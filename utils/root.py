
import uproot
import numpy as np
import pandas

def read_snana_root(filename, keys="FITRES", use_dask=False):
    """ reads the given root file, loops over the keys inside the
    SNANA entry, converts the output in numpy array and finally build
    a DataFrame.
    
    use_dask to use dask for looping over the keys and convert the
    root array in numpy array.

    Returns
    -------
    DataFrame
    """
    rootdata = uproot.open(filename)
    if not use_dask:
        dd = {k:np.asarray(v.array()) for k, v in rootdata[keys].iteritems()}
    else:
        import dask
        d_dict = {}
        for k, v in rootdata[keys].iteritems():
            d_v = dask.delayed(v.array)()
            d_np = dask.delayed(np.asarray)(d_v)
            d_dict[k] = d_np
    
        dd = dask.compute(d_dict)[0]
    
    return pandas.DataFrame.from_dict(dd)
        
