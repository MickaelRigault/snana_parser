
import numpy as np
import pandas

def read_wfit_file(filename):
    """ """
    return pandas.read_csv(filename, 
                    sep=":", names=["parameter","value"], index_col=0)["value"]

def read_wfit_files(filenames, filekeys=None, use_dask=True, client=None):
    """ """
    if filekeys is None:
        filekeys = np.asarray([file_.split("/")[-2].split("-")[-1]
                               for file_ in filenames],
                  dtype="int")

    if use_dask:
        import dask
        fileout = [dask.delayed(read_wfit_file)(f_) for f_ in filenames]
        if client is None:
            alls = dask.delayed(list)(fileout).compute()
        else:
            alls = client.gather(client.compute(fileout))
    else:
        alls = [read_wfit_file(f_) for f_ in filenames]

    wdata = pandas.concat(alls, keys=filekeys, axis=1)
    return wdata
