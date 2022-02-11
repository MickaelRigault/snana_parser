import dask
import pandas as pd
import simulations

from .utils import root
from .io import LCFitIO, SNANA__OUT_ROOTDIR


def get_merged_simdata_file(true_model):
    """ """
    return LCFitIO.get_datafile_filename().replace("datafile",
                                                   f"truemodel-{true_model}")


def build_merged_simdata(true_model, as_dask=None):
    """as_dask: None --> returns nothing and stores
                'compute' --> returns the multi-index dataframe
                'delayed' --> returns the delayed dataframe"""
    mdatafile = pd.read_parquet(SNANA__OUT_ROOTDIR +
                                '/2_LCFIT/merged_simdata_datafile.parquet')
    tmodel_datafile = mdatafile.query(f"true_model=='{true_model}'")
    surveys = tmodel_datafile.groupby('survey')['fullpath'].apply(list)
    d_cdf = []
    for survey in surveys.index:
        d_df = []
        fid = []
        for file_ in surveys[survey]:
            fid.append(int(file_.split('/')[-2].split('-')[-1]))
            d_df.append(dask.delayed(root.read_snana_root)(file_))

        d_cdf.append(dask.delayed(pd.concat)(d_df, keys=fid))
    d_cdf_t = dask.delayed(pd.concat)(d_cdf, keys=surveys.index)
    if as_dask == 'delayed':
        return d_cdf_t
    cdf_t = d_cdf_t.compute()
    if as_dask == 'compute':
        return cdf_t
    filename = simulations.get_merged_simdata_file(true_model)
    return cdf_t.to_parquet(filename)
