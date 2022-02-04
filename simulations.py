

import pandas

from .io import LCFitIO


def get_merged_simdata_file(true_model):
    """ """
    return LCFitIO.get_datafile_filename().replace("datafile", f"truemodel-{true_model}")

def build_merged_simdata(true_model, client, fileout = None, makedir=True):
    """ """
    from .utils.root in read_snana_root
    if fileout is None:
        fileout = get_merged_simdata_file(true_model)

    extension = fileout.split(".")[-1]
    if makedir:
        dirname = os.path.dirname(fileout)
        os.makedirs(dirname, exist_ok=True)
    
    
    files = LCFitIO.fetch_filepaths(true_model=true_model)
    f_df = client.map(read_snana_root, files, keys="FITRES", use_dask=False)
    dd_ = pandas.concat(client.gather(f_df) )
    return getattr(dd_,f"to_{extension}")( fileout )


