
import os
from glob import glob

def get_lcfit_datafile(lcfit_rootdir, input_version="SK",
                           fitted_version="SK"):
    """ """
    import pandas
    "/sps/ztf/users/nnicolas/Data/sims/NN_COMBINE_VSIZE/2_LCFIT"
    rootfiles = glob(os.path.join(lcfit_rootdir, f'*D_*_SIM-{input_version}/output/PIP_{input_version}_VSIZE_{input_version}_*_DATA-{fitted_version}-*/FITOPT000.ROOT')
    datafile = pandas.DataFrame(rootfiles, columns=["fullpath"])
    datafile[["sourcedir","outdir"]] = datafile["fullpath"].str.split("/", expand=True, )[[9,11]]

    datafile["sim_number"] = datafile["outdir"].str.split("-",expand=True).iloc[:,-1].astype(int)
    datafile["sim_input"] = datafile["outdir"].str.split("-",expand=True).iloc[:,1]
    datafile["sim_fitted"] = datafile["outdir"].str.split("-",expand=True).iloc[:,-2]
    return datafile
