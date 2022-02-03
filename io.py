import numpy as np
import os
from glob import glob
import pandas
import warnings

SNANA__OUT_ROOTDIR = "/sps/ztf/users/nnicolas/Data/sims/NN_COMBINE_VSIZE"
SNANA_PARSER__OUTDIR = SNANA__OUT_ROOTDIR

LCFIT_MERGE_DATAFILE = os.path.join(SNANA_PARSER__OUTDIR,"2_LCFIT/merge_data_datafile.parquet")


class _DataFileIO_():
    
    ROOTDIR = "UNKNOWN"
    DATAFILE_NAME = "UNKNOWN.parquet"
    def __init__(self, datafile=None):
        """ """
        self.set_datafile(datafile)

    # =============== #
    #   I/O           #
    # =============== #
    @classmethod
    def get_datafile_filename(cls):
        """ """
        return os.path.join(SNANA_PARSER__OUTDIR, cls.ROOTDIR, cls.DATAFILE_NAME)
    
    @classmethod
    def build_fromdir(cls, lcfit_rootdir=None, store=False):
        """ """
        if lcfit_rootdir is None:
            lcfit_rootdir = os.path.join(SNANA__OUT_ROOTDIR,cls.ROOTDIR)
            
        this = cls( cls.build_datafile_fromdir(lcfit_rootdir) )
        if store:
            this.store()
        return this
    
    
    @classmethod
    def load(cls, filepath=None, **kwargs):
        """ """
        if filepath is None:
            default_filename = cls.get_datafile_filename()
            if not os.path.isfile(default_filename):
                warnings.warn(f"the default loading file {default_filename} does not exists. Running build_fromdir()")
                return cls.build_fromdir(store=True)
            
            filepath = default_filename
            
        extension = filepath.split(".")[-1]
        datafile = getattr(pandas,f"read_{extension}")(filepath, **kwargs)
        return cls(datafile)
    
    def store(self, filepath=None, makedirs=True, **kwargs):
        """ """
        if filepath is None:
            filepath = self.get_datafile_filename()
            
        extension = filepath.split(".")[-1]
        if makedirs:
            dirname = os.path.dirname(filepath)
            os.makedirs(dirname, exist_ok=True)
        
        return getattr(self.datafile, f"to_{extension}")(filepath, **kwargs)    

    # =============== #
    #   Methods       #
    # =============== #        
    def set_datafile(self, datafile):
        """ """
        self._datafile = datafile


    def get_filepath(self, **kwargs):
        """ """
        local = kwargs
        gkeys = list(local.keys())
        values = list(local.values())        
        if len(gkeys)==0:
            raise ValueError("you must provide at least 1 colum to groupby")
        
        gp = self.datafile.groupby(gkeys)["fullpath"].apply(list)

        if len(gkeys) == 1:
            return gp.loc[values[0]]
        

        lkeys, values = np.asarray([[k,v] for k,v in local.items() if v is not None and v not in ["*","all"]],
                                           dtype="object").T
        
        return gp.xs(values, level=list(lkeys)).reset_index()

        
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def datafile(self):
        """ """
        return self._datafile
    
    
class LCFitIO(_DataFileIO_):
    
    ROOTDIR = "2_LCFIT"
    DATAFILE_NAME = "merged_simdata_datafile.parquet"
    
    # =============== #
    #   I/O           #
    # =============== #
    @staticmethod
    def build_datafile_fromdir(lcfit_rootdir, input_version="*"):
        """ static method that builds the datafile.
        
        See build_fromdir() for the classmethod that build and loads.
        
        """
        rootfiles = glob( os.path.join(lcfit_rootdir,
                               f'*D_*_DATA-{input_version}/output/PIP_NN_VSIZE_{input_version}_*_DATA-{input_version}-*/FITOPT000.ROOT')
                               )

        datafile = pandas.DataFrame(rootfiles, columns=["fullpath"])
        datafile[["sourcedir","outdir"]] = datafile["fullpath"].str.split("/", expand=True, )[[9,11]]
        split_name = datafile["outdir"].str.split("-",expand=True)
        datafile["sim_number"] = split_name.iloc[:,-1].astype(int)
        datafile[["true_model","survey"]] = split_name[0].str.split("_",expand=True).iloc[:,[-3,-2]]
        return datafile

    def get_filepath(self, true_model, **kwargs):
        """ """
        return super().get_filepath(true_model=true_model, **kwargs)
    
    
class BiasCorIO(_DataFileIO_):
    
    ROOTDIR = "6_BIASCOR"
    DATAFILE_NAME = "merged_biascor_datafile.parquet"
    
    # =============== #
    #   I/O           #
    # =============== #
    @staticmethod
    def build_datafile_fromdir(lcfit_rootdir, input_version="*", fitted_version="*"):
        """ static method that builds the datafile.
        
        input_version or fitted_version could be NN, NR, SK, BP
        
        See build_fromdir() for the classmethod that build and loads.
        
        """
        rootfiles = glob( os.path.join(lcfit_rootdir,
                               f"ALL_BIASCOR_{input_version}_{fitted_version}",
                                "output/OUTPUT_BBCFIT-*/*MUOPT*"
                               )
                        )
        datafile = pandas.DataFrame(rootfiles, columns=["fullpath"])
        
        datafile[["sourcedir","outdir", "filename"]] = datafile["fullpath"].str.split("/", expand=True)[[9,11,12]]
        datafile[["true_model","fit_model"]] = datafile["sourcedir"].str.split("_", expand=True).iloc[:,[-2,-1]]

        datafile["sim_number"] = datafile["outdir"].str.split("-",expand=True).iloc[:,-1].astype("int")

        opts = datafile["filename"].str.split("OPT",expand=True)[[1,2]]
        datafile["fitopt"] = opts[1].str.split("_",expand=True)[0].astype("int")
        datafile["muopt"] = opts[2].str.split(".",expand=True)[0].astype("int")
        datafile["which"] = datafile["filename"].str.split(".", expand=True).iloc[:,-1].replace("YAML","wfit").str.lower()
        
        return datafile
    
    def get_filepath(self, true_model, fit_model, which, muopt=0, fitopt=0, **kwargs):
        """ """
        return super().get_filepath(true_model=true_model, fit_model=fit_model,
                                    which=which, muopt=muopt, fitopt=fitopt, **kwargs)

    
