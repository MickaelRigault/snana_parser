

def snana_root_to_dataframe(filename, keys="SNANA"):
    """ """
    rootdata = uproot.open(filename)
    return pandas.DataFrame({k:v.array() for k, v in rootdata[keys].iteritems()})


