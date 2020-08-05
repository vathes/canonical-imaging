import datajoint as dj

from .imaging import schema


# ===================================== CaImAn =====================================

@schema
class CaimanParamSet(dj.Manual):
    definition = """
    param_set_name: varchar(36)
    ---
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable CaImAn parameters
    """


# ===================================== Suite2p =====================================

@schema
class Suite2pParamSet(dj.Manual):
    definition = """
    param_set_name: varchar(36)
    ---
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable suite2p params (ops.npy)
    """
