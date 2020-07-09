import datajoint as dj

from .imaging import schema


# ===================================== CaImAn =====================================

@schema
class CaImAnParamSet(dj.Part):
    definition = """
    param_set_name: varchar(36)
    ---
    param_set_hash: uuid
    unique (param_hash)
    params: longlob  # dictionary of all applicable CaImAn parameters
    """
