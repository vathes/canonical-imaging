import datajoint as dj

from .imaging import schema
from .utils import dict_to_hash
from uuid import UUID


def _insert_new_params(tbl_class, param_set_name: str, params: dict):
    param_dict = {'param_set_name': param_set_name,
                  'params': params,
                  'param_set_hash': UUID(dict_to_hash(params))}
    q_param = tbl_class & {'param_set_hash': param_dict['param_set_hash']}

    if q_param:  # If the specified param-set already exists
        pname = q_param.fetch1('param_set_name')
        if pname == param_set_name:  # If the existed set has the same name: job done
            return
        else:  # If not same name: human error, trying to add the same paramset with different name
            raise dj.DataJointError('The specified param-set already exists - name: {}'.format(pname))
    else:
        tbl_class.insert1(param_dict)


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

    @classmethod
    def insert_new_params(cls, param_set_name: str, params: dict):
        _insert_new_params(cls, param_set_name, params)

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

    @classmethod
    def insert_new_params(cls, param_set_name: str, params: dict):
        _insert_new_params(cls, param_set_name, params)

