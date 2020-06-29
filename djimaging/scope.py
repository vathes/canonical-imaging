import datajoint as dj
import numpy as np

from djutils.templates import SchemaTemplate


schema = SchemaTemplate()

# ======================== Scope info and ScanImage acquisition software =========================


@schema
class Field(dj.Lookup):
    definition = """ # fields in mesoscope scans
    field       : tinyint
    """
    contents = [[i] for i in range(1, 25)]


@schema
class Channel(dj.Lookup):
    definition = """  # recording channel
    channel     : tinyint
    """
    contents = [[i] for i in range(1, 5)]

    