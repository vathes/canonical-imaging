import datajoint as dj

from djutils.templates import SchemaTemplate, required

schema = SchemaTemplate()


@schema
class PhysicalFile(dj.Manual):
    definition = """
    file_path: varchar(1000)  # filepath relative to root data directory
    """

    @staticmethod
    @required
    def _get_root_data_dir():
        return None
