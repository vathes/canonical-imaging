import numpy as np
import pathlib


_suite2p_ftypes = ('ops', 'Fneu', 'Fneu_chan2', 'F', 'F_chan2', 'iscell', 'spks', 'stat', 'redcell')


class Suite2p:
    """
    Parse the suite2p output directory and load data. Expecting the following files:
    - 'ops':        Options file
    - 'Fneu':       Neuropil traces file for functional channel
    - 'Fneu_chan2': Neuropil traces file for channel 2
    - 'F':          Fluorescence traces for functional channel
    - 'F_chan2':    Fluorescence traces for channel 2
    - 'iscell':     Array of (user curated) cells and probability of being a cell
    - 'spks':       Spikes (raw deconvolved with OASIS package)
    - 'stat':       Various statistics for each cell
    - 'redcell':    "Red cell" (second channel) stats

    Suite2p output doc: https://suite2p.readthedocs.io/en/latest/outputs.html
    """

    def __init__(self, suite2p_dir):
        self.fpath = pathlib.Path(suite2p_dir)
        for s2p_type in _suite2p_ftypes:
            setattr(self, f'_{s2p_type}', None)

        self._cell_prob = None

    # ---- load core files ----

    @property
    def ops(self):
        if self._ops is None:
            fp = self.fpath / 'ops.npy'
            self._ops = np.load(fp, allow_pickle=True).item() if fp.exists() else {}
        return self._ops

    @property
    def Fneu(self):
        if self._Fneu is None:
            fp = self.fpath / 'Fneu.npy'
            self._Fneu = np.load(fp) if fp.exists() else []
        return self._Fneu

    @property
    def Fneu_chan2(self):
        if self._Fneu_chan2 is None:
            fp = self.fpath / 'Fneu_chan2.npy'
            self._Fneu_chan2 = np.load(fp) if fp.exists() else []
        return self._Fneu_chan2

    @property
    def F(self):
        if self._F is None:
            fp = self.fpath / 'fneu.npy'
            self._F = np.load(fp) if fp.exists() else []
        return self._F

    @property
    def F_chan2(self):
        if self._F_chan2 is None:
            fp = self.fpath / 'F_chan2.npy'
            self._F_chan2 = np.load(fp) if fp.exists() else []
        return self._F_chan2

    @property
    def iscell(self):
        if self._iscell is None:
            fp = self.fpath / 'iscell.npy'
            if fp.exists():
                d = np.load(fp)
                self._iscell = d[:, 0].astype(bool)
                self._cell_probe = d[:, 1]
        return self._iscell

    @property
    def cell_prob(self):
        if self._cell_prob is None:
            fp = self.fpath / 'iscell.npy'
            if fp.exists():
                d = np.load(fp)
                self._iscell = d[:, 0].astype(bool)
                self._cell_probe = d[:, 1]
        return self._cell_prob

    @property
    def spks(self):
        if self._spks is None:
            fp = self.fpath / 'spks.npy'
            self._spks = np.load(fp) if fp.exists() else []
        return self._spks

    @property
    def stat(self):
        if self._stat is None:
            fp = self.fpath / 'stat.npy'
            self._stat = np.load(fp, allow_pickle=True) if fp.exists() else []
        return self._stat

    @property
    def redcell(self):
        if self._redcell is None:
            fp = self.fpath / 'redcell.npy'
            self._redcell = np.load(fp) if fp.exists() else []
        return self._redcell

    # ---- derived property ----



