"""Change format of the sourcedata to make BIDS complient."""
from os.path import basename

from mne import set_log_level

from scripts.bidsify_sourcedata import (bidsify_sourcedata_hirschmann,
                                        bidsify_sourcedata_litvak,
                                        bidsify_sourcedata_neumann,
                                        bidsify_sourcedata_tan,
                                        bidsify_sourcedata_florin)


def bidsify_sourcedata(neumann=False, litvak=False, hirschmann=False,
                       tan=False, florin=False, verbose="error"):
    set_log_level(verbose)
    if neumann:
        bidsify_sourcedata_neumann()
    if litvak:
        bidsify_sourcedata_litvak()
    if hirschmann:
        bidsify_sourcedata_hirschmann()
    if tan:
        bidsify_sourcedata_tan()
    if florin:
        bidsify_sourcedata_florin()
    print(f"{basename(__file__).strip('.py')} done.")
    return None


if __name__ == "__main__":
    bidsify_sourcedata()
