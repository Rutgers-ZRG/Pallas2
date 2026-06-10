"""Back-compatibility shim: PALLAS modules were split out of core.py.

Import from pallas (package root) or the specific modules instead.
"""

from pallas.analysis import listpath, main
from pallas.config import PallasConfig
from pallas.search import Pallas
from pallas.structure import PallasAtom, fp_distance

__all__ = ["Pallas", "PallasConfig", "PallasAtom", "fp_distance", "listpath", "main"]
