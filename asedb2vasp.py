import sys
import ase.db
import numpy as np
from ase import Atoms
from ase.io import read, write



def main():
    id = int(sys.argv[1])
    db = ase.db.connect('pallas.json')
    xx = db.get_atoms(id)
    write('output.vasp', xx, format='vasp', direct=True)


if __name__ == '__main__':
    main()
