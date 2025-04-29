import sys
import os
import socket
from copy import deepcopy as cp
import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.db
import networkx as nx
import joblib
from tsase.neb.util import vunit, vrand
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import fplib2
import time
from collections import deque  # Added for pathfinding

from nequipcal import local_optimization, cal_saddle

class Pallas(object):
    def __init__(self, popsize=10, maxstep=50, press=0.0, fpcutoff=5.5, lmax=0, natx=200, ntyp=2, types=None):
        """Initialize the Pallas PSO search.

        Args:
            popsize (int): Population size for PSO.
            maxstep (int): Maximum number of PSO steps.
            press (float): External pressure.
            fpcutoff (float): Cutoff radius for fingerprints.
            lmax (int): Maximum l value for fingerprints.
            natx (int): Maximum number of atoms expected (for FP array size).
            ntyp (int): Number of atom types.
            types (np.ndarray): Array mapping atom index to type index (1-based).
        """
        self.popsize = popsize
        self.maxstep = maxstep
        self.press = press
        self.fpcutoff = fpcutoff
        self.lmax = lmax
        self.natx = natx # Note: PallasAtom uses a hardcoded natx=200, needs sync
        self.ntyp = ntyp
        # Default types if none provided (example: 4 of type 1, 4 of type 2)
        self.types = types if types is not None else np.array([1]*4 + [2]*4) 
        
        self.db = None # Database connection, initialized in init_run
        self.G = nx.Graph() # Graph to store minima and saddles
        self.baseenergy = 0.0 # Reference energy (usually reactant energy)
        self.reactant = None # Reactant structure (PallasAtom)
        self.product = None # Product structure (PallasAtom)
        self.reactant_id = None
        self.product_id = None

        # PSO State Variables (initialize as empty lists)
        # These will hold data for each particle in the population (size = popsize)
        self.stepx_particles = [] # List of ParticleState objects for reactant side
        self.stepy_particles = [] # List of ParticleState objects for product side

        # Global best found so far (connecting reactant and product)
        self.gbest_particle_x = None # Best particle state from reactant side leading to connection
        self.gbest_particle_y = None # Best particle state from product side leading to connection
        self.gbest_distance = float('inf') # Best fingerprint distance found between sides
        self.gbest_barrier = float('inf') # Lowest barrier found for a connected path

        # Keep track of all minima/saddles found (optional, maybe handled by db/graph)
        # self.all_minima_map = {} # Store minima by ID
        # self.all_saddle_map = {} # Store saddles by ID 

        # Distance matrix (potentially large, consider if needed or use on-the-fly calcs)
        # self.dij = np.zeros((10000, 10000), float) 
        # self.dij[:][:] = 1000.
        # np.fill_diagonal(self.dij, 0.0)


# Helper class to store state for each particle in PSO
class ParticleState:
    def __init__(self):
        self.min = None        # Current minimum structure (PallasAtom)
        self.sad = None        # Saddle point generated from self.min (PallasAtom)
        self.v = None          # PSO velocity (mode for perturbation)
        self.pbest = None      # Personal best minimum structure found (PallasAtom)
        self.pbest_distance = float('inf') # Best distance to the *other* side for this particle's pbest

    def init_run(self, flist):
        self.db = ase.db.connect('pallas.json')
        self.dij[:][:] = 1000.
        np.fill_diagonal(self.dij, 0.0)
        # num_init_min = 2 # for testing
        self.init_minima = self.read_init_structure(flist)
        self.num_init_min = len(self.init_minima)
        visualizer = GraphVisualizer(G)
        visualizer.animate(frames=10, interval=1000)

    def read_init_structure(self, flist):
        init_minima = []
        for xf in flist:
            x = read(xf, format='vasp')
            x = PallasAtom(x)
            init_minima.append(cp(x))
        return init_minima
    
    def run(self):
        """Main loop to optimize minima and calculate saddle points."""
        xlist = []

        # Iterate through initial minima
        for i in range(self.num_init_min):
            # Optimize the initial structure
            optx = local_optimization(self.init_minima[i])
            # Update minima and get new ID
            idm, isnew = self.update_minima(optx)
            optx.id = idm
            
            # Calculate energy (h) relative to base energy
            if i == 0:
                # Set base energy for the first minimum
                self.baseenergy = optx.get_volume()*self.press/1602.176487 + optx.get_potential_energy()
                h = 0.0
            else:
                # Calculate relative energy for subsequent minima
                h = optx.get_volume()*self.press/1602.176487 + optx.get_potential_energy() - self.baseenergy
            
            # Get volume of the optimized structure
            volume = optx.get_volume()
            
            # Add node to the graph
            self.G.add_node(idm, xname='M'+str(idm), e=h, volume=volume)
            
            # Print information about the added node
            print(f"Added node: ID={idm}, Type=Minimum, Energy={h:.4f}, Volume={volume:.4f}")
            
            # Save the updated graph
            self.save_graph()  

            # For each initial minimum, perform popsize number of saddle point optimizations
            for ip in range(self.popsize):
                try:
                    sadx = cal_saddle(optx)
                except:
                    print(f"Failed to calculate saddle for Minimum {optx.id}")
                    continue
                if sadx.converged:
                    ids, isnew = self.update_saddle(sadx)
                    sadx.id = ids
                    h = sadx.get_volume()*self.press/1602.176487 + sadx.get_potential_energy() - self.baseenergy
                    volume = sadx.get_volume()
                    self.G.add_node(ids, xname='S'+str(ids), e=h, volume=volume)
                    # Add edge with weight (saddle energy)
                    self.G.add_edge(idm, ids, weight=h)  
                    print(f"Added node: ID={ids}, Type=Saddle, Energy={h:.4f}, Volume={volume:.4f}")
                    print(f"Added edge: Minimum {idm} -> Saddle {ids}")
                    self.save_graph()
                    xlist.append(cp(sadx))

        # Iterate until reaching maxstep
        for istep in range(self.maxstep):
            print(f'Step: {istep + 1}')
            new_xlist = []

            # Process each saddle point
            for saddle in xlist:
                # Generate one local minimum from each saddle point
                try:
                    new_min = local_optimization(saddle)
                except:
                    print(f"Failed to optimize from Saddle {saddle.id}")
                    continue

                if new_min.converged:
                    idm, isnew = self.update_minima(new_min)
                    new_min.id = idm
                    h = new_min.get_volume()*self.press/1602.176487 + new_min.get_potential_energy() - self.baseenergy
                    volume = new_min.get_volume()
                    self.G.add_node(idm, xname=f'M{idm}', e=h, volume=volume)
                    # Add edge with weight (saddle energy from which this minimum came)
                    saddle_energy = self.G.nodes[saddle.id]['e'] # Get energy of the connecting saddle
                    self.G.add_edge(saddle.id, idm, weight=saddle_energy) 
                    print(f"Added node: ID={idm}, Type=Minimum, Energy={h:.4f}, Volume={volume:.4f}")
                    print(f"Added edge: Saddle {saddle.id} -> Minimum {idm}")
                    self.save_graph()

                    # Generate one saddle point from the new minimum
                    try:
                        new_saddle = cal_saddle(new_min)
                    except:
                        print(f"Failed to calculate saddle for Minimum {new_min.id}")
                        continue

                    if new_saddle.converged:
                        ids, isnew = self.update_saddle(new_saddle)
                        new_saddle.id = ids
                        h = new_saddle.get_volume()*self.press/1602.176487 + new_saddle.get_potential_energy() - self.baseenergy
                        volume = new_saddle.get_volume()
                        self.G.add_node(ids, xname=f'S{ids}', e=h, volume=volume)
                        # Add edge with weight (saddle energy)
                        self.G.add_edge(idm, ids, weight=h) 
                        print(f"Added node: ID={ids}, Type=Saddle, Energy={h:.4f}, Volume={volume:.4f}")
                        print(f"Added edge: Minimum {idm} -> Saddle {ids}")
                        self.save_graph()
                        new_xlist.append(cp(new_saddle))

            # Update xlist for the next iteration
            xlist = cp(new_xlist)


            if not xlist:
                print("No new structures generated. Stopping the iteration.")
                break

            self.save_graph()

        # for istep in range(self.maxstep):
        #     print('step: ' + str(istep))
        #     tmplist = []
        #     # tmpids = []
        #     for i in range(len(xlist)):
        #         if istep > 0:
        #             optx = local_optimization(xlist[i])
        #             id_oldsaddle = xlist[i].id
        #             idm, isnew = self.update_minima(optx)
        #             optx.id = idm
        #             h = optx.get_volume()*self.press/1602.176487 + optx.get_potential_energy() - self.baseenergy
        #             volume = optx.get_volume()
        #             self.G.add_node(idm, xname='M'+str(idm), e=h, volume=volume)
        #             self.G.add_edge(id_oldsaddle, idm)
        #             print(f"Added node: ID={idm}, Type=Minimum, Energy={h:.4f}, Volume={volume:.4f}")
        #             print(f"Added edge: Saddle {id_oldsaddle} -> Minimum {idm}")
        #             self.save_graph()  
        #         else:
        #             optx = xlist[i]
        #             idm = optx.id
        #         for ip in range(self.popsize):
        #             try:
        #                 sadx = cal_saddle(optx)
        #             except:
        #                 print(f"Failed to calculate saddle for Minimum {optx.id}")
        #                 continue
        #             if sadx.converged:
        #                 ids, isnew = self.update_saddle(sadx)
        #                 sadx.id = ids
        #                 h = sadx.get_volume()*self.press/1602.176487 + sadx.get_potential_energy() - self.baseenergy
        #                 volume = sadx.get_volume()
        #                 self.G.add_node(ids, xname='S'+str(ids), e=h, volume=volume)
        #                 self.G.add_edge(idm, ids)
        #                 print(f"Added node: ID={ids}, Type=Saddle, Energy={h:.4f}, Volume={volume:.4f}")
        #                 print(f"Added edge: Minimum {idm} -> Saddle {ids}")
        #                 self.save_graph()  
        #                 tmplist.append(cp(sadx))
        #                 random.shuffle(tmplist)
        #                 # tmpids.append(ids)
                        

                        
        #     xlist = tmplist[1:10]
        #     self.save_graph()  

    
    def save_graph(self):
        joblib.dump(self.G, 'graph.pkl')
        nx.write_gml(self.G, 'graph.gml')
        nx.write_gexf(self.G, 'graph.gexf')
        joblib.dump(self.dij, 'dij.pkl')       

    def add_perturbation(self, structure):
        print("Adding perturbation to structure")
        """Add a small random perturbation to atomic positions."""
        # perturbed = structure.copy()
        # positions = perturbed.get_positions()
        # perturbation = np.random.uniform(-magnitude, magnitude, positions.shape)
        # perturbed.set_positions(positions + perturbation)
        # return perturbed
        atoms = cp(structure)
        # print energy
        print("Energy before perturbation: ", atoms.get_potential_energy())
        # calculate the jacobian for the tangent
        natom = len(atoms)
        vol   = atoms.get_volume()
        jacob = (vol/natom)**(1.0/3.0) * natom**0.5

        #######################################
        # set the initial mode randomly
        mode = np.zeros((len(atoms)+3,3))
        mode = vrand(mode)
        ##constrain 3 redundant freedoms
        mode[0]    *=0
        mode[-3,1:]*=0
        mode[-2,2] *=0
        ########################################
        #
        ## displace along the initial mode direction
        mode = vunit(mode)
        cellt = atoms.get_cell()+np.dot(atoms.get_cell(), mode[-3:]/jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + mode[:-3])
        print("Energy after perturbation: ", atoms.get_potential_energy())
        return atoms
    
    # def evo(self, istart):
    #     for istep in range(istart, self.maxstep):
    #         print('step: ' + str(istep))
    #         for i in range(self.num_initmin):
    #             for ip in range(itin.popsize):
    #                 xmode = joblib.load('xmode.' + str(i) + '.' + str(ip))
    #                 optx = joblib.load('optx.' + str(i) + '.' + str(ip))
    #                 sadx = joblib.load('calfile.' + str(i) + '.' + str(ip))
    #                 ids = self.update_saddle(sadx) 
    #                 h = sadx.get_volume()*self.press/1602.176487 + sadx.get_potential_energy() - self.baseenergy
    #                 volume = sadx.get_volume()
    #                 self.G.add_node(ids, xname='S'+str(ids), e=h, volume=volume)
                    


    # def read_caldata(self, calfile):
    #     minima = joblib.load(calfile)
    #     return minima
        
        
    def cal_fp(self, minima):
        lat = minima.get_cell()
        rxyz = minima.get_positions()
        types = self.types
        znucl = minima.get_atomic_numbers()
        fp = fplib2.get_fp(False, self.ntyp, self.natx, self.lmax, lat, rxyz, types, znucl, self.fpcutoff)
        return fp
    
        
    # def run(self):
    #     self.init_reac()

    
    def update_dij(self, id1, id2, fp_dist):
        if id1 > len(self.dij) or id2 > len(self.dij):
            dij_back = self.dij.copy()
            nlen = len(self.dij) * 2
            self.dij = np.zeros((nlen, nlen), float)
            self.dij[:][:] = 1000.
            np.fill_diagonal(self.dij, 0.0)
            self.dij[:len(dij_back)][:len(dij_back)] = dij_back
        self.dij[id1][id2] = fp_dist
        self.dij[id2][id1] = fp_dist

    def update_minima(self, minima):
        fpm = minima.get_fp()
        em = minima.get_potential_energy()
        isnew = True
        for x in self.db.select(ctyp='minima'):
            fpx = x.data['fp']
            fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fpm, fpx)
            ediff = np.abs(em - x.data['energy'])
            print ("fp_dist: ", fp_dist, ediff )
            if fp_dist < 0.005 and ediff < 0.001:
                print('minima already in database')
                idm = x.id
                isnew = False
                break
        if isnew:
            print('new minima')
            idm = self.db.write(minima, ctyp='minima', data={'fp': fpm, 'energy': em})
        
        for x in self.db.select(ctyp='minima'):
            if x.id != idm:
                fpx = x.data['fp']
                fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fpm, fpx)
                self.update_dij(idm, x.id, fp_dist)

        for x in self.db.select(ctyp='saddle'):
            fpx = x.data['fp']
            fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fpm, fpx)
            self.update_dij(idm, x.id, fp_dist)
        return idm, isnew
    
    def update_saddle(self, saddle):
        fps = saddle.get_fp()
        es = saddle.get_potential_energy()
        isnew = True
        for x in self.db.select(ctyp='saddle'):
            fpx = x.data['fp']
            fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fps, fpx)
            ediff = np.abs(es - x.data['energy'])
            print ("fp_dist saddle: ", fp_dist, ediff )
            if fp_dist < 0.005 and ediff < 0.001:
                print('saddle already in database')
                ids = x.id
                isnew = False
                break
        if isnew:
            print('new saddle')
            ids = self.db.write(saddle, ctyp='saddle', data={'fp': fps, 'energy': es})

        for x in self.db.select(ctyp='saddle'):
            if x.id != ids:    
                fpx = x.data['fp']
                fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fps, fpx)
                self.update_dij(ids, x.id, fp_dist)
        
        for x in self.db.select(ctyp='minima'):
            fpx = x.data['fp']
            fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fps, fpx)
            self.update_dij(ids, x.id, fp_dist)
        return ids, isnew
        

    def save_graph(self):
        joblib.dump(self.G, 'graph.pkl')
        nx.write_gml(self.G, 'graph.gml')
        nx.write_gexf(self.G, 'graph.gexf')
        joblib.dump(self.dij, 'dij.pkl')

    def add_perturbation(self, structure, magnitude=0.1):
        """Add a small random perturbation to atomic positions."""
        perturbed = structure.copy()
        positions = perturbed.get_positions()
        perturbation = np.random.uniform(-magnitude, magnitude, positions.shape)
        perturbed.set_positions(positions + perturbation)
        return perturbed

class PallasAtom(Atoms):
    def __init__(self, *args, **kwargs):
        # Initialize the Atom class
        super().__init__(*args, **kwargs)
        
        # Initialize any new attributes for the subclass
        self.natx = 200
        self.fpcutoff = 5.5
        self.lmax = 0
        self.fp = None
        self.converged = False
        self.id = None

    def get_fp(self):
        if self.fp is None:
            self.fp = self.cal_fp()
        return self.fp
    
    def cal_fp(self):
        # Example new function
        lat = self.get_cell()
        rxyz = self.get_positions()
        types = np.array([1,1,1,1,2,2,2,2])
        znucl = self.get_atomic_numbers()
        ntyp = 2
        fp = fplib2.get_fp(False, ntyp, self.natx, self.lmax, lat, rxyz, types, znucl, self.fpcutoff)
        self.fp = fp
        return fp

    def another_function(self):
        # Another new function
        print("This function could perform operations specific to EnhancedAtom.")


def main():
    pallas = Pallas()
    poscars = ['POSCAR1', 'POSCAR2']
    pallas.init_run(poscars)
    pallas.run()


def x2PAtom(xdb, x):
    xx = xdb.get_atoms(x.id)
    xxx = PallasAtom(xx)
    xxx.fp = x.data['fp']
    return xxx
        


# --- Start: Minimax Pathfinding Code from barrier.py ---

# 1. Union–Find with path-compression + union-by-rank
class UnionFind:
    __slots__ = ("p", "r")

    def __init__(self, nodes):
        self.p = {v: v for v in nodes}   # parent
        self.r = {v: 0 for v in nodes}   # rank

    def find(self, v):                   # iterative, path-compressed
        p = self.p
        while p[v] != v:
            p[v] = p[p[v]]
            v = p[v]
        return v

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True

# 2. Kruskal with early stop for minimax path based on edge weights
def minimax_path(G: nx.Graph, start, goal, wkey: str = "weight"):
    if start not in G or goal not in G:
        raise nx.NodeNotFound("Either source or target is not in G")
        
    uf   = UnionFind(G.nodes)
    adj  = {v: [] for v in G.nodes}          # adjacency in the partial MST
    edges = sorted(G.edges(data=True), key=lambda e: e[2].get(wkey, float('inf'))) # Use get with default

    for u, v, data in edges:
        # Check if weight exists, otherwise skip or handle as needed
        if wkey not in data:
            # print(f"Warning: Edge ({u}, {v}) missing weight attribute '{wkey}'. Skipping.")
            continue # Or assign a default weight like float('inf')?
            
        if uf.find(u) != uf.find(v): # Only add edge to adj if it connects components
            uf.union(u, v)
            adj[u].append(v)
            adj[v].append(u)

        # once the components touch, we have the minimax bottleneck
        if uf.find(start) == uf.find(goal):
            bottleneck = data[wkey]
            path = _restore_path(adj, start, goal)
            if path: # Ensure path was actually found
                return path, bottleneck
            else: # Should not happen if find(start)==find(goal) but safety check
                 break # Exit loop, path not found via BFS for some reason

    raise nx.NetworkXNoPath(f"No path found between {start} and {goal}")


def _restore_path(adj, s, t):
    """BFS in the partial MST to get the actual path."""
    q, prev = deque([s]), {s: None}
    visited = {s} # Keep track of visited nodes during BFS

    while q:
        cur = q.popleft()
        if cur == t: # Found target
             break
        
        # Ensure cur is in adj and adj[cur] is iterable
        if cur not in adj or not hasattr(adj[cur], '__iter__'):
             continue # Skip if cur has no neighbors defined in adj

        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                prev[nxt] = cur
                q.append(nxt)

    # Reconstruct path
    path = []
    curr = t
    while curr is not None:
        if curr not in prev and curr != s: # Path broken
             return None # Indicate path not found
        path.append(curr)
        curr = prev.get(curr) # Safely get predecessor

    if not path or path[-1] != s: # Path doesn't reach source
        return None
        
    path.reverse()
    return path


# 3. Minimax barrier based on node energy along the minimax edge-weight path
def minimax_barrier(G, start, goal, weight="weight", energy="energy"):
    """
    Finds the path between start and goal nodes that minimizes the maximum 
    edge weight (bottleneck) along the path using Kruskal's algorithm (via minimax_path).
    Then, it returns the maximum node energy encountered along this specific path.

    Args:
        G (nx.Graph): The graph.
        start: The starting node.
        goal: The target node.
        weight (str): The key for edge weights used in minimax_path.
        energy (str): The key for node energy attributes.

    Returns:
        tuple: (max_energy, path) where max_energy is the highest node energy 
               on the minimax path, and path is the list of nodes in the path.
        Raises nx.NetworkXNoPath if no path exists.
        Raises nx.NodeNotFound if start or goal node doesn't exist.
        Raises KeyError if nodes on the path lack the specified energy attribute.
    """
    try:
        path, bottleneck = minimax_path(G, start, goal, weight)
        
        # Check if path is valid before calculating max energy
        if not path:
             raise nx.NetworkXNoPath(f"Path reconstruction failed between {start} and {goal}")

        max_energy = -float('inf')
        for n in path:
             if n not in G.nodes:
                  raise nx.NodeNotFound(f"Node {n} from path not found in graph G")
             if energy not in G.nodes[n]:
                  raise KeyError(f"Node {n} does not have energy attribute '{energy}'")
             max_energy = max(max_energy, G.nodes[n][energy])
             
        return max_energy, path
        
    except nx.NetworkXNoPath:
         print(f"No path could be found between {start} and {goal}.")
         raise # Re-raise the exception
    except nx.NodeNotFound as e:
         print(f"Error: {e}")
         raise
    except KeyError as e:
         print(f"Error calculating barrier: {e}")
         raise


# --- End: Minimax Pathfinding Code ---


def listpath():
    # Load the saved graph
    G = joblib.load('graph.pkl')
    
    # Load the pallas.json database
    db = ase.db.connect('pallas.json')
    
    # Define start and end nodes (Assuming these are IDs in the graph/db)
    start_node = 1  # Example start node ID
    end_node = 11   # Example end node ID
    
    paths = [] # Initialize paths to empty list
    try:
        # Find the path with the minimum barrier using node energy 'e'
        barrier, path = minimax_barrier(G, start_node, end_node, weight='weight', energy='e')
        
        print(f"\nFound minimax barrier path between {start_node} and {end_node}:")
        print(f"  Barrier Energy = {barrier:.4f}")
        print(f"  Path Length = {len(path)}")
        print(f"  Path Nodes = {path}")

        paths = [path] # Store the single best path found

        # Create a folder for this path
        path_folder = f"minimax_barrier_path_{start_node}_to_{end_node}"
        os.makedirs(path_folder, exist_ok=True)
            
        # Process the nodes in the found path
        for node in path:
            if node in G.nodes:
                 node_data = G.nodes[node]
                 node_type = 'Minimum' if node_data.get('xname', '').startswith('M') else 'Saddle'
                 print(f"    Node {node} ({node_type}): Energy = {node_data.get('e', float('nan')):.4f}, Volume = {node_data.get('volume', float('nan')):.4f}")
                 
                 # Get the structure from the database
                 try:
                      structure_data = db.get_atoms(id=int(node)) # Ensure node is int for db query
                      if structure_data:
                           write(os.path.join(path_folder, f"{node}_POSCAR"), structure_data, format='vasp', direct=True)
                      else:
                           print(f"      Warning: Structure data not found in db for node {node}")
                 except Exception as e:
                      print(f"      Warning: Error retrieving structure for node {node} from db: {e}")
            else:
                 print(f"    Warning: Node {node} from path not found in graph G.")


    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError) as e:
        print(f"Could not find minimax barrier path between {start_node} and {end_node}: {e}")

    # --- Energy Profile Plotting (using the single best path) ---
    if paths: # If a path was successfully found
        types = np.array([1,1,1,1,2,2,2,2]) # TODO: Get types dynamically if needed
        ntyp = 2                            # TODO: Get ntyp dynamically if needed
        
        os.makedirs("path_energies", exist_ok=True)
        
        path = paths[0] # Get the single best path
        filename = f"path_energies/minimax_barrier_path_{start_node}_to_{end_node}_energy.txt"
        with open(filename, 'w') as f:
            f.write(f"#Minimax Barrier Path: {path}\n")
            f.write(f"#Barrier Energy: {barrier:.6f}\n")
            f.write("#Cumulative_FP_Distance Energy\n")
            
            cumulative_distance = 0.0
            for j, node in enumerate(path):
                try:
                    node_data = G.nodes[node]
                    energy = node_data['e']
                    
                    if j == 0:
                        f.write(f"{cumulative_distance:.6f} {energy:.6f}\n")
                        print(f"  Plot Point {j}: Dist=0.0, Energy={energy:.4f}")
                    else:
                        prev_node = path[j-1]
                        
                        # Safely get fingerprints from db
                        try:
                             prev_fp = db.get(id=int(prev_node)).data['fp']
                             curr_fp = db.get(id=int(node)).data['fp']
                             fp_dist = fplib2.get_fpdist(ntyp, types, prev_fp, curr_fp)
                             cumulative_distance += fp_dist
                             f.write(f"{cumulative_distance:.6f} {energy:.6f}\n")
                             print(f"  Plot Point {j}: Dist={cumulative_distance:.4f}, Energy={energy:.4f}")
                        except Exception as db_err:
                             print(f"    Warning: Could not get FP data for edge ({prev_node}, {node}) from db: {db_err}. Skipping distance calculation for this step.")
                             # Optionally write with NaN or previous distance if needed
                             f.write(f"{cumulative_distance:.6f} {energy:.6f} # FP Distance calculation failed\n")


                except Exception as node_err:
                     print(f"    Warning: Error processing node {node} for energy profile: {node_err}")

        print(f"\nEnergy profile for the minimax barrier path written to '{filename}'")

    # (Keep plotting code for multiple paths commented out for now)
    # # Create a directory to store all path files
    # os.makedirs("path_energies", exist_ok=True)
    # for i, path in enumerate(paths, 1):
    #     filename = f"path_energies/path_{i}_energy.txt"
    #     with open(filename, 'w') as f:
    #         f.write(f"#Path {i}\n")
    #         f.write("#FP_Distance Energy\n")
            
    #         cumulative_distance = 0.0
    #         for j, node in enumerate(path):
    #             node_data = G.nodes[node]
    #             energy = node_data['e']
                
    #             if j == 0:
    #                 f.write(f"{cumulative_distance:.6f} {energy:.6f}\n")
    #             else:
    #                 prev_node = path[j-1]
    #                 prev_fp = db.get(id=prev_node).data['fp']
    #                 curr_fp = db.get(id=node).data['fp']
    #                 fp_dist = fplib2.get_fpdist(ntyp, types, prev_fp, curr_fp)
    #                 cumulative_distance += fp_dist
    #                 f.write(f"{cumulative_distance:.6f} {energy:.6f}\n")
    # print("Energy profiles for all paths have been written to separate files in the 'path_energies' directory.")

def test():
    pp0 = read('POSCAR', format='vasp')
    print (pp0.numbers)

    pp = PallasAtom(pp0)

    print (pp.get_cell())
    write('px', pp, format='vasp')
    print (pp.get_atomic_numbers())
    print (pp.get_positions())
    print (pp.get_scaled_positions())
    print (pp.get_chemical_symbols())
    print (pp.get_masses())
    # print (pp.get_tags())
    # print (pp.get_velocities())
    t1 = time.time()
    fp1= pp.get_fp()
    t2 = time.time()
    # print (fp1)
    fp1 = pp.get_fp()
    t3 = time.time()
    print (t2-t1)
    print (t3-t2)
    print (np.shape(fp1))




    optatom = local_optimization(pp)
    print (optatom.get_potential_energy())
    print (optatom.get_forces())
    print (optatom.get_stress())
    print (optatom.get_cell()) 
    print (optatom.get_volume())
    # optatom = cal_saddle(pp)
    # print (optatom.get_cell())
    # print (optatom.get_fp())
    # fff = optatom.get_forces()
    # print (fff)
    # print (np.max(np.abs(fff)))
    # print (optatom.converged)

class GraphVisualizer:
    def __init__(self, graph):
        """
        Initializes the graph visualization with a given graph.
        
        :param graph: The graph to visualize (networkx.Graph)
        """
        self.graph = graph
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.pos = nx.spring_layout(graph)
        self.node_size = 500
        self.node_color = 'lightblue'
        self.edge_color = 'gray'

    def draw_graph(self):
        nx.draw_networkx_nodes(self.graph, self.pos, node_size=self.node_size, node_color=self.node_color, ax=self.ax)
        nx.draw_networkx_edges(self.graph, self.pos, width=1, edge_color=self.edge_color, ax=self.ax)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=12, font_weight='bold', ax=self.ax)
        plt.title("Dynamic Graph Visualization")

    def update(self, frame):
        #Updates the graph for a given frame in the animation. 
        self.ax.clear()
        
        self.graph.add_node(frame)
        if frame > 0:
            self.graph.add_edge(frame-1, frame)
        
        self.draw_graph()
        self.ax.set_title(f"Frame {frame}", fontsize=16)

    def animate(self, frames=10, interval=1000):
        ani = FuncAnimation(self.fig, self.update, frames=frames, interval=interval, repeat=False)
        plt.show()



if __name__ == "__main__":
    # test()
    main()
    # listpath()
    
