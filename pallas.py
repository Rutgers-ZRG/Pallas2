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

from nequipcal import local_optimization, cal_saddle

class Pallas(object):
    def __init__(self):
        self.init_minima = []
        self.ipso = []
        self.all_minima = []
        self.all_saddle = []
        self.fpcutoff = 5.5
        self.lmax = 0
        self.natx = 200
        self.ntyp = 2
        self.types = np.array([1,1,1,1,2,2,2,2])
        self.dij = np.zeros((10000, 10000), float)
        self.baseenergy = 0.0
        self.G = nx.Graph()
        self.press = 0.0
        self.maxstep = 50
        self.popsize = 10
        

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
                    self.G.add_edge(idm, ids)
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
                    self.G.add_edge(saddle.id, idm)
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
                        self.G.add_edge(idm, ids)
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
        


def find_path(graph, start, end):
    """
    Find the minimax path between two given nodes in the Graph.
    This function extracts the lowest-barrier paths with the least number of intermediate transition states.
    
    Args:
    graph (networkx.Graph): The graph representing the energy landscape.
    
    Returns:
    list: The minimax path between the two lowest energy minima.
    """
    # Find the two lowest energy minima
    # minima = [node for node, data in graph.nodes(data=True) if data['xname'].startswith('M')]
    # minima.sort(key=lambda x: graph.nodes[x]['e'])
    # start, end = minima[:2]
    
    def minimax_cost(path):
        """Calculate the minimax cost of a path."""
        return max(graph.nodes[node]['e'] for node in path)
    
    def dfs_paths(start, end, path=None):
        """Depth-first search to find all paths."""
        if path is None:
            path = [start]
        if start == end:
            yield path
        for neighbor in graph.neighbors(start):
            if neighbor not in path:
                yield from dfs_paths(neighbor, end, path + [neighbor])
    
    # Find all paths and sort them by minimax cost and length
    all_paths = list(dfs_paths(start, end))
    all_paths.sort(key=lambda p: (minimax_cost(p), len(p)))
    
    # Return the path with the lowest minimax cost and least intermediate states
    # return all_paths[0] if all_paths else None
    return all_paths



def listpath():
    # Load the saved graph
    G = joblib.load('graph.pkl')
    
    # Load the pallas.json database
    db = ase.db.connect('pallas.json')
    
    start = 1
    end = 11
    
    # Find all paths between the two minima
    paths = find_path(G, start, end)
    
    if paths:
        print(f"Found {len(paths)} paths between {start} and {end}:")
        for i, path in enumerate(paths, 1):
            minimax_cost = max(G.nodes[node]['e'] for node in path)
            print(f"\nPath {i}: Minimax cost = {minimax_cost:.4f}, Length = {len(path)}")
            
            # Create a folder for this path
            path_folder = f"path_{i}"
            os.makedirs(path_folder, exist_ok=True)
            
            for node in path:
                node_data = G.nodes[node]
                node_type = 'Minimum' if node_data['xname'].startswith('M') else 'Saddle'
                print(f"  Node {node} ({node_type}): Energy = {node_data['e']:.4f}, Volume = {node_data['volume']:.4f}")
                
                # Get the structure from the database
                structure_data = db.get_atoms(node)
                if structure_data:
                    write(os.path.join(path_folder, f"{node}_POSCAR"), structure_data, format='vasp', direct=True)
                else:
                    print(f"    Warning: Structure data not found for node {node}")

    else:
        print(f"No paths found between {start} and {end}.")
    
    types = np.array([1,1,1,1,2,2,2,2])
    ntyp = 2




    # Create a directory to store all path files
    os.makedirs("path_energies", exist_ok=True)

    for i, path in enumerate(paths, 1):
        filename = f"path_energies/path_{i}_energy.txt"
        with open(filename, 'w') as f:
            f.write(f"#Path {i}\n")
            f.write("#FP_Distance Energy\n")
            
            cumulative_distance = 0.0
            for j, node in enumerate(path):
                node_data = G.nodes[node]
                energy = node_data['e']
                
                if j == 0:
                    f.write(f"{cumulative_distance:.6f} {energy:.6f}\n")
                else:
                    prev_node = path[j-1]
                    prev_fp = db.get(id=prev_node).data['fp']
                    curr_fp = db.get(id=node).data['fp']
                    fp_dist = fplib2.get_fpdist(ntyp, types, prev_fp, curr_fp)
                    cumulative_distance += fp_dist
                    f.write(f"{cumulative_distance:.6f} {energy:.6f}\n")

    print("Energy profiles for all paths have been written to separate files in the 'path_energies' directory.")
    # with open('pathenergy.txt', 'w') as f:
    #     for i, path in enumerate(paths, 1):
    #         f.write(f"#Path {i}\n")
    #         f.write("#FP_Distance_Energy\n")
            
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
            
    #         f.write("\n")  # Add a blank line between paths
    
    # print("Energy profiles for all paths have been written to pathenergy.txt")

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
    
