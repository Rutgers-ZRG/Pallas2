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
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import libfp
import time
from ase.units import GPa

from ase.optimize import FIRE
# from ase.constraints import StrainFilter, UnitCellFilter
from ase.filters import FrechetCellFilter

from xcal import XCalculator

from zfunc import local_optimization, cal_saddle, vunit, vrand

class Pallas(object):
    def __init__(self):
        self.init_minima = []
        self.ipso = []
        self.all_minima = []
        self.all_saddle = []
        self.fpcutoff = 5.5
        self.lmax = 0
        self.natx = 200
        self.ntyp = None
        self.znucl = None
        self.types = None # np.array([1,1,1,1,2,2,2,2])
        self.dij = np.zeros((10000, 10000), float)
        self.baseenergy = 0.0
        self.G = nx.Graph()
        self.press = 0.0
        self.maxstep = 50
        self.popsize = 10
        
        # PSO parameters
        self.pbestx = []
        self.pbesty = []
        self.gbestx = None
        self.gbesty = None
        self.pdistx = []
        self.pdisty = []
        self.bestdist = float('inf')
        self.bestmdist = float('inf')
        self.ediff = 0.001
        self.dist_threshold = 0.01
        self.velocity_weight = 0.9
        self.c1 = 2.0  # Personal best weight
        self.c2 = 1.5  # Global best weight
        

    def init_run(self, flist):
        if len(flist) < 2:
            raise ValueError("At least two structures (reactant and product) are required for PSO")
            
        self.db = ase.db.connect('pallas.db')
        self.dij[:][:] = float('inf')
        np.fill_diagonal(self.dij, 0.0)
        
        # Ensure types and znucl are set properly before we create PallasAtom objects
        if self.types is None:
            print("Warning: self.types is None in init_run! This will cause segfaults.")
        
        if self.znucl is None:
            print("Warning: self.znucl is empty in init_run! This may cause problems.")
        
        self.init_minima = self.read_init_structure(flist)
        self.num_init_min = len(self.init_minima)


        # Use the first structure as reactant and the second as product
        self.reactant = self.init_minima[0]
        self.product = self.init_minima[1]
        
        # Double-check the types are set properly
        for struct in self.init_minima:
            if struct.types is None:
                print(f"Warning: types not properly set in structure after initialization")
                struct.types = self.types

    def read_init_structure(self, flist):
        init_minima = []
        for xf in flist:
            x = read(xf, format='vasp')
            x = PallasAtom(x)
            x.fpcutoff = self.fpcutoff
            x.types = self.types
            x.znucl = self.znucl
            init_minima.append(cp(x))
        return init_minima
    
    def run_pso(self):
        """Main PSO loop to optimize saddle points between reactant and product.
        
        This method implements Particle Swarm Optimization (PSO) to find reaction pathways
        between reactant and product structures. The algorithm works by:
        
        1. Starting with populations of perturbed reactant and product structures
        2. Finding saddle points and local minima from each structure
        3. Measuring fingerprint distances between structures from opposite sides
        4. Updating particle velocities based on personal and global best positions
        5. Searching for connections between reactant and product sides
        6. Continuing until a pathway is found or maximum iterations are reached
        
        The PSO approach has several advantages:
        - Bidirectional search from both reactant and product sides simultaneously
        - Optimization for minimum fingerprint distance and energy barriers
        - "Swarm intelligence" where particles share information about best pathways
        
        Returns:
            networkx.Graph: The final graph representing the energy landscape
        """
        print("Starting PSO-based path optimization")
        
        # Optimize the reactant and product structures
        print("Optimizing reactant structure")
        react_opt = local_optimization(self.reactant)
        # print (react_opt.calc)
        react_id, _ = self.update_minima(react_opt)
        react_opt.id = react_id
        
        print("Optimizing product structure")
        prod_opt = local_optimization(self.product)
        prod_id, _ = self.update_minima(prod_opt)
        prod_opt.id = prod_id
        
        # Set base energy to reactant energy
        self.baseenergy = react_opt.get_volume()*self.press*GPa + react_opt.get_potential_energy()
        
        # Add reactant and product to graph
        h_react = 0.0
        vol_react = react_opt.get_volume()
        self.G.add_node(react_id, xname=f'M{react_id}', e=h_react, volume=vol_react)
        
        h_prod = prod_opt.get_volume()*self.press*GPa + prod_opt.get_potential_energy() - self.baseenergy
        vol_prod = prod_opt.get_volume()
        self.G.add_node(prod_id, xname=f'M{prod_id}', e=h_prod, volume=vol_prod)
        
        print(f"Added reactant: ID={react_id}, Energy={h_react:.4f}")
        print(f"Added product: ID={prod_id}, Energy={h_prod:.4f}")
        
        # Initialize particles
        reactant_particles = []
        product_particles = []
        reactant_velocities = []
        product_velocities = []
        
        print(f"Initializing {self.popsize} particles for PSO")
        
        # Create initial population of perturbed structures and random velocities
        for i in range(self.popsize):
            # Create perturbed copies for reactant side
            perturbed_reactant = self.add_perturbation(react_opt)
            reactant_particles.append(perturbed_reactant)
            
            # Create random velocity for reactant
            reactant_vel = self.gen_random_velocity(perturbed_reactant)
            reactant_velocities.append(reactant_vel)
            
            # Create perturbed copies for product side
            perturbed_product = self.add_perturbation(prod_opt)
            product_particles.append(perturbed_product)
            
            # Create random velocity for product
            product_vel = self.gen_random_velocity(perturbed_product)
            product_velocities.append(product_vel)
            
            # Initialize personal best distances as infinity
            self.pdistx.append(float('inf'))
            self.pdisty.append(float('inf'))
        
        # Initialize best structures
        self.pbestx = cp(reactant_particles)
        self.pbesty = cp(product_particles)
        
        # Main PSO iteration loop
        for step in range(self.maxstep):
            print(f"PSO iteration {step+1}/{self.maxstep}")
            
            reactant_saddles = []
            product_saddles = []
            
            # Calculate saddle points for each particle
            for i in range(self.popsize):
                print(f"Processing particle {i+1}/{self.popsize}")
                
                # Calculate saddle point from reactant side
                # try:
                reactant_saddle = self.calculate_saddle_with_velocity(
                    reactant_particles[i], 
                    reactant_velocities[i]
                )
                reactant_saddles.append(reactant_saddle)
                
                # Update the saddle in database and graph
                sadr_id, _ = self.update_saddle(reactant_saddle)
                reactant_saddle.id = sadr_id
                h = reactant_saddle.get_volume()*self.press*GPa + reactant_saddle.get_potential_energy() - self.baseenergy
                volume = reactant_saddle.get_volume()
                
                # Add node and edge to graph
                self.G.add_node(sadr_id, xname=f'S{sadr_id}', e=h, volume=volume)
                # Calculate edge weight (max energy) and add with fingerprint distance
                react_energy = self.G.nodes[react_id]['e']
                edge_weight = max(react_energy, h)
                # Use previous fingerprint distance calculation or calculate new one
                fp_r = react_opt.get_fp()
                fp_s = reactant_saddle.get_fp()
                if self.types is not None:
                    try:
                        fp_dist = libfp.get_fp_dist(fp_r, fp_s, self.types)
                    except Exception as e:
                        print(f"Error calculating fp_dist for edge: {e}")
                        fp_dist = float('inf')
                else:
                    fp_dist = float('inf')
                self.G.add_edge(react_id, sadr_id, weight=edge_weight, dist=fp_dist)
                print(f"Added saddle from reactant: ID={sadr_id}, Energy={h:.4f}")
                
                # Find local minimum from this saddle
                sadxcal = self.xcal(reactant_saddle, prod_opt.get_fp())
                new_min_r = local_optimization(sadxcal)
                min_r_id, _ = self.update_minima(new_min_r)
                new_min_r.id = min_r_id
                h = new_min_r.get_volume()*self.press*GPa + new_min_r.get_potential_energy() - self.baseenergy
                volume = new_min_r.get_volume()
                
                # Update graph with new minimum
                self.G.add_node(min_r_id, xname=f'M{min_r_id}', e=h, volume=volume)
                # Calculate edge weight and add with fingerprint distance
                saddle_energy = self.G.nodes[sadr_id]['e']
                edge_weight = max(saddle_energy, h)
                # Use previous fingerprint distance calculation or calculate new one
                fp_s = reactant_saddle.get_fp()
                fp_m = new_min_r.get_fp()
                if self.types is not None:
                    try:
                        fp_dist = libfp.get_fp_dist(fp_s, fp_m, self.types)
                    except Exception as e:
                        print(f"Error calculating fp_dist for edge: {e}")
                        fp_dist = float('inf')
                else:
                    fp_dist = float('inf')
                self.G.add_edge(sadr_id, min_r_id, weight=edge_weight, dist=fp_dist)
                print(f"Added minimum from reactant saddle: ID={min_r_id}, Energy={h:.4f}")
                
                # Update particle position
                reactant_particles[i] = cp(new_min_r)

                # except Exception as e:
                #     print(f"Error calculating reactant saddle for particle {i}: {e}")
                
                # Calculate saddle point from product side
                try:
                    product_saddle = self.calculate_saddle_with_velocity(
                        product_particles[i], 
                        product_velocities[i]
                    )
                    product_saddles.append(product_saddle)
                    
                    # Update the saddle in database and graph
                    sadp_id, _ = self.update_saddle(product_saddle)
                    product_saddle.id = sadp_id
                    h = product_saddle.get_volume()*self.press*GPa + product_saddle.get_potential_energy() - self.baseenergy
                    volume = product_saddle.get_volume()
                    
                    # Add node and edge to graph
                    self.G.add_node(sadp_id, xname=f'S{sadp_id}', e=h, volume=volume)
                    # Calculate edge weight (max energy) and add with fingerprint distance
                    prod_energy = self.G.nodes[prod_id]['e']
                    edge_weight = max(prod_energy, h)
                    # Use previous fingerprint distance calculation or calculate new one
                    fp_p = prod_opt.get_fp()
                    fp_s = product_saddle.get_fp()
                    if self.types is not None:
                        try:
                            fp_dist = libfp.get_fp_dist(fp_p, fp_s, self.types)
                        except Exception as e:
                            print(f"Error calculating fp_dist for edge: {e}")
                            fp_dist = float('inf')
                    else:
                        fp_dist = float('inf')
                    self.G.add_edge(prod_id, sadp_id, weight=edge_weight, dist=fp_dist)
                    print(f"Added saddle from product: ID={sadp_id}, Energy={h:.4f}")
                    
                    # Find local minimum from this saddle
                    sadxcal = self.xcal(product_saddle, react_opt.get_fp())
                    new_min_p = local_optimization(sadxcal)
                    min_p_id, _ = self.update_minima(new_min_p)
                    new_min_p.id = min_p_id
                    h = new_min_p.get_volume()*self.press*GPa + new_min_p.get_potential_energy() - self.baseenergy
                    volume = new_min_p.get_volume()
                    
                    # Update graph with new minimum
                    self.G.add_node(min_p_id, xname=f'M{min_p_id}', e=h, volume=volume)
                    # Calculate edge weight and add with fingerprint distance
                    saddle_energy = self.G.nodes[sadp_id]['e']
                    edge_weight = max(saddle_energy, h)
                    # Use previous fingerprint distance calculation or calculate new one
                    fp_s = product_saddle.get_fp()
                    fp_m = new_min_p.get_fp()
                    if self.types is not None:
                        try:
                            fp_dist = libfp.get_fp_dist(fp_s, fp_m, self.types)
                        except Exception as e:
                            print(f"Error calculating fp_dist for edge: {e}")
                            fp_dist = float('inf')
                    else:
                        fp_dist = float('inf')
                    self.G.add_edge(sadp_id, min_p_id, weight=edge_weight, dist=fp_dist)
                    print(f"Added minimum from product saddle: ID={min_p_id}, Energy={h:.4f}")
                    
                    # Update particle position
                    product_particles[i] = cp(new_min_p)
                except Exception as e:
                    print(f"Error calculating product saddle for particle {i}: {e}")
            
            # Check fingerprint distances between all minima pairs
            connection_found = False
            print("Checking for connections between reactant and product sides")
            
            # Get all minima from reactant side
            minima_from_reactant = []
            for p in reactant_particles:
                minima_from_reactant.append(p)
            
            # Get all minima from product side
            minima_from_product = []
            for p in product_particles:
                minima_from_product.append(p)
            
            # Find closest pairs based on fingerprint distance
            min_dist = float('inf')
            best_pair = None
            
            for i, min_r in enumerate(minima_from_reactant):
                for j, min_p in enumerate(minima_from_product):
                    # Calculate fingerprint distance
                    fp_r = min_r.get_fp()
                    fp_p = min_p.get_fp()
                    
                    # Make sure types is not None before calling the function
                    if self.types is None:
                        print("Warning: self.types is None in fingerprint comparison!")
                        continue
                        
                    try:
                        # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fp_r, fp_p)
                        fp_dist = libfp.get_fp_dist(fp_r, fp_p, self.types)
                        energy_diff = abs(min_r.get_potential_energy() - min_p.get_potential_energy())
                        
                        # Only print if distance is below a threshold or it meets a significant condition
                        if fp_dist < 0.1:  # Only print "interesting" distances
                            print(f"FP distance between minima {min_r.id} and {min_p.id}: {fp_dist:.5f}, energy diff: {energy_diff:.5f}")
                        
                        # If distance is below threshold and energy difference is small, add edge
                        if fp_dist < self.dist_threshold and energy_diff < self.ediff:
                            print(f"Connection found between minima {min_r.id} and {min_p.id}!")
                            # Calculate edge weight (max energy)
                            min_r_energy = self.G.nodes[min_r.id]['e'] if min_r.id in self.G.nodes else min_r.get_volume()*self.press*GPa + min_r.get_potential_energy() - self.baseenergy
                            min_p_energy = self.G.nodes[min_p.id]['e'] if min_p.id in self.G.nodes else min_p.get_volume()*self.press*GPa + min_p.get_potential_energy() - self.baseenergy
                            edge_weight = max(min_r_energy, min_p_energy)
                            self.G.add_edge(min_r.id, min_p.id, weight=edge_weight, dist=fp_dist)
                            connection_found = True
                        
                        # Update minimum distance
                        if fp_dist < min_dist:
                            min_dist = fp_dist
                            best_pair = (i, j)
                    except Exception as e:
                        print(f"Error calculating distance between minima {min_r.id} and {min_p.id}: {e}")
                        continue
            
            # Update global best if new minimum distance is found
            if min_dist < self.bestdist:
                self.bestdist = min_dist
                self.gbestx = cp(minima_from_reactant[best_pair[0]])
                self.gbesty = cp(minima_from_product[best_pair[1]])
                print(f"New global best distance: {min_dist:.5f} between minima {self.gbestx.id} and {self.gbesty.id}")
            
            # Update personal bests
            for i, min_r in enumerate(minima_from_reactant):
                # Find minimum distance to any product-side minimum
                best_dist_for_r = float('inf')
                for min_p in minima_from_product:
                    fp_r = min_r.get_fp()
                    fp_p = min_p.get_fp()
                    
                    # Make sure types is not None before calling the function
                    if self.types is None:
                        print("Warning: self.types is None in personal best update!")
                        continue
                    
                    try:
                        # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fp_r, fp_p)
                        fp_dist = libfp.get_fp_dist(fp_r, fp_p, self.types)
                        if fp_dist < best_dist_for_r:
                            best_dist_for_r = fp_dist
                    except Exception as e:
                        print(f"Error calculating distance for personal best update: {e}")
                        continue
                
                # Update personal best if improved
                if best_dist_for_r < self.pdistx[i]:
                    self.pdistx[i] = best_dist_for_r
                    self.pbestx[i] = cp(min_r)
                    # Only print if significant improvement (e.g., >1% better)
                    if i == 0 or best_dist_for_r < 0.9 * min(self.pdistx[:i]):
                        print(f"Updated personal best for reactant particle {i}: {best_dist_for_r:.5f}")
            
            for j, min_p in enumerate(minima_from_product):
                # Find minimum distance to any reactant-side minimum
                best_dist_for_p = float('inf')
                for min_r in minima_from_reactant:
                    fp_r = min_r.get_fp()
                    fp_p = min_p.get_fp()
                    
                    # Make sure types is not None before calling the function
                    if self.types is None:
                        print("Warning: self.types is None in personal best update!")
                        continue
                    
                    try:
                        # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fp_r, fp_p)
                        fp_dist = libfp.get_fp_dist(fp_r, fp_p, self.types)
                        if fp_dist < best_dist_for_p:
                            best_dist_for_p = fp_dist
                    except Exception as e:
                        print(f"Error calculating distance for personal best update: {e}")
                        continue
                
                # Update personal best if improved
                if best_dist_for_p < self.pdisty[j]:
                    self.pdisty[j] = best_dist_for_p
                    self.pbesty[j] = cp(min_p)
                    # Only print if significant improvement (e.g., >1% better)
                    if j == 0 or best_dist_for_p < 0.9 * min(self.pdisty[:j]):
                        print(f"Updated personal best for product particle {j}: {best_dist_for_p:.5f}")
            
            # Update velocities for next iteration
            for i in range(self.popsize):
                # Update reactant velocity
                w = self.velocity_weight - 0.5 * step / self.maxstep  # Linearly decreasing weight
                r1, r2 = np.random.rand(2)
                
                # Get velocity components
                v_pbest_r = self.get_velocity_component(reactant_particles[i], self.pbestx[i])
                v_gbest_r = self.get_velocity_component(reactant_particles[i], self.gbestx)
                
                # Update velocity
                reactant_velocities[i] = w * reactant_velocities[i] + \
                                        self.c1 * r1 * v_pbest_r + \
                                        self.c2 * r2 * v_gbest_r
                
                # Update product velocity
                r1, r2 = np.random.rand(2)
                
                # Get velocity components
                v_pbest_p = self.get_velocity_component(product_particles[i], self.pbesty[i])
                v_gbest_p = self.get_velocity_component(product_particles[i], self.gbesty)
                
                # Update velocity
                product_velocities[i] = w * product_velocities[i] + \
                                       self.c1 * r1 * v_pbest_p + \
                                       self.c2 * r2 * v_gbest_p
            
            # Check for termination
            if connection_found:
                print("Connection found between reactant and product!")
                # Try to find path
                try:
                    paths = find_path(self.G, react_id, prod_id)
                    if paths:
                        print(f"Found {len(paths)} possible paths from reactant to product")
                        best_path = paths[0]  # First path is the one with lowest energy barrier
                        print("Best path:", best_path)
                        # Optional: Can stop if a good path is found
                        # break
                except Exception as e:
                    print(f"Error finding path: {e}")
            
            # Save graph after each iteration
            self.save_graph()
            
            # Check if max iterations reached
            if step == self.maxstep - 1:
                print("Maximum iterations reached")
        
        print("PSO optimization complete")
        return self.G
    
    def gen_random_velocity(self, structure):
        """Generate a random initial velocity for PSO."""
        natom = len(structure)
        mode = np.zeros((natom + 3, 3))
        mode = vrand(mode)
        # Constrain redundant freedoms
        mode[0] *= 0
        mode[-3, 1:] *= 0
        mode[-2, 2] *= 0
        # Normalize
        mode = vunit(mode)
        return mode
    
    def calculate_saddle_with_velocity(self, structure, velocity):
        """Calculate saddle point using the given velocity as an initial direction."""
        # Make a copy of the structure to avoid modifying the original
        atoms = cp(structure)
        
        # Apply the velocity to get initial displacement
        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol/natom)**(1.0/3.0) * natom**0.5
        
        # Displace along the velocity direction
        velocity = vunit(velocity)
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), velocity[-3:]/jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + velocity[:-3])
        
        # Calculate saddle point
        saddle = cal_saddle(atoms)
        return saddle
    
    def get_velocity_component(self, current, target):
        """Get velocity component pointing from current to target structure."""
        if current is None or target is None:
            # Return random velocity if either structure is None
            return self.gen_random_velocity(current if current is not None else target)
        
        # Get positions and cell difference
        natom = len(current)
        vol = current.get_volume()
        jacob = (vol/natom)**(1.0/3.0) * natom**0.5
        
        # Initialize velocity
        velocity = np.zeros((natom + 3, 3))
        
        # Position component
        pos_diff = target.get_positions() - current.get_positions()
        velocity[:natom] = pos_diff
        
        # Cell component
        cell_diff = target.get_cell() - current.get_cell()
        velocity[-3:] = cell_diff / jacob
        
        # Normalize
        velocity = vunit(velocity)
        return velocity
        
    # def run(self):
    #     """Original run method - kept for backward compatibility."""
    #     xlist = []

    #     # Iterate through initial minima
    #     for i in range(self.num_init_min):
    #         # Optimize the initial structure
    #         optx = local_optimization(self.init_minima[i])
    #         # Update minima and get new ID
    #         idm, isnew = self.update_minima(optx)
    #         optx.id = idm
            
    #         # Calculate energy (h) relative to base energy
    #         if i == 0:
    #             # Set base energy for the first minimum
    #             self.baseenergy = optx.get_volume()*self.press/1602.176487 + optx.get_potential_energy()
    #             h = 0.0
    #         else:
    #             # Calculate relative energy for subsequent minima
    #             h = optx.get_volume()*self.press/1602.176487 + optx.get_potential_energy() - self.baseenergy
            
    #         # Get volume of the optimized structure
    #         volume = optx.get_volume()
            
    #         # Add node to the graph
    #         self.G.add_node(idm, xname='M'+str(idm), e=h, volume=volume)
            
    #         # Print information about the added node
    #         print(f"Added node: ID={idm}, Type=Minimum, Energy={h:.4f}, Volume={volume:.4f}")
            
    #         # Save the updated graph
    #         self.save_graph()  

    #         # For each initial minimum, perform popsize number of saddle point optimizations
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
    #                 xlist.append(cp(sadx))

    #     # Iterate until reaching maxstep
    #     for istep in range(self.maxstep):
    #         print(f'Step: {istep + 1}')
    #         new_xlist = []

    #         # Process each saddle point
    #         for saddle in xlist:
    #             # Generate one local minimum from each saddle point
    #             try:
    #                 new_min = local_optimization(saddle)
    #             except:
    #                 print(f"Failed to optimize from Saddle {saddle.id}")
    #                 continue

    #             if new_min.converged:
    #                 idm, isnew = self.update_minima(new_min)
    #                 new_min.id = idm
    #                 h = new_min.get_volume()*self.press/1602.176487 + new_min.get_potential_energy() - self.baseenergy
    #                 volume = new_min.get_volume()
    #                 self.G.add_node(idm, xname=f'M{idm}', e=h, volume=volume)
    #                 self.G.add_edge(saddle.id, idm)
    #                 print(f"Added node: ID={idm}, Type=Minimum, Energy={h:.4f}, Volume={volume:.4f}")
    #                 print(f"Added edge: Saddle {saddle.id} -> Minimum {idm}")
    #                 self.save_graph()

    #                 # Generate one saddle point from the new minimum
    #                 try:
    #                     new_saddle = cal_saddle(new_min)
    #                 except:
    #                     print(f"Failed to calculate saddle for Minimum {new_min.id}")
    #                     continue

    #                 if new_saddle.converged:
    #                     ids, isnew = self.update_saddle(new_saddle)
    #                     new_saddle.id = ids
    #                     h = new_saddle.get_volume()*self.press/1602.176487 + new_saddle.get_potential_energy() - self.baseenergy
    #                     volume = new_saddle.get_volume()
    #                     self.G.add_node(ids, xname=f'S{ids}', e=h, volume=volume)
    #                     self.G.add_edge(idm, ids)
    #                     print(f"Added node: ID={ids}, Type=Saddle, Energy={h:.4f}, Volume={volume:.4f}")
    #                     print(f"Added edge: Minimum {idm} -> Saddle {ids}")
    #                     self.save_graph()
    #                     new_xlist.append(cp(new_saddle))

    #         # Update xlist for the next iteration
    #         xlist = cp(new_xlist)

    #         if not xlist:
    #             print("No new structures generated. Stopping the iteration.")
    #             break

    #         self.save_graph()
    
    def save_graph(self):
        joblib.dump(self.G, 'graph.pkl')
        nx.write_gml(self.G, 'graph.gml')
        nx.write_gexf(self.G, 'graph.gexf')
        joblib.dump(self.dij, 'dij.pkl')       

    def xcal(self, structure, fp0):
        atoms = cp(structure)
        calc = XCalculator(
            parallel=False,
            atoms = atoms,
            znucl = self.znucl,
            fp0 = fp0,
            cutoff = self.fpcutoff,
            contract=False, 
            lmax = self.lmax,
            nx=self.natx,
            ntyp=self.ntyp)
        atoms.calc = calc
        af = FrechetCellFilter(atoms)
        opt = FIRE(af, maxstep=0.1, logfile='xcal.log')
        opt.run(fmax=0.01, steps=60)
        return atoms

    def add_perturbation(self, structure):

        
        # """Add a small random perturbation to atomic positions."""
        atoms = cp(structure)
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
        
    def cal_fp(self, structure):
        lat = structure.cell[:]
        rxyz = structure.get_positions()
        types = self.types
        znucl = self.znucl
        cell = (lat, rxyz, types, znucl)
        fp = libfp.get_lfp(cell, cutoff=self.fpcutoff, log=False, natx = self.natx, orbital='s')
        # fp = fplib2.get_fp(False, self.ntyp, self.natx, self.lmax, lat, rxyz, types, znucl, self.fpcutoff)
        return fp
    
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
            fpx = np.array(x.data['fp'])
            # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fpm, fpx)
            # Make sure types is not None before calling the function
            if self.types is None:
                print("Warning: self.types is None in update_minima!")
                continue
            fp_dist = libfp.get_fp_dist(fpm, fpx, self.types)
            ediff = np.abs(em - x.data['energy'])
            if fp_dist < 0.005 and ediff < 0.001:
                idm = x.id
                isnew = False
                break
        if isnew:
            fpm_serializable = fpm.tolist() if hasattr(fpm, 'tolist') else fpm
            idm = self.db.write(minima, ctyp='minima', data={'fp': fpm_serializable, 'energy': float(em)})
        
        for x in self.db.select(ctyp='minima'):
            if x.id != idm:
                fpx = np.array(x.data['fp'])
                # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fpm, fpx)
                # Make sure types is not None before calling the function
                if self.types is None:
                    print("Warning: self.types is None in update_minima loop!")
                    continue
                fp_dist = libfp.get_fp_dist(fpm, fpx, self.types)
                self.update_dij(idm, x.id, fp_dist)

        for x in self.db.select(ctyp='saddle'):
            fpx = np.array(x.data['fp'])
            # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fpm, fpx)
            # Make sure types is not None before calling the function
            if self.types is None:
                print("Warning: self.types is None in update_minima saddle loop!")
                continue
            fp_dist = libfp.get_fp_dist(fpm, fpx, self.types)
            self.update_dij(idm, x.id, fp_dist)
        return idm, isnew
    
    def update_saddle(self, saddle):
        fps = saddle.get_fp()
        es = saddle.get_potential_energy()
        isnew = True
        for x in self.db.select(ctyp='saddle'):
            fpx = np.array(x.data['fp'])
            # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fps, fpx)
            # Make sure types is not None before calling the function
            if self.types is None:
                print("Warning: self.types is None in update_saddle!")
                continue
            fp_dist = libfp.get_fp_dist(fps, fpx, self.types)
            ediff = np.abs(es - x.data['energy'])
            if fp_dist < 0.005 and ediff < 0.001:
                ids = x.id
                isnew = False
                break
        if isnew:
            fps_serializable = fps.tolist() if hasattr(fps, 'tolist') else fps
            ids = self.db.write(saddle, ctyp='saddle', data={'fp': fps_serializable, 'energy': float(es)})

        for x in self.db.select(ctyp='saddle'):
            if x.id != ids:    
                fpx = np.array(x.data['fp'])
                # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fps, fpx)
                # Make sure types is not None before calling the function
                if self.types is None:
                    print("Warning: self.types is None in update_saddle loop!")
                    continue
                fp_dist = libfp.get_fp_dist(fps, fpx, self.types)
                self.update_dij(ids, x.id, fp_dist)
        
        for x in self.db.select(ctyp='minima'):
            fpx = np.array(x.data['fp'])
            # fp_dist = fplib2.get_fpdist(self.ntyp, self.types, fps, fpx)
            # Make sure types is not None before calling the function
            if self.types is None:
                print("Warning: self.types is None in update_saddle minima loop!")
                continue
            fp_dist = libfp.get_fp_dist(fps, fpx, self.types)
            self.update_dij(ids, x.id, fp_dist)
        return ids, isnew

class PallasAtom(Atoms):
    def __init__(self, *args, **kwargs):
        # Initialize the Atom class
        super().__init__(*args, **kwargs)
        
        # Initialize any new attributes for the subclass
        self.natx = 200
        self.fpcutoff = 5.5
        self.fp = None
        self.converged = False
        self.id = None
        self.types = None
        self.znucl = None

    def get_fp(self):
        types = self.types
        znucl = self.znucl
        natx = self.natx
        if self.fp is None:
            # Check necessary attributes are set
            if self.types is None:
                print("Warning: types is None in PallasAtom.get_fp!")
                return None
            if self.znucl is None or len(self.znucl) == 0:
                print("Warning: znucl is None or empty in PallasAtom.get_fp!")
                return None
                
            try:
                self.fp = self.cal_fp()
            except Exception as e:
                print(f"Error calculating fingerprint: {e}")
                return None
        return self.fp
    
    def cal_fp(self):
        lat = self.get_cell()
        rxyz = self.get_positions()
        types = self.types
        znucl = self.znucl
        natx = self.natx
        
        # Safety checks
        if types is None:
            print("Warning: types is None in PallasAtom.cal_fp!")
            return None
        if znucl is None or len(znucl) == 0:
            print("Warning: znucl is None or empty in PallasAtom.cal_fp!")
            return None
            
        try:
            cell = (lat, rxyz, types, znucl)
            fp = libfp.get_lfp(cell, cutoff=self.fpcutoff, log=False, natx = natx, orbital='s')
            self.fp = fp
            return fp
        except Exception as e:
            print(f"Error in libfp.get_lfp: {e}")
            return None


def main():
    pallas = Pallas()
    poscars = ['POSCAR1', 'POSCAR2']
    pallas.init_run(poscars)
    pallas.run_pso()


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
    start, end: The starting and ending node IDs
    
    Returns:
    list: Paths sorted by energy barrier (lowest first) and then by length
    """
    
    def minimax_cost(path):
        """Calculate the minimax cost of a path using edge weights."""
        max_weight = 0
        total_dist = 0
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i+1])
            weight = edge_data.get('weight', float('inf'))
            dist = edge_data.get('dist', float('inf'))
            max_weight = max(max_weight, weight)
            total_dist += dist
        return max_weight, total_dist
    
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
    # Sort by: 1) max energy barrier, 2) total fingerprint distance, 3) path length 
    all_paths.sort(key=lambda p: (minimax_cost(p)[0], minimax_cost(p)[1], len(p)))
    
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
            # Calculate path properties using edge attributes
            max_energy = 0
            total_distance = 0
            
            for j in range(len(path)-1):
                edge_data = G.get_edge_data(path[j], path[j+1])
                weight = edge_data.get('weight', float('inf'))
                dist = edge_data.get('dist', float('inf'))
                max_energy = max(max_energy, weight)
                total_distance += dist
            
            print(f"\nPath {i}: Max Energy Barrier = {max_energy:.4f}, Total FP Distance = {total_distance:.4f}, Length = {len(path)}")
            
            # Create a folder for this path
            path_folder = f"path_{i}"
            os.makedirs(path_folder, exist_ok=True)
            
            # Write path information to a summary file
            with open(os.path.join(path_folder, "path_info.txt"), 'w') as f:
                f.write(f"Path {i}\n")
                f.write(f"Max Energy Barrier: {max_energy:.6f}\n")
                f.write(f"Total FP Distance: {total_distance:.6f}\n")
                f.write(f"Number of nodes: {len(path)}\n\n")
                f.write("Node details:\n")
                
                for j, node in enumerate(path):
                    node_data = G.nodes[node]
                    node_type = 'Minimum' if node_data['xname'].startswith('M') else 'Saddle'
                    f.write(f"Node {node} ({node_type}): Energy = {node_data['e']:.6f}, Volume = {node_data['volume']:.6f}\n")
                    
                    # Write edge information
                    if j < len(path) - 1:
                        edge_data = G.get_edge_data(path[j], path[j+1])
                        edge_weight = edge_data.get('weight', float('inf'))
                        edge_dist = edge_data.get('dist', float('inf'))
                        f.write(f"  Edge to next node: Weight = {edge_weight:.6f}, FP Distance = {edge_dist:.6f}\n")
            
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
    
    # Create a directory to store all path files
    os.makedirs("path_energies", exist_ok=True)

    for i, path in enumerate(paths, 1):
        filename = f"path_energies/path_{i}_energy.txt"
        with open(filename, 'w') as f:
            f.write(f"#Path {i}\n")
            f.write("#Distance Energy EdgeWeight\n")
            
            cumulative_distance = 0.0
            for j, node in enumerate(path):
                node_data = G.nodes[node]
                energy = node_data['e']
                
                if j == 0:
                    f.write(f"{cumulative_distance:.6f} {energy:.6f} 0.0\n")
                else:
                    prev_node = path[j-1]
                    edge_data = G.get_edge_data(prev_node, node)
                    edge_weight = edge_data.get('weight', float('inf'))
                    edge_dist = edge_data.get('dist', 0.0)
                    cumulative_distance += edge_dist
                    f.write(f"{cumulative_distance:.6f} {energy:.6f} {edge_weight:.6f}\n")

    print("Energy profiles for all paths have been written to separate files in the 'path_energies' directory.")

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
    
