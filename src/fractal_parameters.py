
import numpy as np

class Fractal: 
    """
    Fractal class for understanding fractal patterns. Uses FracVAL generated
    algorithm data structure. 
    """
    def __init__(self, D_f, k_f, R_g, N, r_PP, monodisperse=True):
        """
        Initialize the Fractal object. The fractal parameters are 
        given as input. The fractal is assumed to be monodisperse by default.
        Initial parameters:
            D_f: Fractal dimension 
            k_f: Fractal prefactor
            R_g: Radius of gyration
            N: Number of particles
            r_PP: Particle radius
            monodisperse: Monodispersity
        """
        self.D_f = D_f
        self.k_f = k_f
        self.R_g = R_g
        self.N = N
        self.r_PP = r_PP
        self.monodisperse = monodisperse

    def learn_points(self, filename):
        """
        Learn the primary particle positions from the given filename. The 
        file is expexted to have to following structure: 
            x_1 y_1 z_1 r_1
            x_2 y_2 z_2 r_2
            ...
            x_N y_N z_N r_N
        where x, y, z are the coordinates of the primary particle and r is
        the radius of the primary particle.
        """
        particle_data = np.loadtxt(filename)
        self.points = particle_data[:, :3]
        self.radii = particle_data[:, 3]
    
    def measure(self):
        """
        Take measurements of the fractal. This function will continously be 
        updated. Current ideas for measurements are as follows:
            - Distances between each particle 
            - Angles made by each triplet of particles
            - Number of contact points for each particle
            - The tipline, the longest length between any two particles
        """
        Measurements.measure_distances(self)
        Measurements.find_chains(self)

class Measurements:
    """
    A class for storing measumement methods for the fractal. 
    """
    @staticmethod
    def measure_distances(fractal):
        """
        Calculate the distances between each pair of points. 
        """
        fractal.distances = np.zeros((fractal.N, fractal.N))
        fractal.contacts = np.zeros((fractal.N, fractal.N))
        for i in range(fractal.N):
            for j in range(i+1, fractal.N): 
                distance = np.linalg.norm(fractal.points[i] - fractal.points[j])
                fractal.distances[i, j] = distance
                if distance <= fractal.radii[i] + fractal.radii[j]:
                    fractal.contacts[i, j] = 1
                    fractal.contacts[j, i] = 1


    @staticmethod
    def identify_points(fractal):
        """
        Calculate the number of contact points for each particle. 
        """
        fractal.tips = []
        fractal.links = []
        fractal.intersections = []
        for particle in range(fractal.N):
            if sum(fractal.contacts[particle, :]) == 1:
                fractal.tips.append(particle)
            elif sum(fractal.contacts[particle, :]) == 2:
                fractal.links.append(particle)
            else:
                fractal.intersections.append(particle)


    @staticmethod
    def find_chains(fractal):
        Measurements.identify_points(fractal)

        if not fractal.intersections:
            fractal.chains = range(fractal.N)
            return
        
        fractal.chains = []
        for inter in fractal.intersections:
            neighbors = np.where(fractal.contacts[inter, :] == 1)[0]
            chain = []
            for neighbor in neighbors:
                chain.append(inter)
                while neighbor and len(neighbor < 3):
                    chain.append(neighbor)
                    next_particle = np.where(fractal.contacts[neighbor, :] == 1)[0]
                    neighbor = [contact for contact in next_particle if not any(contact in chain for chain in fractal.chains)]
                fractal.chains.append(chain)
        #TODO: find maximum length chain, named main chain

    #TODO: include angle calculations (iterate over all links and intersections, calculate angles in all positions)
