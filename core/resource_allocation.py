import numpy as np
import math
from scipy.optimize import minimize_scalar, brentq


class ResourceAllocationOptimizer:
    def __init__(self, D_n=10.08e6, B=56e6, e_max=0.1, mu=1e7, kappa=1e-28, G_n=0.5e9, P_n=15):
        self.D_n = D_n
        self.B = B
        self.e_max = e_max
        self.mu = mu
        self.kappa = kappa
        self.G_n = G_n
        self.P_n = P_n

    def compute_computation_delay(self, chi_n, zeta_n):
        if chi_n <= 0:
            return float('inf')
        
        tau_comp = (self.mu * zeta_n) / (chi_n * self.G_n)
        return tau_comp

    def compute_transmission_delay(self, delta_tx_n):
        return delta_tx_n

    def compute_total_delay(self, chi_n, delta_tx_n, zeta_n):
        tau_comp = self.compute_computation_delay(chi_n, zeta_n)
        tau_tx = self.compute_transmission_delay(delta_tx_n)
        return tau_comp + tau_tx

    def compute_computation_energy(self, chi_n, zeta_n):
        E_comp = self.kappa * self.mu * zeta_n * (chi_n * self.G_n) ** 2
        return E_comp

    def compute_transmission_energy(self, delta_tx_n, h_n_squared):
        if delta_tx_n <= 0:
            return float('inf')
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        E_tx = delta_tx_n * (exponential_term - 1) / h_n_squared
        return E_tx

    def compute_total_energy(self, chi_n, delta_tx_n, zeta_n, h_n_squared):
        E_comp = self.compute_computation_energy(chi_n, zeta_n)
        E_tx = self.compute_transmission_energy(delta_tx_n, h_n_squared)
        return E_comp + E_tx

    def check_energy_constraint(self, chi_n, delta_tx_n, zeta_n, h_n_squared):
        total_energy = self.compute_total_energy(chi_n, delta_tx_n, zeta_n, h_n_squared)
        return total_energy <= self.e_max

    def optimize_high_snr(self, zeta_n, h_n_squared):
        def energy_constraint_func(delta_tx):
            if delta_tx <= 0:
                return float('inf')
            
            chi_n = 1.0
            total_energy = self.compute_total_energy(chi_n, delta_tx, zeta_n, h_n_squared)
            return total_energy - self.e_max

        delta_tx_min = self.D_n / (self.B * math.log2(1 + self.P_n * h_n_squared))
        delta_tx_max = 10.0

        try:
            delta_tx_star = brentq(energy_constraint_func, delta_tx_min, delta_tx_max, xtol=1e-6, maxiter=100)
        except:
            delta_tx_star = delta_tx_min

        chi_n_star = 1.0
        
        exponential_term = 2 ** (self.D_n / (delta_tx_star * self.B))
        rho_n_star = min((exponential_term - 1) / (self.P_n * h_n_squared), 1.0)
        rho_n_star = max(rho_n_star, 0.0)

        return chi_n_star, rho_n_star, delta_tx_star

    def optimize_lagrangian(self, zeta_n, h_n_squared, lambda_1=0.0):
        if lambda_1 == 0:
            chi_n_star = 1.0
            rho_n_star = 1.0
        else:
            chi_n_star = min(((1.0 / (2 * lambda_1 * self.kappa * (self.G_n ** 3))) ** (1.0 / 3.0)), 1.0)
            chi_n_star = max(chi_n_star, 0.0)

            def energy_constraint_func(delta_tx):
                if delta_tx <= 0:
                    return float('inf')
                
                total_energy = self.compute_total_energy(chi_n_star, delta_tx, zeta_n, h_n_squared)
                return total_energy - self.e_max

            delta_tx_min = self.D_n / (self.B * math.log2(1 + self.P_n * h_n_squared))
            delta_tx_max = 10.0

            try:
                delta_tx_star = brentq(energy_constraint_func, delta_tx_min, delta_tx_max, xtol=1e-6, maxiter=100)
            except:
                delta_tx_star = delta_tx_min

            exponential_term = 2 ** (self.D_n / (delta_tx_star * self.B))
            rho_n_star = min((exponential_term - 1) / (self.P_n * h_n_squared), 1.0)
            rho_n_star = max(rho_n_star, 0.0)

        return chi_n_star, rho_n_star, delta_tx_star

    def solve_multi_vehicle_optimization(self, vehicles_data):
        results = {}
        
        for vehicle_id, vehicle_info in vehicles_data.items():
            zeta_n = vehicle_info['zeta_n']
            h_n_squared = vehicle_info['h_n_squared']
            
            chi_n, rho_n, delta_tx = self.optimize_high_snr(zeta_n, h_n_squared)
            
            results[vehicle_id] = {
                'chi_n': chi_n,
                'rho_n': rho_n,
                'delta_tx': delta_tx,
                'total_delay': self.compute_total_delay(chi_n, delta_tx, zeta_n),
                'total_energy': self.compute_total_energy(chi_n, delta_tx, zeta_n, h_n_squared),
                'feasible': self.check_energy_constraint(chi_n, delta_tx, zeta_n, h_n_squared)
            }
        
        return results

