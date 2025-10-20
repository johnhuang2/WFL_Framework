import torch
import numpy as np
import math


class FLMDCalculator:
    def __init__(self, N=20, lambda_theta=0.1, beta=1.0, mu_over_L=0.01):
        self.N = N
        self.lambda_theta = lambda_theta
        self.beta = beta
        self.mu_over_L = mu_over_L
        
        self.flmd_history = [[] for _ in range(N)]
        self.drift_velocity = np.zeros(N)
        self.drift_acceleration = np.zeros(N)

    def compute_model_drift(self, local_model, global_model):
        local_params = torch.cat([p.flatten() for p in local_model.parameters()])
        global_params = torch.cat([p.flatten() for p in global_model.parameters()])
        
        numerator = torch.norm(local_params - global_params)
        denominator = torch.norm(global_params)
        
        if denominator == 0:
            theta_n = 0.0
        else:
            theta_n = (numerator / denominator).item()
        
        return theta_n

    def compute_weighted_flmd(self, theta_n, sample_size):
        sample_weight = math.log(sample_size + 1) / math.log(1001)
        weighted_drift = theta_n * sample_weight
        return weighted_drift

    def compute_temporal_evolution(self, vehicle_id, current_drift):
        if len(self.flmd_history[vehicle_id]) == 0:
            self.flmd_history[vehicle_id].append(current_drift)
            return 0.0, 0.0
        
        previous_drift = self.flmd_history[vehicle_id][-1]
        current_velocity = current_drift - previous_drift
        
        previous_velocity = self.drift_velocity[vehicle_id]
        current_acceleration = current_velocity - previous_velocity
        
        self.drift_velocity[vehicle_id] = current_velocity
        self.drift_acceleration[vehicle_id] = current_acceleration
        self.flmd_history[vehicle_id].append(current_drift)
        
        return current_velocity, current_acceleration

    def select_eligible_clients(self, flmd_values):
        eligible_clients = []
        for client_id, theta_n in enumerate(flmd_values):
            if theta_n <= self.lambda_theta:
                eligible_clients.append(client_id)
        return eligible_clients

    def compute_adaptive_weights(self, flmd_values, communication_round):
        adaptive_weights = np.zeros(self.N)
        
        for n in range(self.N):
            theta_n = flmd_values[n]
            
            if theta_n <= self.lambda_theta:
                adaptive_weights[n] = 1.0
            else:
                exponent = -self.beta * theta_n * ((1 - self.mu_over_L) ** (-communication_round))
                adaptive_weights[n] = math.exp(exponent)
        
        return adaptive_weights

    def update_adaptive_threshold(self, flmd_values, convergence_rate):
        mean_flmd = np.mean(flmd_values)
        std_flmd = np.std(flmd_values)
        
        if convergence_rate > 0.8:
            self.lambda_theta = max(0.05, self.lambda_theta * 0.95)
        elif convergence_rate < 0.3:
            self.lambda_theta = min(0.5, self.lambda_theta * 1.05)
        
        return self.lambda_theta

    def get_flmd_statistics(self):
        all_flmd_values = []
        for history in self.flmd_history:
            if history:
                all_flmd_values.append(history[-1])
        
        if not all_flmd_values:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0}
        
        return {
            'mean': np.mean(all_flmd_values),
            'std': np.std(all_flmd_values),
            'max': np.max(all_flmd_values),
            'min': np.min(all_flmd_values)
        }

    def reset(self):
        self.flmd_history = [[] for _ in range(self.N)]
        self.drift_velocity = np.zeros(self.N)
        self.drift_acceleration = np.zeros(self.N)

