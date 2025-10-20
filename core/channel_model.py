import numpy as np
import math


class ChannelModel:
    def __init__(self, B=56e6, N0=1e-9, alpha=3.76, P_error=0.1):
        self.B = B
        self.N0 = N0
        self.alpha = alpha
        self.P_error = P_error
        
        self.eta = 1.0
        self.f_c = 2.4e9

    def compute_path_loss(self, distance):
        if distance <= 0:
            distance = 1.0
        
        path_loss_db = 32.45 + 20 * math.log10(self.f_c / 1e9) + 20 * math.log10(distance)
        path_loss_linear = 10 ** (-path_loss_db / 10)
        
        return path_loss_linear

    def compute_shadowing(self, sigma_sh=8.0):
        shadowing_db = np.random.normal(0, sigma_sh)
        shadowing_linear = 10 ** (shadowing_db / 10)
        
        return shadowing_linear

    def compute_fading(self):
        fading = np.random.rayleigh(1.0) ** 2
        return fading

    def compute_channel_gain(self, distance):
        path_loss = self.compute_path_loss(distance)
        shadowing = self.compute_shadowing()
        fading = self.compute_fading()
        
        h_hat = fading * self.eta * path_loss
        
        h_tilde = np.random.rayleigh(1.0) ** 2
        
        h_n_squared = math.sqrt(1 - self.P_error) * h_hat + math.sqrt(self.P_error) * h_tilde
        
        return max(h_n_squared, 1e-10)

    def compute_snr(self, h_n_squared, P_tx):
        snr = P_tx * h_n_squared / self.N0
        return snr

    def compute_channel_capacity(self, h_n_squared, P_tx):
        snr = self.compute_snr(h_n_squared, P_tx)
        capacity = self.B * math.log2(1 + snr)
        return capacity

    def compute_transmission_time(self, data_size, h_n_squared, P_tx):
        capacity = self.compute_channel_capacity(h_n_squared, P_tx)
        
        if capacity <= 0:
            return float('inf')
        
        transmission_time = data_size / capacity
        return transmission_time

    def compute_transmission_energy(self, transmission_time, P_tx, h_n_squared):
        energy = P_tx * transmission_time
        return energy

    def generate_channel_state_information(self, vehicle_positions, communication_round):
        N = len(vehicle_positions)
        csi = {}
        
        for n in range(N):
            distance = vehicle_positions[n]
            
            h_n_squared = self.compute_channel_gain(distance)
            
            csi[n] = {
                'h_n_squared': h_n_squared,
                'distance': distance,
                'communication_round': communication_round
            }
        
        return csi

    def estimate_channel_with_error(self, true_channel_gain, pilot_contamination_ratio=0.1):
        estimation_error = np.random.normal(0, pilot_contamination_ratio * true_channel_gain)
        estimated_gain = true_channel_gain + estimation_error
        
        return max(estimated_gain, 1e-10)

