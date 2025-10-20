import numpy as np
import math


class IntelligentDriverModel:
    def __init__(self, N=20, tau=1.0):
        self.N = N
        self.tau = tau
        
        self.a_max = 0.73
        self.b_max = 1.67
        self.v_des = 30.0
        self.delta = 4
        self.d_min = 2.0
        self.t_min = 1.5
        
        self.positions = np.random.uniform(0, 1000, N)
        self.velocities = np.random.uniform(20, 30, N)
        self.accelerations = np.zeros(N)
        
        self.vehicle_length = 5.0

    def compute_relative_spacing(self, vehicle_id):
        if vehicle_id == 0:
            return float('inf')
        
        delta_x = self.positions[vehicle_id - 1] - self.positions[vehicle_id] - self.vehicle_length
        return delta_x

    def compute_relative_velocity(self, vehicle_id):
        if vehicle_id == 0:
            return 0.0
        
        delta_v = self.velocities[vehicle_id] - self.velocities[vehicle_id - 1]
        return delta_v

    def compute_desired_spacing(self, vehicle_id):
        v_n = self.velocities[vehicle_id]
        delta_v = self.compute_relative_velocity(vehicle_id)
        
        H = self.d_min + self.t_min * v_n + \
            (v_n * delta_v) / (2 * math.sqrt(self.a_max * self.b_max))
        
        return max(H, self.d_min)

    def compute_acceleration(self, vehicle_id):
        if vehicle_id == 0:
            return 0.0
        
        v_n = self.velocities[vehicle_id]
        delta_x = self.compute_relative_spacing(vehicle_id)
        H = self.compute_desired_spacing(vehicle_id)
        
        if delta_x <= 0:
            return -self.b_max
        
        velocity_term = (v_n / self.v_des) ** self.delta
        spacing_term = (H / delta_x) ** 2
        
        a_n = self.a_max * (1 - velocity_term - spacing_term)
        
        return np.clip(a_n, -self.b_max, self.a_max)

    def update_vehicle_state(self, vehicle_id):
        a_n = self.compute_acceleration(vehicle_id)
        self.accelerations[vehicle_id] = a_n
        
        self.velocities[vehicle_id] = self.velocities[vehicle_id] + a_n * self.tau
        self.velocities[vehicle_id] = np.clip(self.velocities[vehicle_id], 0, self.v_des)
        
        self.positions[vehicle_id] = self.positions[vehicle_id] + \
                                     self.velocities[vehicle_id] * self.tau + \
                                     0.5 * a_n * self.tau ** 2

    def update_all_vehicles(self):
        for vehicle_id in range(self.N):
            self.update_vehicle_state(vehicle_id)

    def check_safety_constraints(self):
        violations = []
        for vehicle_id in range(1, self.N):
            delta_x = self.compute_relative_spacing(vehicle_id)
            if delta_x < self.d_min:
                violations.append({
                    'vehicle_id': vehicle_id,
                    'spacing': delta_x,
                    'min_spacing': self.d_min
                })
        return violations

    def compute_string_stability(self):
        velocity_amplifications = []
        for vehicle_id in range(1, self.N):
            if self.velocities[vehicle_id - 1] != 0:
                amplification = abs(self.velocities[vehicle_id]) / abs(self.velocities[vehicle_id - 1])
                velocity_amplifications.append(amplification)
        
        if not velocity_amplifications:
            return 1.0
        
        max_amplification = max(velocity_amplifications)
        return max_amplification

    def compute_fuel_efficiency(self):
        total_acceleration_squared = np.sum(self.accelerations ** 2)
        avg_acceleration_squared = total_acceleration_squared / self.N
        
        fuel_efficiency = 1.0 / (1.0 + avg_acceleration_squared)
        return fuel_efficiency

    def compute_platoon_cohesion(self):
        spacings = []
        for vehicle_id in range(1, self.N):
            delta_x = self.compute_relative_spacing(vehicle_id)
            spacings.append(delta_x)
        
        if not spacings:
            return 0.0
        
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        
        cohesion = 1.0 / (1.0 + std_spacing / (mean_spacing + 1e-6))
        return cohesion

    def get_state(self):
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'accelerations': self.accelerations.copy()
        }

    def set_state(self, state):
        self.positions = state['positions'].copy()
        self.velocities = state['velocities'].copy()
        self.accelerations = state['accelerations'].copy()

    def reset(self):
        self.positions = np.random.uniform(0, 1000, self.N)
        self.velocities = np.random.uniform(20, 30, self.N)
        self.accelerations = np.zeros(self.N)

