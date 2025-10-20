import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.deeplabv3plus import DeepLabV3Plus
from models.tsfen import TSFEN
from models.federated_model import FederatedModel
from core.vehicle_dynamics import IntelligentDriverModel
from core.flmd_calculator import FLMDCalculator
from core.channel_model import ChannelModel
from core.resource_allocation import ResourceAllocationOptimizer
from training.federated_learning import FederatedAveragingEngine
from training.mappo_agent import MAPPOAgent
from training.gradient_compression import GradientCompressionEngine
from utils.config import (
    SYSTEM_CONFIG, VEHICLE_DYNAMICS_CONFIG, CHANNEL_CONFIG,
    RESOURCE_ALLOCATION_CONFIG, FLMD_CONFIG, FEDERATED_LEARNING_CONFIG,
    MAPPO_CONFIG, TSFEN_CONFIG, DEEPLABV3PLUS_CONFIG
)
from utils.data_loader import DataLoaderWrapper, get_dummy_batch


def initialize_system():
    N = SYSTEM_CONFIG['N']
    K = SYSTEM_CONFIG['K']
    M = SYSTEM_CONFIG['M']
    
    base_model = DeepLabV3Plus(
        num_classes=DEEPLABV3PLUS_CONFIG['num_classes'],
        output_stride=DEEPLABV3PLUS_CONFIG['output_stride']
    )
    
    federated_model = FederatedModel(base_model)
    
    federated_engine = FederatedAveragingEngine(N=N, learning_rate=FEDERATED_LEARNING_CONFIG['learning_rate'])
    federated_engine.initialize_global_model(federated_model)
    
    vehicle_dynamics = IntelligentDriverModel(N=N, tau=SYSTEM_CONFIG['tau'])
    
    flmd_calculator = FLMDCalculator(
        N=N,
        lambda_theta=FLMD_CONFIG['lambda_theta'],
        beta=FLMD_CONFIG['beta'],
        mu_over_L=FLMD_CONFIG['mu_over_L']
    )
    
    channel_model = ChannelModel(
        B=CHANNEL_CONFIG['B'],
        N0=CHANNEL_CONFIG['N0'],
        alpha=CHANNEL_CONFIG['alpha'],
        P_error=CHANNEL_CONFIG['P_error']
    )
    
    resource_optimizer = ResourceAllocationOptimizer(
        D_n=RESOURCE_ALLOCATION_CONFIG['D_n'],
        B=RESOURCE_ALLOCATION_CONFIG['B'],
        e_max=RESOURCE_ALLOCATION_CONFIG['e_max'],
        mu=RESOURCE_ALLOCATION_CONFIG['mu'],
        kappa=RESOURCE_ALLOCATION_CONFIG['kappa'],
        G_n=RESOURCE_ALLOCATION_CONFIG['G_n'],
        P_n=RESOURCE_ALLOCATION_CONFIG['P_n']
    )
    
    gradient_compressor = GradientCompressionEngine(
        compression_ratio=FEDERATED_LEARNING_CONFIG['gradient_compression_ratio']
    )
    
    tsfen_network = TSFEN(
        d_model=TSFEN_CONFIG['d_model'],
        num_heads=TSFEN_CONFIG['num_heads'],
        hidden_size=TSFEN_CONFIG['hidden_size'],
        num_agents=K
    )
    
    mappo_agents = [
        MAPPOAgent(
            agent_id=k,
            state_dim=M * N * 3,
            action_dim=N,
            hidden_dim=MAPPO_CONFIG['hidden_dim'],
            learning_rate=MAPPO_CONFIG['learning_rate']
        )
        for k in range(K)
    ]
    
    return {
        'federated_engine': federated_engine,
        'vehicle_dynamics': vehicle_dynamics,
        'flmd_calculator': flmd_calculator,
        'channel_model': channel_model,
        'resource_optimizer': resource_optimizer,
        'gradient_compressor': gradient_compressor,
        'tsfen_network': tsfen_network,
        'mappo_agents': mappo_agents
    }


def construct_state_tensor(vehicle_dynamics, flmd_values, channel_csi, M, N):
    state_sequence = []
    
    for m in range(M):
        time_slot_state = []
        for n in range(N):
            flmd_n = flmd_values[n] if n < len(flmd_values) else 0.0
            h_n_squared = channel_csi[n]['h_n_squared'] if n in channel_csi else 1.0
            aoi_n = 0.0
            
            state_vector = [flmd_n, h_n_squared, aoi_n]
            time_slot_state.append(state_vector)
        
        state_sequence.append(time_slot_state)
    
    state_tensor = torch.FloatTensor(state_sequence)
    return state_tensor.unsqueeze(0)


def compute_mappo_reward(vehicle_id, delay, flmd_value, alpha=3.76, beta=1.0, M=5, K=4):
    reward = -(alpha * delay + beta * (flmd_value ** 2)) / (M * K)
    return reward


def training_loop(systems, num_rounds=500):
    N = SYSTEM_CONFIG['N']
    K = SYSTEM_CONFIG['K']
    M = SYSTEM_CONFIG['M']
    
    federated_engine = systems['federated_engine']
    vehicle_dynamics = systems['vehicle_dynamics']
    flmd_calculator = systems['flmd_calculator']
    channel_model = systems['channel_model']
    resource_optimizer = systems['resource_optimizer']
    gradient_compressor = systems['gradient_compressor']
    tsfen_network = systems['tsfen_network']
    mappo_agents = systems['mappo_agents']
    
    data_loader = DataLoaderWrapper(batch_size=32, num_batches=10)
    
    for communication_round in range(num_rounds):
        flmd_values = []
        for client_id in range(N):
            theta_n = flmd_calculator.compute_model_drift(
                federated_engine.client_models[client_id],
                federated_engine.global_model
            )
            flmd_values.append(theta_n)
        
        eligible_clients = flmd_calculator.select_eligible_clients(flmd_values)
        
        sample_sizes = {n: np.random.randint(800, 1200) for n in range(N)}
        
        for client_id in eligible_clients:
            batch_images, batch_labels = get_dummy_batch(32)
            loss = federated_engine.local_training_step(client_id, batch_images, batch_labels, num_epochs=1)
        
        federated_engine.aggregate_models(eligible_clients, sample_sizes)
        
        channel_csi = channel_model.generate_channel_state_information(
            vehicle_dynamics.positions,
            communication_round
        )
        
        vehicles_data = {}
        for n in range(N):
            vehicles_data[n] = {
                'zeta_n': sample_sizes[n],
                'h_n_squared': channel_csi[n]['h_n_squared']
            }
        
        resource_results = resource_optimizer.solve_multi_vehicle_optimization(vehicles_data)
        
        state_tensor = construct_state_tensor(vehicle_dynamics, flmd_values, channel_csi, M, N)
        
        with torch.no_grad():
            actor_logits, critic_value, attention_weights = tsfen_network(state_tensor)
        
        for agent_k in mappo_agents:
            state_flat = state_tensor.view(-1).numpy()
            action, log_prob, value = agent_k.select_action(state_flat)
            
            delay = resource_results[action % N]['total_delay']
            flmd_val = flmd_values[action % N]
            reward = compute_mappo_reward(action % N, delay, flmd_val)
            
            next_state_flat = state_flat.copy()
            done = False
            
            agent_k.store_experience(state_flat, action, reward, next_state_flat, done, log_prob, value)
        
        for agent_k in mappo_agents:
            if len(agent_k.experience_buffer['states']) > 0:
                actor_loss, critic_loss = agent_k.update_networks(
                    gamma=MAPPO_CONFIG['gamma'],
                    lambda_gae=MAPPO_CONFIG['lambda_gae'],
                    epsilon_clip=MAPPO_CONFIG['epsilon_clip'],
                    num_epochs=MAPPO_CONFIG['num_epochs']
                )
        
        vehicle_dynamics.update_all_vehicles()
        
        safety_violations = vehicle_dynamics.check_safety_constraints()
        string_stability = vehicle_dynamics.compute_string_stability()
        fuel_efficiency = vehicle_dynamics.compute_fuel_efficiency()
        platoon_cohesion = vehicle_dynamics.compute_platoon_cohesion()
        
        federated_engine.increment_communication_round()
        
        if (communication_round + 1) % 50 == 0:
            print(f"Round {communication_round + 1}/{num_rounds}")
            print(f"  FLMD Mean: {np.mean(flmd_values):.6f}")
            print(f"  String Stability: {string_stability:.4f}")
            print(f"  Fuel Efficiency: {fuel_efficiency:.4f}")
            print(f"  Platoon Cohesion: {platoon_cohesion:.4f}")
            print(f"  Safety Violations: {len(safety_violations)}")


def main():
    print("Initializing Vehicular Platooning with Wireless Federated Learning System...")
    systems = initialize_system()
    
    print("Starting training loop...")
    training_loop(systems, num_rounds=SYSTEM_CONFIG['communication_rounds'])
    
    print("Training completed.")


if __name__ == '__main__':
    main()

