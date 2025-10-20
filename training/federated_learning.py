import torch
import torch.nn as nn
import copy
import numpy as np


class FederatedAveragingEngine:
    def __init__(self, N=20, learning_rate=1e-4):
        self.N = N
        self.learning_rate = learning_rate
        self.global_model = None
        self.client_models = [None] * N
        self.client_optimizers = [None] * N
        self.communication_round = 0

    def initialize_global_model(self, model):
        self.global_model = copy.deepcopy(model)
        for n in range(self.N):
            self.client_models[n] = copy.deepcopy(model)
            self.client_optimizers[n] = torch.optim.AdamW(
                self.client_models[n].parameters(),
                lr=self.learning_rate
            )

    def local_training_step(self, client_id, batch_data, batch_labels, num_epochs=1):
        model = self.client_models[client_id]
        optimizer = self.client_optimizers[client_id]
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
        
        return loss.item()

    def compute_model_drift(self, client_id):
        local_params = torch.cat([p.flatten() for p in self.client_models[client_id].parameters()])
        global_params = torch.cat([p.flatten() for p in self.global_model.parameters()])
        
        numerator = torch.norm(local_params - global_params)
        denominator = torch.norm(global_params)
        
        if denominator == 0:
            theta_n = 0.0
        else:
            theta_n = (numerator / denominator).item()
        
        return theta_n

    def select_eligible_clients(self, flmd_values, lambda_theta=0.1):
        eligible_clients = []
        for client_id, theta_n in enumerate(flmd_values):
            if theta_n <= lambda_theta:
                eligible_clients.append(client_id)
        
        if not eligible_clients:
            eligible_clients = [0]
        
        return eligible_clients

    def aggregate_models(self, eligible_clients, sample_sizes):
        total_samples = sum(sample_sizes[c] for c in eligible_clients)
        
        aggregated_state_dict = {}
        for name, param in self.global_model.state_dict().items():
            aggregated_state_dict[name] = torch.zeros_like(param)
        
        for client_id in eligible_clients:
            weight = sample_sizes[client_id] / total_samples
            
            for name, param in self.client_models[client_id].state_dict().items():
                aggregated_state_dict[name] += weight * param
        
        self.global_model.load_state_dict(aggregated_state_dict)
        
        for client_id in range(self.N):
            self.client_models[client_id].load_state_dict(self.global_model.state_dict())

    def broadcast_global_model(self):
        for client_id in range(self.N):
            self.client_models[client_id].load_state_dict(self.global_model.state_dict())

    def get_global_model(self):
        return copy.deepcopy(self.global_model)

    def get_client_model(self, client_id):
        return copy.deepcopy(self.client_models[client_id])

    def increment_communication_round(self):
        self.communication_round += 1

    def get_communication_round(self):
        return self.communication_round

    def compute_global_loss(self, test_data, test_labels):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            outputs = self.global_model(test_data)
            loss = criterion(outputs, test_labels)
        
        return loss.item()

    def compute_client_losses(self, test_data, test_labels):
        losses = {}
        criterion = nn.CrossEntropyLoss()
        
        for client_id in range(self.N):
            self.client_models[client_id].eval()
            
            with torch.no_grad():
                outputs = self.client_models[client_id](test_data)
                loss = criterion(outputs, test_labels)
            
            losses[client_id] = loss.item()
        
        return losses

