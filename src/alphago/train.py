from alphago.config import cfg
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")



def train_model(model,dataset= None, epochs=1, learning_rate=None, cfg = cfg):
    if not learning_rate:
        learning_rate = cfg.learning_rate

    model.train()
    # Load dataset
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Initialize model, loss functions, and optimizer
    criterion_policy = nn.CrossEntropyLoss(reduction="none")  # Policy loss (assuming class labels)
    criterion_value = nn.MSELoss() # Value loss (regression)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        running_policy_loss = 0.0
        running_value_loss = 0.0

        for i, (obs, action, reward) in enumerate(dataloader):
            if i >= cfg.batch_samples_from_buffer:
                break
            # Forward pass
            optimizer.zero_grad()
            policy_pred, value_pred = model(obs)
            
            torch.clamp(policy_pred, min=1e-8, max=1 - 1e-8)

            # Compute Loss for Value Network
            loss_value = criterion_value(value_pred, reward)
            
            # AlphaZero
            if cfg.num_simulations > 0:
                loss_policy = criterion_policy(policy_pred, action)
                
            # REINFORCE
            elif cfg.num_simulations == 0:
                m = Categorical(policy_pred)
                loss_policy = -m.log_prob(torch.argmax(action,dim=-1)) * reward


            # Backward pass
            total_loss = torch.mean(loss_policy) + loss_value
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_policy_loss += torch.mean(loss_policy).item()
            running_value_loss += loss_value.item()
            

        logger.debug(f"Epoch {epoch+1}/{epochs}, Policy Loss: {running_policy_loss:.4f}, Value Loss: {running_value_loss:.4f} N_batches: {len(dataloader)}")
    return model, (running_policy_loss/cfg.batch_samples_from_buffer,running_value_loss/cfg.batch_samples_from_buffer)
