from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torch.distributions import Categorical
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def train_model(
    model, dataset=None, epochs=1, iteration=0, learning_rate=None, cfg=None
):
    if not learning_rate:
        learning_rate = cfg.learning_rate

    model.to(cfg.device)
    model.train()
    # Load dataset
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Initialize model, loss functions, and optimizer
    criterion_policy = nn.CrossEntropyLoss(
        reduction="none"
    )  # Policy loss (assuming class labels)
    criterion_value = nn.MSELoss()  # Value loss (regression)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg.weight_decay,
        momentum=0.9,
    )

    # Training loop
    for epoch in range(epochs):
        running_policy_loss = 0.0
        running_value_loss = 0.0
        with tqdm(total=len(dataloader)) as pbar:
            for i, (obs, action, reward) in enumerate(dataloader):
                # Forward pass
                obs, action, reward = (
                    obs.to(cfg.device),
                    action.to(cfg.device),
                    reward.to(cfg.device),
                )
                optimizer.zero_grad()
                policy_pred, value_pred = model(obs)

                torch.clamp(policy_pred, min=1e-8, max=1 - 1e-8)

                # Compute Loss for Value Network
                loss_value = criterion_value(value_pred, reward)

                # AlphaZero
                if cfg.num_simulations > 0:
                    loss_policy = criterion_policy(policy_pred, action)

                # REINFORCE with Baseline
                elif cfg.num_simulations == 0:
                    # In the first iteration the value function is not yet trained
                    if iteration == 0:
                        advantage = reward
                    else:
                        with torch.no_grad():
                            advantage = reward - value_pred
                    m = Categorical(policy_pred)
                    loss_policy = -m.log_prob(torch.argmax(action, dim=-1)) * advantage

                # Backward pass
                total_loss = torch.mean(loss_policy) + loss_value
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_policy_loss += torch.mean(loss_policy).item()
                running_value_loss += loss_value.item()
                pbar.set_description(
                    f"Train Iteration {iteration} - Policy Loss {running_policy_loss/(i+1)} - Value Loss {running_value_loss/(i+1)}"
                )
                pbar.update(1)

        logger.debug(
            f"Epoch {epoch+1}/{epochs}, Policy Loss: {running_policy_loss:.4f}, Value Loss: {running_value_loss:.4f} N_batches: {len(dataloader)}"
        )
    model.to("cpu")
    return model, (
        running_policy_loss / len(dataloader),
        running_value_loss / len(dataloader),
    )
