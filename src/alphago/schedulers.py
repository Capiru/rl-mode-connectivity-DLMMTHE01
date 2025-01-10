from alphago.config import cfg
import math

def cosine_decay(f_init, f_min,epoch, max_epoch):
    if epoch >= max_epoch/2:
        epoch = max_epoch/2
    return f_min + (f_init - f_min) * (1 + math.cos((math.pi / 2)+(math.pi / 2) * epoch / (max_epoch/2)))

def lr_scheduler(epoch, cfg = cfg):
    return cosine_decay(cfg.learning_rate, cfg.min_learning_rate, epoch, cfg.max_n_episodes // cfg.episodes_per_epoch)

def eps_scheduler(epoch, cfg = cfg):
    return cosine_decay(cfg.eps, 0.05, epoch, cfg.max_n_episodes // cfg.episodes_per_epoch)