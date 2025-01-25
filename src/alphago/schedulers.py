import math


def cosine_decay(f_init, f_min, epoch, max_epoch):
    if epoch >= max_epoch:
        return f_min
    return f_min + (f_init - f_min) * (
        1 + math.cos((math.pi / 2) + (math.pi / 2) * epoch / (max_epoch))
    )


def lr_scheduler(epoch, cfg=None):
    return cosine_decay(
        cfg.learning_rate,
        cfg.min_learning_rate,
        epoch,
        cfg.lr_schedule_max_epochs // cfg.episodes_per_epoch,
    )


def eps_scheduler(epoch, cfg=None):
    return cosine_decay(
        cfg.eps, 0.05, epoch, cfg.eps_schedule_max_epochs // cfg.episodes_per_epoch
    )


def buffer_size_scheduler(episodes, cfg=None):
    threshold = cfg.buffer_size_schedule_max_epochs
    if episodes >= threshold:
        return int(
            min(
                cfg.episodes_per_epoch * ((episodes // threshold) + 1),
                cfg.episodes_replay_buffer_size,
            )
        )
    else:
        return cfg.episodes_per_epoch
