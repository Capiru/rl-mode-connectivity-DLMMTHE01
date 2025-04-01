import os
import matplotlib.pyplot as plt
import seaborn as sns


def interpolation_plot(plot_df, y, iteration, epoch_split, cfg=None):
    plt.clf()
    plot_df.plot(x="alpha", y=y, kind="line")
    try:
        plt.savefig(
            cfg.episode_save_path
            + f"imgs/interpolation_{y}_{iteration}_epoch_split_{epoch_split}.png"
        )
    except Exception:
        os.makedirs(cfg.episode_save_path + "imgs/")
        plt.savefig(
            cfg.episode_save_path
            + f"imgs/interpolation_{y}_{iteration}_epoch_split_{epoch_split}.png"
        )


def model_iterations_plot(model_id, cfg=None):
    df = cfg.models_df.copy()
    df = df.loc[df["model_id"] == model_id]

    # Avg Moves
    plt.clf()
    sns.lineplot(data=df, x="episodes", y="avg_moves", hue="num_sims")
    try:
        plt.savefig(cfg.episode_save_path + f"imgs/{model_id}_model_avg_moves.png")
    except Exception:
        os.makedirs(cfg.episode_save_path + "imgs/")
        plt.savefig(cfg.episode_save_path + f"imgs/{model_id}_model_avg_moves.png")

    # Elo (random opponent)
    plt.clf()
    sns.lineplot(data=df, x="episodes", y="elo", hue="num_sims")
    plt.savefig(cfg.episode_save_path + f"imgs/{model_id}_model_elo.png")

    # Policy Loss
    plt.clf()
    sns.lineplot(data=df, x="episodes", y="policy_loss", hue="num_sims")
    plt.savefig(cfg.episode_save_path + f"imgs/{model_id}_model_policy_loss.png")

    # Value Loss
    plt.clf()
    sns.lineplot(data=df, x="episodes", y="value_loss", hue="num_sims")
    plt.savefig(cfg.episode_save_path + f"imgs/{model_id}_model_value_loss.png")

    # Eval Policy Loss
    plt.clf()
    sns.lineplot(data=df, x="episodes", y="eval_policy_loss", hue="num_sims")
    plt.savefig(cfg.episode_save_path + f"imgs/{model_id}_model_eval_policy_loss.png")

    # Eval Value Loss
    plt.clf()
    sns.lineplot(data=df, x="episodes", y="eval_value_loss", hue="num_sims")
    plt.savefig(cfg.episode_save_path + f"imgs/{model_id}_model_eval_value_loss.png")
