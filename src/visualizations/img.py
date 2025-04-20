import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def load_experiment(
    paths: list, separator: str, separator_list: list, episodes_per_epoch: int = 1000
):
    df = pd.DataFrame()
    for path, sepator_i in zip(paths, separator_list):
        path_0 = path + "lmc_0_epoch_split_0.csv"
        df_1 = pd.read_csv(path_0)
        df_1.drop(columns="Unnamed: 0", inplace=True)
        df_1["Episodes"] = 0
        df_1["Episode Split"] = 0.0
        df_1[separator] = sepator_i
        df = pd.concat([df, df_1])

    for i in range(1, 2000):
        try:
            for path, sepator_i in zip(paths, separator_list):
                path_0 = path + f"lmc_{i}_epoch_split_0.csv"
                partial_df = pd.read_csv(path_0)
                partial_df.drop(columns="Unnamed: 0", inplace=True)
                partial_df["Episodes"] = episodes_per_epoch * i
                partial_df["Episode Split"] = 0.0
                partial_df[separator] = sepator_i
                df = pd.concat([df, partial_df])
        except FileNotFoundError:
            break

    return df


def load_experiment_norm(
    paths: list, separator: str, separator_list: list, episodes_per_epoch: int = 1000
):
    df = load_experiment(paths, separator, separator_list, episodes_per_epoch)
    df["max_policy_loss"] = df.groupby(["Episodes", separator])[
        "policy_loss"
    ].transform("max")
    df["min_policy_loss"] = df.groupby(["Episodes", separator])[
        "policy_loss"
    ].transform("min")
    df["max_elo"] = df.groupby(["Episodes", separator])["elo"].transform("max")
    df["min_elo"] = df.groupby(["Episodes", separator])["elo"].transform("min")
    df["max_value_loss"] = df.groupby(["Episodes", separator])["value_loss"].transform(
        "max"
    )
    df["min_value_loss"] = df.groupby(["Episodes", separator])["value_loss"].transform(
        "min"
    )
    df["policy_loss_barrier"] = (df["max_policy_loss"] - df["policy_loss"]) / (
        df["max_policy_loss"] - df["min_policy_loss"]
    )
    df["value_loss_barrier"] = (df["max_value_loss"] - df["value_loss"]) / (
        df["max_value_loss"] - df["min_value_loss"]
    )
    return df


def visualize_model_range(df, separator, plot_column, include_min_max=False):
    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Unique separators and color mapping
    separators = df[separator].unique()
    cmap = plt.get_cmap("tab10")
    colors = {sep: cmap(i) for i, sep in enumerate(separators)}

    # Plot shading and lines
    for sep in separators:
        subset = df[df[separator] == sep]
        episodes = sorted(subset["Episodes"].unique())
        # prepare lists for fill_between
        mins = [
            subset[subset["Episodes"] == ep]["min_" + plot_column].iloc[0]
            for ep in episodes
        ]
        maxs = [
            subset[subset["Episodes"] == ep]["max_" + plot_column].iloc[0]
            for ep in episodes
        ]
        ax.fill_between(
            episodes,
            mins,
            maxs,
            color=colors[sep],
            alpha=0.2,
            label=f"{sep} {plot_column} interval",
        )
        if include_min_max:
            for alpha, style in [(0, "-"), (1, "--")]:
                alpha_subset = subset[subset["alpha"] == alpha]
                ax.plot(
                    alpha_subset["Episodes"],
                    alpha_subset[plot_column],
                    linestyle=style,
                    color=colors[sep],
                    label=f"{sep} α={alpha}",
                )
    return fig, ax
