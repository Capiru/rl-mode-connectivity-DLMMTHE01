from rl.utils import get_model


def interpolate_models(model_1, model_2, alpha: float = 0.0):
    params_model_1 = [param.clone() for param in model_1.parameters()]
    params_model_2 = [param.clone() for param in model_2.parameters()]

    # Make sure models have the same architecture
    assert isinstance(model_1, type(model_2))
    interpolated_model = get_model(model_type=model_1.cfg.model_type, cfg=model_1.cfg)
    params_interpolated = [
        alpha * param1 + (1 - alpha) * param2
        for param1, param2 in zip(params_model_1, params_model_2)
    ]
    assert len(params_interpolated) == len(list(interpolated_model.parameters()))
    for param in interpolated_model.parameters():
        param.data = params_interpolated.pop(0)
    return interpolated_model


if __name__ == "__main__":
    from rl.utils import get_model
    from rl.game import eval_tournament
    from rl.config import Settings
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    cfg = Settings()
    model_1 = get_model(cfg=cfg)
    model_1.load_state_dict(torch.load("model_48_elo_622.0.pth"))
    model_2 = get_model()
    model_2.load_state_dict(torch.load("model_420_elo_449.0.pth"))

    n_points = 10
    alpha_space = np.linspace(0.0, 1.0, n_points)
    plot_df = pd.DataFrame(columns=["alpha", "elo"])
    for i in range(n_points):
        interpolated_model = interpolate_models(model_1, model_2, alpha_space[i])
        model_elo, outcomes = eval_tournament(
            models_df=pd.read_csv("data/num_sims_20/models.csv"),
            model=interpolated_model,
            iteration=0,
        )
        new_eval = pd.DataFrame({"alpha": [alpha_space[i]], "elo": [model_elo]})
        plot_df = pd.concat([plot_df, new_eval])
    plot_df.plot(x="alpha", y="elo", kind="line")
    plt.show()
