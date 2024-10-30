from typing import cast

import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Create a 2-hot vector of size n, with a 1 in i1, and a -1 in i2
def match_two_hot(i1: int, i2: int, n: int) -> np.ndarray:
    x = np.zeros(n)
    x[i1] = 1
    x[i2] = -1
    return x

def score_from_results(results):
    return (results[0] - results[2]) / (sum(results) + 1)

# Data should be a symmetric 2d dictionary
def compute_elo(data, wls=False, partial_model=None, partial_n_games=1, std=None, average=1000):
    if std is None:
        std = wls

    players = list(data.keys())
    player_id = {
        players[i]: i
        for i in range(len(players))
    }

    extra_X, extra_r_bar, extra_w = [], [], []

    if partial_model is not None:
        rs = np.random.RandomState(seed=list(map(ord, partial_model)))

        matches = [
            (player1, player2)
            for player1 in players
            for player2 in data[player1].keys()
            if (
                player1 < player2
                and partial_model not in player1
                and partial_model not in player2
            )
        ]

        # Linear search to add exactly partial_n_games games to the matches
        # played by each iteration of partial_model
        for player1 in players:
            if partial_model not in player1:
                continue
            cnt = 0
            for player2 in data[player1].keys():
                cur = sum(data[player1][player2])
                if cnt + cur < partial_n_games:
                    cnt += cur
                    matches.append((player1, player2))
                    continue
                extra_X.append(match_two_hot(
                    player_id[player1],
                    player_id[player2],
                    len(players)
                ))
                # Generate a random order over the games played in order to
                # randomly select $partial_n_games - cnt$ games
                res = (
                      [0] * data[player1][player2][0]
                    + [1] * data[player1][player2][1]
                    + [2] * data[player1][player2][2]
                )
                rs.shuffle(res)
                res_cpt = [0, 0, 0]
                for r in res[:partial_n_games - cnt]:
                    res_cpt[r] += 1
                extra_r_bar.append(score_from_results(res_cpt))
                extra_w.append(partial_n_games - cnt)
                break
    else:
        matches = [
            (player1, player2)
            for player1 in players
            for player2 in data[player1].keys()
            if player1 < player2
        ]

    X = np.array([
        match_two_hot(player_id[player1], player_id[player2], len(players))
        for player1, player2 in matches
    ] + extra_X +  [
        [1] * len(players)
    ])
    r_bar = np.array([
        score_from_results(data[player1][player2])
        for player1, player2 in matches
    ] + extra_r_bar)
    p_bar = (r_bar + 1) / 2
    y = np.concatenate([
        -(400 / np.log(10)) * np.log(1 / p_bar - 1),
        [average * len(players)] # We set the average elo to 1000
    ])
    w = np.concatenate([
        [sum(data[player1][player2]) for player1, player2 in matches],
        extra_w,
        [1]
    ])

    std_deviations = None
    if wls:
        var = (400 / np.log(10)) ** 2 / (w[:-1] * p_bar * (1-p_bar))
        var = np.concatenate([var, [1e-7]])

        mod_wls = sm.WLS(y, X, weights=cast(float, 1.0 / var))
        res_wls = mod_wls.fit()
        elo = res_wls.params
        std_deviations = res_wls.bse
    else:
        reg = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=w)
        elo = reg.coef_

        if std:
            # EXPERIMENTAL
            # Calculate residuals and standard errors
            residuals = y - reg.predict(X)
            residual_sum_of_squares = (residuals ** 2).sum()
            df_residuals = X.shape[0] - X.shape[1]  # degrees of freedom

            # Estimate variance of residuals
            sigma_squared = residual_sum_of_squares / df_residuals
            XtX_inv = np.linalg.inv(X.T @ X)

            # Standard errors for each coefficient
            std_deviations = np.sqrt(np.diag(sigma_squared * XtX_inv))

    elos = dict(sorted({
        player: int(elo[i]+0.5)
        for i, player in enumerate(players)
    }.items()))
    if std:
        assert(std_deviations is not None)
        std_deviations = dict(sorted({
            player: std_deviations[i]
            for i, player in enumerate(players)
        }.items()))
        return elos, std_deviations
    return elos
