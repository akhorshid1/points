#!/usr/bin/env python3
"""
Contextual Thompson-Sampling bandit with SMOTE + plot saving.
Usage: python oversampled_bandit.py config.json
"""

import argparse, json, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Helper functions
def ppp_from_discount(discount):
    return 0.03 * (1 - discount)

def expected_reward(prob_buy, points_per_transaction, discount):
    return prob_buy * points_per_transaction * ppp_from_discount(discount)

def projected_ppp(revenue, points, add_revenue, add_points):
    new_points = points + add_points
    if new_points == 0:  # no successful tx yet
        return ppp_from_discount(0.40)  # safe upper bound
    return (revenue + add_revenue) / new_points

class ThompsonSamplingBandit:
    def __init__(self,
                 *,
                 arms,
                 context_context_cols,
                 min_ppp,
                 points_per_txn,
                 min_obs,
                 init_alpha,
                 init_beta,
                 ppp_margin,
                 refit_every,
                 seed=42):

        self.arms = arms
        self.context_cols = context_context_cols
        self.min_ppp = min_ppp
        self.points_per_txn = points_per_txn
        self.min_obs = min_obs
        self.ppp_margin = ppp_margin
        self.seed = seed
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.refit_every = refit_every

    def oversampled_contextual_bandit(self, df):
        width = len(self.context_cols)

        # THOMPSON-SAMPLING STATE

        models = {arm: LogisticRegression() for arm in self.arms}
        X_mem = {arm: np.empty((0, width)) for arm in self.arms}  # buffered feature rows
        y_mem = {arm: np.array([]) for arm in self.arms}  # buffered labels

        alpha = {arm: self.init_alpha for arm in self.arms}  # successes + prior
        beta = {arm: self.init_beta for arm in self.arms}  # failures + prior

        arm_hits = defaultdict(int)
        arm_hits_revenue = defaultdict(int)

        cum_revenue = 0.0
        cum_points = 0.0
        revenue_curve = []
        ppp_curve = []

        smote = SMOTE(k_neighbors=1)

        # OFFLINE-REPLAY LOOP
        for _, row in df.iterrows():
            ctx = row[self.context_cols].values.reshape(1, -1)
            # build candidate list that passes PPP constraint
            candidate_arms = []

            for arm in self.arms:
                # predict probability that a user will buy offer
                # use Jeffreys/Beta (0.5, 0.5) prior if model is not ready
                if len(X_mem[arm]) < self.min_obs:
                    prob = 0.5
                else:
                    try:
                        prob = models[arm].predict_proba(ctx)[0, 1]
                        # because we are oversampling we want to calibrate the
                        # model output to the true distribution of the observed arm
                        calibration = alpha[arm] / (alpha[arm] + beta[arm])
                        prob = prob / (prob + (1 - prob) * calibration)

                    except NotFittedError:
                        prob = 0.5

                expected_revenue = expected_reward(prob, self.points_per_txn, arm)
                expected_points = prob * self.points_per_txn

                if projected_ppp(cum_revenue, cum_points, expected_revenue, expected_points) >= self.min_ppp + self.ppp_margin:
                    candidate_arms.append(arm)

            if not candidate_arms:
                candidate_arms = [0.40]

            # Thompson draw & arm selection
            sampled_rewards = {}
            for arm in candidate_arms:
                theta = np.random.beta(alpha[arm], beta[arm])  # conversion sample
                sampled_rewards[arm] = theta * self.points_per_txn * ppp_from_discount(arm)

            chosen_arm = max(sampled_rewards, key=sampled_rewards.get)

            # OFFLINE REPLAY FILTER
            if chosen_arm != row["OFFER_RICHNESS_SERVED"]:
                continue  # we don't observe counterfactual

            # Observe outcome & update posterior
            success = int(row["FLAG_TRANSACTION"] == 1)
            if success:
                alpha[chosen_arm] += 1
                revenue = row["POINTS_PURCHASED"] * row["PRICE_PER_POINT"]
                points = row["POINTS_PURCHASED"]
                arm_hits_revenue[chosen_arm] += 1
            else:
                beta[chosen_arm] += 1
                revenue = 0.0
                points = 0.0

            # running totals and logs
            cum_revenue += revenue
            cum_points += points
            revenue_curve.append(cum_revenue)
            ppp_curve.append(cum_revenue / cum_points if cum_points else ppp_from_discount(0.40))
            arm_hits[chosen_arm] += 1

            # store experience
            X_mem[chosen_arm] = np.vstack([X_mem[chosen_arm], ctx.flatten()])
            y_mem[chosen_arm] = np.append(y_mem[chosen_arm], int(row['FLAG_TRANSACTION']))

            # periodic refit
            if len(y_mem[chosen_arm]) >= self.min_obs and \
                    len(y_mem[chosen_arm]) % self.refit_every == 0 and \
                    0 in y_mem[chosen_arm] and \
                    1 in y_mem[chosen_arm] and \
                    Counter(y_mem[chosen_arm])[1] >= 2:  # make sure we have at least 2 positive samples for kNN
                # Oversample minority class using SMOTE
                X_train_resampled, y_train_resampled = smote.fit_resample(X_mem[chosen_arm], y_mem[chosen_arm])

                models[chosen_arm].fit(X_train_resampled, y_train_resampled)

        return revenue_curve, ppp_curve, arm_hits, arm_hits_revenue, alpha, beta


    def contextual_bandit(self, df):
        width = len(self.context_cols)

        # THOMPSON-SAMPLING STATE

        models = {arm: LogisticRegression() for arm in self.arms}
        X_mem = {arm: np.empty((0, width)) for arm in self.arms}  # buffered feature rows
        y_mem = {arm: np.array([]) for arm in self.arms}  # buffered labels

        alpha = {arm: self.init_alpha for arm in self.arms}  # successes + prior
        beta = {arm: self.init_beta for arm in self.arms}  # failures + prior

        arm_hits = defaultdict(int)
        arm_hits_revenue = defaultdict(int)

        cum_revenue = 0.0
        cum_points = 0.0
        revenue_curve = []
        ppp_curve = []

        # OFFLINE-REPLAY LOOP
        for _, row in df.iterrows():
            ctx = row[self.context_cols].values.reshape(1, -1)
            # build candidate list that passes PPP constraint
            candidate_arms = []

            for arm in self.arms:
                # predict probability that a user will buy offer
                # use Jeffreys/Beta (0.5, 0.5) prior if model is not ready
                if len(X_mem[arm]) < self.min_obs:
                    prob = 0.5
                else:
                    try:
                        prob = models[arm].predict_proba(ctx)[0, 1]

                    except NotFittedError:
                        prob = 0.5

                expected_revenue = expected_reward(prob, self.points_per_txn, arm)
                expected_points = prob * self.points_per_txn

                if projected_ppp(cum_revenue, cum_points, expected_revenue, expected_points) >= self.min_ppp + self.ppp_margin:
                    candidate_arms.append(arm)

            if not candidate_arms:
                candidate_arms = [0.40]  # fall back to safest arm

            # Thompson draw & arm selection
            sampled_rewards = {}
            for arm in candidate_arms:
                theta = np.random.beta(alpha[arm], beta[arm])  # conversion sample
                sampled_rewards[arm] = theta * self.points_per_txn * ppp_from_discount(arm)

            chosen_arm = max(sampled_rewards, key=sampled_rewards.get)

            # OFFLINE REPLAY FILTER
            if chosen_arm != row["OFFER_RICHNESS_SERVED"]:
                continue  # we don't observe counterfactual

            # Observe outcome & update posterior
            success = int(row["FLAG_TRANSACTION"] == 1)
            if success:
                alpha[chosen_arm] += 1
                revenue = row["POINTS_PURCHASED"] * row["PRICE_PER_POINT"]
                points = row["POINTS_PURCHASED"]
                arm_hits_revenue[chosen_arm] += 1
            else:
                beta[chosen_arm] += 1
                revenue = 0.0
                points = 0.0

            # running totals and logs
            cum_revenue += revenue
            cum_points += points
            revenue_curve.append(cum_revenue)
            ppp_curve.append(cum_revenue / cum_points if cum_points else ppp_from_discount(0.40))
            arm_hits[chosen_arm] += 1

            # store experience
            X_mem[chosen_arm] = np.vstack([X_mem[chosen_arm], ctx.flatten()])
            y_mem[chosen_arm] = np.append(y_mem[chosen_arm], int(row['FLAG_TRANSACTION']))

            # periodic refit
            if len(y_mem[chosen_arm]) >= self.min_obs and \
                    len(y_mem[chosen_arm]) % self.refit_every == 0 and \
                    0 in y_mem[chosen_arm]and \
                    1 in y_mem[chosen_arm]:
                models[chosen_arm].fit(X_mem[chosen_arm], y_mem[chosen_arm])

        return revenue_curve, ppp_curve, arm_hits, arm_hits_revenue, alpha, beta

    def basic_bandit(self, df):

        # THOMPSON-SAMPLING STATE
        arm_hits = defaultdict(int)
        arm_hits_revenue = defaultdict(int)

        alpha = {arm: self.init_alpha for arm in self.arms}  # successes + prior
        beta = {arm: self.init_beta for arm in self.arms}  # failures + prior

        cum_revenue = 0.0
        cum_points = 0.0
        revenue_curve = []
        ppp_curve = []
        # OFFLINE-REPLAY LOOP
        for _, row in df.iterrows():

            # build candidate list that passes PPP constraint
            candidate_arms = []
            for arm in self.arms:
                prob = alpha[arm] / (alpha[arm] + beta[arm])   # probability to buy
                expected_revenue = expected_reward(prob, self.points_per_txn, arm)
                expected_points = prob * self.points_per_txn
                if projected_ppp(cum_revenue, cum_points, expected_revenue, expected_points) >= self.min_ppp + self.ppp_margin:
                    candidate_arms.append(arm)

            if not candidate_arms:
                candidate_arms = [0.40]  # fall back to safest arm

            # Thompson draw & arm selection
            sampled_rewards = {}
            for arm in candidate_arms:
                theta = np.random.beta(alpha[arm], beta[arm])    # conversion sample
                sampled_rewards[arm] = theta * self.points_per_txn * ppp_from_discount(arm)

            chosen_arm = max(sampled_rewards, key=sampled_rewards.get)

            # OFFLINE REPLAY FILTER
            if chosen_arm != row["OFFER_RICHNESS_SERVED"]:
                continue  # we don't observe counterfactual, skip the rest

            # Observe outcome & update posterior
            success = int(row["FLAG_TRANSACTION"] == 1)
            if success:
                alpha[chosen_arm] += 1
                revenue = row["POINTS_PURCHASED"] * row["PRICE_PER_POINT"]
                points  = row["POINTS_PURCHASED"]
                arm_hits_revenue[chosen_arm] += 1
            else:
                beta[chosen_arm] += 1
                revenue = 0.0
                points  = 0.0

            # running totals and logs
            cum_revenue += revenue
            cum_points += points
            revenue_curve.append(cum_revenue)
            ppp_curve.append(cum_revenue / cum_points if cum_points else ppp_from_discount(0.40))
            arm_hits[chosen_arm] += 1

        return revenue_curve, ppp_curve, arm_hits, arm_hits_revenue, alpha, beta

    def perfect_bandit(self, df):
        # THOMPSON-SAMPLING STATE
        arm_hits = defaultdict(int)
        arm_hits_revenue = defaultdict(int)

        alpha = {arm: self.init_alpha for arm in self.arms}  # successes + prior
        beta = {arm: self.init_beta for arm in self.arms}  # failures + prior

        cum_revenue = 0.0
        cum_points = 0.0
        revenue_curve = []
        ppp_curve = []
        # OFFLINE-REPLAY LOOP
        for _, row in df.iterrows():

            # build candidate list that passes PPP constraint
            candidate_arms = []
            for arm in self.arms:
                # Perfect Foresight, it makes the naive assumption that the probability a user buys
                # at the offer_richness_served is exactly 100%, otherwise it's 0%
                if arm == row["OFFER_RICHNESS_SERVED"]:
                    prob = row['FLAG_TRANSACTION']  # probability to buy
                else:
                    prob = 0
                expected_revenue = expected_reward(prob, row['POINTS_PURCHASED'], arm)
                expected_points = prob * row['POINTS_PURCHASED']
                if projected_ppp(cum_revenue, cum_points, expected_revenue, expected_points) >= self.min_ppp + self.ppp_margin:
                    candidate_arms.append(arm)

            if not candidate_arms:
                candidate_arms = [0.40]  # fall back to safest arm

            # Thompson draw & arm selection
            sampled_rewards = {}
            for arm in candidate_arms:
                theta = np.random.beta(alpha[arm], beta[arm])    # conversion sample
                sampled_rewards[arm] = theta * self.points_per_txn * ppp_from_discount(arm)

            chosen_arm = max(sampled_rewards, key=sampled_rewards.get)

            # OFFLINE REPLAY FILTER
            if chosen_arm != row["OFFER_RICHNESS_SERVED"]:
                continue  # we don't observe counterfactual, skip the rest

            # Observe outcome & update posterior
            success = int(row["FLAG_TRANSACTION"] == 1)
            if success:
                alpha[chosen_arm] += 1
                revenue = row["POINTS_PURCHASED"] * row["PRICE_PER_POINT"]
                points  = row["POINTS_PURCHASED"]
                arm_hits_revenue[chosen_arm] += 1
            else:
                beta[chosen_arm] += 1
                revenue = 0.0
                points  = 0.0

            # running totals and logs
            cum_revenue += revenue
            cum_points += points
            revenue_curve.append(cum_revenue)
            ppp_curve.append(cum_revenue / cum_points if cum_points else ppp_from_discount(0.40))
            arm_hits[chosen_arm] += 1

        return revenue_curve, ppp_curve, arm_hits, arm_hits_revenue, alpha, beta

    def oversampled_contextual_epsilon(self, df):
        width = len(self.context_cols)

        # THOMPSON-SAMPLING STATE

        models = {arm: LogisticRegression() for arm in self.arms}
        X_mem = {arm: np.empty((0, width)) for arm in self.arms}  # buffered feature rows
        y_mem = {arm: np.array([]) for arm in self.arms}  # buffered labels

        alpha = {arm: self.init_alpha for arm in self.arms}  # successes + prior
        beta = {arm: self.init_beta for arm in self.arms}  # failures + prior

        arm_hits = defaultdict(int)
        arm_hits_revenue = defaultdict(int)

        cum_revenue = 0.0
        cum_points = 0.0
        revenue_curve = []
        ppp_curve = []

        smote = SMOTE(k_neighbors=1)

        # OFFLINE-REPLAY LOOP
        for _, row in df.iterrows():
            ctx = row[self.context_cols].values.reshape(1, -1)
            # build candidate list that passes PPP constraint
            candidate_arms = []

            for arm in self.arms:
                # predict probability or use random score before model is ready

                if len(X_mem[arm]) < self.min_obs:
                    prob = 0.5
                else:
                    try:
                        prob = models[arm].predict_proba(ctx)[0, 1]
                        calibration = alpha[arm] / (alpha[arm] + beta[arm])
                        prob = prob / (prob + (1 - prob) * calibration)
                    except NotFittedError:
                        prob = 0.5

                expected_revenue = expected_reward(prob, self.points_per_txn, arm)
                expected_points = prob * self.points_per_txn

                if projected_ppp(cum_revenue, cum_points, expected_revenue, expected_points) >= self.min_ppp + self.ppp_margin:
                    candidate_arms.append((arm, expected_revenue))

            # guarantee feasibility: if none safe, fall back to 0.40
            if not candidate_arms:
                candidate_arms = [(0.40, 0)]  # reward unused

            epsilon = 0.1
            # -------- choose arm (ε-greedy on expected reward) ----------------------
            if random.random() < epsilon:

                chosen_arm = random.choice([a for a, _ in candidate_arms])
            else:

                chosen_arm = max(candidate_arms, key=lambda x: x[1])[0]

            # OFFLINE REPLAY FILTER
            if chosen_arm != row["OFFER_RICHNESS_SERVED"]:
                continue  # we don't observe counterfactual

            # Observe outcome & update posterior
            success = int(row["FLAG_TRANSACTION"] == 1)
            if success:
                alpha[chosen_arm] += 1
                revenue = row["POINTS_PURCHASED"] * row["PRICE_PER_POINT"]
                points = row["POINTS_PURCHASED"]
                arm_hits_revenue[chosen_arm] += 1
            else:
                beta[chosen_arm] += 1
                revenue = 0.0
                points = 0.0

            # running totals and logs
            cum_revenue += revenue
            cum_points += points
            revenue_curve.append(cum_revenue)
            ppp_curve.append(cum_revenue / cum_points if cum_points else ppp_from_discount(0.40))
            arm_hits[chosen_arm] += 1

            # store experience
            X_mem[chosen_arm] = np.vstack([X_mem[chosen_arm], ctx.flatten()])
            y_mem[chosen_arm] = np.append(y_mem[chosen_arm], int(row['FLAG_TRANSACTION']))

            # periodic refit
            if len(y_mem[chosen_arm]) >= self.min_obs and \
                    len(y_mem[chosen_arm]) % self.refit_every == 0 and \
                    0 in y_mem[chosen_arm] and \
                    1 in y_mem[chosen_arm] and \
                    Counter(y_mem[chosen_arm])[1] >= 2:  # make sure we have at least 2 positive samples for kNN
                # Oversample minority class using SMOTE
                X_train_resampled, y_train_resampled = smote.fit_resample(X_mem[chosen_arm], y_mem[chosen_arm])

                models[chosen_arm].fit(X_train_resampled, y_train_resampled)

        return revenue_curve, ppp_curve, arm_hits, arm_hits_revenue, alpha, beta



def save_plots(results: dict, out_dir: Path, min_ppp: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    scenarios = list(results.keys())
    arms = sorted(set(a for res in results.values() for a in res["arm_hits"].keys()))
    n_scenarios = len(scenarios)

    colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_scenarios))  # one color per scenario

    for idx, scenario in enumerate(scenarios):
        res = results[scenario]
        col = colors[idx]

        # Cumulative Revenue
        axs[0].plot(res["revenue_curve"], label=scenario, color=col)

        # Running PPP
        axs[1].plot(res["ppp_curve"], label=scenario, color=col)

    # Bar plot
    bar_width = 0.8 / n_scenarios
    group_positions = np.arange(len(arms))

    for idx, scenario in enumerate(scenarios):
        res = results[scenario]
        col = colors[idx]
        hits = {a: res["arm_hits"].get(a, 0) for a in arms}
        values = [hits[a] for a in arms]
        offsets = group_positions + (idx - n_scenarios/2) * bar_width + bar_width / 2
        axs[2].bar(offsets, values, width=bar_width, label=scenario, color=col)

    axs[0].set_title("Cumulative Revenue (Thompson Sampling)")
    axs[0].set_ylabel("Revenue ($)")
    axs[0].set_xlabel("Arm hits matching offer served")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title("Running PPP")
    axs[1].axhline(min_ppp, color="black", ls="--", label=f"PPP Constraint ({min_ppp})")
    axs[0].set_ylabel("PPP ($)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].set_title("Arm Selections")
    axs[2].set_ylabel("Count")
    axs[2].set_xlabel("Arm (Discount %)")
    axs[2].set_xticks(group_positions)
    axs[2].set_xticklabels([f"{int(a * 100)}%" for a in arms])
    axs[2].legend()
    axs[2].grid(axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "simulation_results.png", dpi=150)
    plt.close()


def main(cfg_path: Path):
    cfg = json.loads(Path(cfg_path).read_text())

    df = pd.read_csv(cfg["csv"])
    # We assume that the dataset has been rescaled before and
    # we are just transforming it using a previously fit scaler
    scaler = StandardScaler()
    df[cfg["context_cols"]] = scaler.fit_transform(df[cfg["context_cols"]])

    results = {}

    bandit = ThompsonSamplingBandit(
        arms = cfg["arms"],
        context_context_cols = cfg["context_cols"],
        min_ppp = cfg["min_ppp"],
        points_per_txn = cfg["points_per_txn"],
        min_obs = cfg["min_obs"],
        init_alpha = 1,
        init_beta = 1,
        ppp_margin = cfg["ppp_margin"],
        refit_every = 1
    )

    scenarios = [
        'basic_bandit',
        'contextual_bandit',
        'oversampled_contextual_bandit',
        'perfect_bandit',
        'oversampled_contextual_epsilon'
    ]

    out_dir = Path(cfg.get("out_dir", "bandit_outputs"))

    for scenario in scenarios:
        print(f"\n=== Running scenario: {scenario} ===\n")
        revenue_curve, ppp_curve, arm_hits, arm_hits_revenue, alpha, beta = eval(f"bandit.{scenario}(df.copy())")
        results[scenario] = {'revenue_curve': revenue_curve, 'ppp_curve': ppp_curve, 'arm_hits': arm_hits}
        print(f"arm hits matching offer served = {dict(sorted(arm_hits.items()))}")
        print(f"arm hits with revenue = {dict(sorted(arm_hits_revenue.items()))}")
        print(f"total hits = {sum(list(arm_hits.values()))}")
        print(f"total hits with revenue = {sum(list(arm_hits_revenue.values()))}")
        print(f"alpha = {alpha}")
        print(f"beta = {beta}")
        ratio = {x: alpha[x] / (alpha[x] + beta[x]) for x in alpha.keys()}
        print(f"conversion rates = {ratio}")
        print(f"total revenue = {revenue_curve[-1]}")
        curves = pd.DataFrame({"revenue": revenue_curve, "ppp": ppp_curve})
        curves.to_csv(out_dir / f"{scenario}_curves.csv", index=False)
        print("\n=== RUN COMPLETE ===")

    save_plots(results, out_dir, cfg["min_ppp"])
    print(f"Figures saved: {out_dir / 'simulation_results.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Oversampled TS bandit from JSON config")
    parser.add_argument("config", type=Path, help="Path to JSON config file")
    main(parser.parse_args().config)


