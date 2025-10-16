# Points - Assignment for R&D Senior Data Scientist
## Multi-Armed Bandit / Contextual Bandit Simulation

1. Optional
    * Setup a virtual environment: `python -m venv .venv`
    * Activate new environment: `source .venv/bin/activate`
    * This code was tested on python 3.9

2. Install requirements: `pip install -r requirements.txt`

3. Run main app using an IDE or CLI via `python thompson_bandit.py config.json`
    * Main app runs the following scenarios once:
      * Basic Thompson Sampling
      * Contextual Thompson Sampling
      * Contextual Thompson Sampling with SMOTE
      * Perfect Thompson Sampling

Note:
For convenience, the CLI args are preloaded in the `config.json` file with the following format:

```
{
  "csv": "DATA_MULTI_ARM_BANDIT.csv",
  "arms": [0.40, 0.45, 0.50],
  "context_cols": [
    "FLAG_FIRST_TIME_VISITOR",
    "FLAG_FIRST_TIME_BUYER",
    "CURRENT_BALANCE",
    "DAYS_SINCE_LAST_PURCHASE_L12M",
    "COUNT_TRANX_L12M",
    "DAYS_SINCE_LAST_VISIT_NO_PURCHASE",
    "LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M",
    "OFFER_RICHNESS_APPLIED_ON_LAST_PURCHASE_L12M",
    "POINTS_PURCHASED_LAST_TRANX_L12M"
  ],
  "min_ppp": 0.016,
  "points_per_txn": 3000,
  "min_obs": 20,
  "ppp_margin": 0.0001,
  "out_dir": "output"
}
```
Feel free to change these parameters.
The main app runs the simulation once, and since TS is a probabilistic model using a 
$\beta$ distribution, the results will very slightly every
run. The code will store the generated figures and raw simulation results in the output directory e.g.:

<img alt="simulation_results.png" src="simulation_results.png" width="600"/>

For a more comprehensive "Monte-Carlo" style simulation check out the jupyter notebook `analysis.ipynb`. 
A simulation of each scenario was run 50 times and the distributions of the total revenues are
overlayed. It shows how the SMOTE contextual TS model beats the baseline. 
A perfect model knowing the transactions apriori is also compared:

<img alt="monte_carlo_simulation.png" src="monte_carlo_simulation.png" width="600"/>

A/B testing can be easily performed to see if these results are statistically significant.

## Deep Dive

N.B. 
* Some terms are used interchangeably throughout this document. e.g. `offer ~ discount ~ arm` or `true ~ observed`
* The work here is assessed using offline replay evaluation.

### Why Thompson Sampling

Thompson Sampling is a Bayesian algorithm.
For each arm, it maintains a probability distribution over how good that arm is. At every time step:

   * Sample a reward probability for each arm from its current distribution.
   * Choose the arm with the highest sampled value.
   * Observe the actual reward.
   * Update that arm’s distribution based on the result (success/failure).

This naturally balances:
   * Exploration → Sampling favors arms with high uncertainty.
   * Exploitation → Arms that perform well are likely to keep getting selected.

If each arm gives a binary reward (0 or 1), 
we model the arm's success rate with a beta distribution, 
which is a natural choice for probabilities:
* For arm i, maintain:
     * $\alpha(i)$: number of successes + 1 (prior)
     * $\beta(i)$: number of failures + 1 (prior)

Say you have 3 discount offers (arms): 40%, 45%, 50%,
each time a user visits:
   * You sample a conversion rate for each offer from its Beta distribution 
   * You pick the offer with the highest sampled rate 
   * If the user buys → success → update $\alpha$
   * If not → failure → update $\beta$

Thompson Sampling explores uncertain options early on because their distributions are wide. 
As you gather data, the distributions narrow around the true conversion rates, 
and the algorithm naturally exploits the best arms more often.

<img alt="beta_distribution.png" src="beta_distribution.png" title="image from wikipedia"/>

I dabbled a bit with $\epsilon$-greedy models, but not knowing what the optimal $\epsilon$ adds a new hyperparameter
to be explored. The results were also not very promising (we can discuss this in person later).


### PPP Constraint

How to keep PPP ≥ 0.016?

   * Look-ahead: Before we commit to an arm, we estimate what would happen if the member buys (using the model’s conversion probability).
   * Filter: We drop any arm that would drive the projected average below 0.016. 
   * Fallback: If every arm fails the check, we force the safest one (40%).
   * Replay consistency: Because we’re doing offline replay, if the historical impression did not serve the fallback arm, we simply skip that impression. The constraint is therefore never violated in the running totals we actually update.

### Basic Thompson Sampling (Naive)

In basic Thompson Sampling, the probability of a user purchasing points at a given offer richness is estimated directly from the observed data. 
For example, the probability of conversion at a 50% discount is computed as:

$p(50\%) = \frac{\text{clicks at }50\%}{\text{impressions at }50\%}$

In other words, it corresponds to the click-through rate (CTR) for each specific offer. 
Over time, this approach converges to the true population level conversion rates for each offer arm.
While not a great solution, it provides an excellent baseline to compare more sophisticated models against.

### Contextual Thompson Sampling

Contextual Thompson Sampling improves upon this by incorporating 
user- or session-specific features to estimate the probability of conversion using logistic regression:

```
prob = models[arm].predict_proba(ctx)[0, 1]
```

A separate logistic regression model is trained for each offer arm. 
The model outputs the probability that a user will convert, based on contextual features, 
as estimated by the sigmoid of the learned logits. The model used in this code is simple logistical
regression to run efficiently given the limited number of features. In a real production code, we should make take
advantage of neural networks which can extract non-linear relationships between
the complex features and the targets.

The contextual features are the ones listed in the `config.json`. The dataset was nicely formatted and didn't require
any preprocessing. For instance, if the user didn't purchase any points the value for `DAYS_SINCE_LAST_PURCHASE_L12M`
was set to 9999, which is typical when for such an application. The only transformation applied was scaling the 
feature set using a `StandardScaler()`

The model starts generating a probability that a user will purchase points given that a specific offer has been
presented to them, then the expected reward is calculated as:

$E(reward|offer) = p(buy|offer) \times \text{avg points per txn} \times \text{PPP from discount}$

The expectation value is a salient feature of this simulation, lets say a user has a 90% chance of buying
at a 50% discount, but only 15% chance of buying at a 40% discount; despite in reality making more money on 40% discounts,
we should present the 50% discount due to the high likelyhood of the user walking away if we don't → $E(reward|0.5) > E(reward|0.4)$.
It's the same concept implemented in ad ranking systems

The projected PPP is then calculated and is compared to them minimum constraint we set, to make sure we're not going over budget:

```
if projected_ppp(cum_revenue, cum_points, expected_revenue, expected_points) >= self.min_ppp + self.ppp_margin:
    candidate_arms.append(arm)
```

Finally, we perform a random draw using the TS method discussed earlier:

```
# Thompson draw & arm selection
sampled_rewards = {}
for arm in candidate_arms:
    theta = np.random.beta(alpha[arm], beta[arm])    # conversion sample
    sampled_rewards[arm] = theta * self.points_per_txn * ppp_from_discount(arm)

chosen_arm = max(sampled_rewards, key=sampled_rewards.get)
```

This brings us to the elephant in the room, any ML model would face a significant challenge that the dataset is highly imbalanced.

   * `offer_richness_applied = {0.5: 1178, 0.45: 169, 0.4: 24}`
   * `offer_richness_served = {0.5: 7705, 0.45: 1820, 0.4: 475}`
   * `true_conversion_rates = {0.5: 0.15003, 0.45: 0.09120, 0.4: 0.05052}`

The imbalance is twofold:

   * Class imbalance → More generous discounts (e.g. 50%) are overrepresented in the served data.
   * Outcome imbalance  Even within classes, positive conversion outcomes are relatively rare, reflecting a low overall CTR.

While the true CTRs confirm that users tend to respond more positively to higher discounts (15% CTR at 50%), 
the skewed data distribution poses a challenge for training reliable models. 
Without addressing the imbalance, the contextual probability estimates are likely to be biased and underperform, 
especially for underrepresented arms.

### Contextual Thompson Sampling with Oversampling

Oversampling the minority class is a classic technique to overcome class imbalance. SMOTE is a popular technique that 
I've used here, but everything else implemented in contextual TS still applies. 
By default, the SMOTE class balances a binary dataset to 50% positive and 50% negative (implemented here). This is done by 
generating synthetic data points from known positive classes using the kNN method. I applied some guardrails to ensure that
the SMOTE method does not produce any errors:

```
if len(y_mem[chosen_arm]) >= self.min_obs and \
        len(y_mem[chosen_arm]) % self.refit_every == 0 and \
        0 in y_mem[chosen_arm] and \
        1 in y_mem[chosen_arm] and \
        Counter(y_mem[chosen_arm])[1] >= 2:  # make sure we have at least 2 positive samples for kNN
    # Oversample minority class using SMOTE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_mem[chosen_arm], y_mem[chosen_arm])
```

This way the model will better learns to predict if a user will buy by learning from more positive samples. However, we
don't really care much about the classification itself as much as we do about the probability generated of the learned logits.
Since, the distribution of the training set is now 50/50, the probability is not calibrated to the true distribution of the
dataset seen so far. We need to then calibrate probability by the observed $p$.

$\text{calibrated p} = \frac{\text{predicted p}}{\text{observed p}}$

```
prob = models[arm].predict_proba(ctx)[0, 1]
calibration = alpha[arm] / (alpha[arm] + beta[arm])
prob = prob / calibration
```

Instead of keeping track of the observed distribution, I used the alpha and beta dicts, which technically has an additional
prior 1 added to each arm. Not quite the observed distribution!

On a slightly related note, we are applying reinforcement learning here meaning that the model is trained
every time we see new information, the frequency of retraining is set by `refit_every = 1`.

That's it! Despite having a very limited dataset, this method can actually improve the maximum promotion revenue quite significantly.


### Perfect Thompson Sampling

This method is used to generate an upper baseline. It makes the naive assumption that if a user buys points at a 
specific discount $p(buy|offer) = 100\%$, otherwise, all other arms will be assigned a probability of $0\%$. We also use the
exact amount of points they buy instead of an average amount.
This means the model has perfect foresight of what's about to happen, but it's also naive in the sense that it
assumes the user will never buy any other offers not presented to them.

### Q&A

1. How would you modify your approach if member behavior changes over time?

Great question! in this exercise I used an expanding window to train a model for the separate arms, meaning that the 
training set will increase in size over time. If member behavior changes this will lead to a drift in model predictions.
A simple remedy is applying a rolling fixed window in the RL training loop, this ensures that old data is periodically
thrown out. That said, it is sometimes not wise to throw data away if it can lead to better performance.
For this, more sophisticated methods to detect drift can be applied, then we adjust the sizing of the training window accordingly.
Such methods include statistical tests to compare distributions of old vs new data such as the Kolmogorov-Smirnov test.
Another method is model-based drift detection by tracking the model performance on new data with old data.

2. How would your algorithm adapt if the business objective shifted mid-promotion?

I think removing the PPP constraint is a first good step, this ensures that we don't throw away potential clicks
if we are projected to go over budget. below is a simulation result with the constrained removed. Notice how the
number of successful arm hits increases significantly (the model also trains on more samples):

<img alt="no_constraint.png" src="no_constraint.png" width="400"/>


