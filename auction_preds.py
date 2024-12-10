import numpy as np

def budget_pacing_ucb_ctr(num_agents, budget, time_horizon, step_size, upper_bound, valuations):
    pacing_multipliers = np.zeros(num_agents)
    ctr_estimates = np.zeros(num_agents)
    click_counts = np.zeros(num_agents)
    impressions = np.ones(num_agents) 
    budgets_remaining = budget.copy()
    bids = np.zeros(num_agents)
    results = []

    for t in range(time_horizon):
        ctr_estimates = click_counts / impressions + np.sqrt(3 * np.log(time_horizon) / (2 * impressions))

        for k in range(num_agents):
            adjusted_value = valuations[k] * ctr_estimates[k]
            bids[k] = min(adjusted_value / (1 + pacing_multipliers[k]), budgets_remaining[k])

        sorted_agents = np.argsort(bids * ctr_estimates)[::-1]
        winner = sorted_agents[0]
        second_highest_bid = bids[sorted_agents[1]] * ctr_estimates[sorted_agents[1]]

        payment = second_highest_bid / ctr_estimates[winner] if ctr_estimates[winner] > 0 else 0
        budgets_remaining[winner] -= payment
        results.append((t, winner, payment))

        for k in range(num_agents):
            pacing_multipliers[k] = max(0, pacing_multipliers[k] - step_size * (budget[k] / time_horizon - payment))
            impressions[k] += 1
            if k == winner:
                click_counts[k] += 1

    return results


num_agents = 3
budget = np.array([100, 150, 120])
time_horizon = 50
step_size = 0.01
upper_bound = 10
valuations = np.array([0.5, 0.8, 0.6])

results = budget_pacing_ucb_ctr(num_agents, budget, time_horizon, step_size, upper_bound, valuations)

import matplotlib.pyplot as plt

times, winners, payments = zip(*results)

plt.figure(figsize=(8, 6))
plt.plot(times, payments, label="Payments", marker='o')
plt.xlabel("Time")
plt.ylabel("Payments")
plt.title("Auction Payments Over Time")
plt.legend()
plt.grid(True)

plt.savefig("test.png", format="png", dpi=300)
plt.show()

