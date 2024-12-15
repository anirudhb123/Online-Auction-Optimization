import numpy as np
import matplotlib.pyplot as plt

class BudgetPacingUCBCTR:
    def __init__(self, num_agents, budgets, valuations, step_size, time_horizon, upper_bound):
        self.num_agents = num_agents
        self.budgets = np.array(budgets)
        self.valuations = np.array(valuations)
        self.step_size = step_size
        self.time_horizon = time_horizon
        self.upper_bound = upper_bound

        self.pacing_multipliers = np.zeros(self.num_agents)
        self.ctr_estimates = np.zeros(self.num_agents)
        self.click_counts = np.zeros(self.num_agents)
        self.impressions = np.ones(self.num_agents)
        self.remaining_budgets = self.budgets.copy()

    def run(self):
        bids = np.zeros(self.num_agents)
        results = []

        for t in range(self.time_horizon):
            self.ctr_estimates = (
                self.click_counts / self.impressions +
                np.sqrt(3 * np.log(self.time_horizon) / (2 * self.impressions))
            )

            for k in range(self.num_agents):
                adjusted_value = self.valuations[k] * self.ctr_estimates[k]
                bids[k] = min(
                    adjusted_value / (1 + self.pacing_multipliers[k]),
                    self.remaining_budgets[k]
                )

            sorted_agents = np.argsort(bids * self.ctr_estimates)[::-1]
            winner = sorted_agents[0]
            second_highest_bid = bids[sorted_agents[1]] * self.ctr_estimates[sorted_agents[1]]

            payment = (
                second_highest_bid / self.ctr_estimates[winner]
                if self.ctr_estimates[winner] > 0 else 0
            )

            self.remaining_budgets[winner] -= payment
            results.append((t, winner, payment))

            for k in range(self.num_agents):
                self.pacing_multipliers[k] = max(
                    0,
                    self.pacing_multipliers[k] - self.step_size * (
                        self.budgets[k] / self.time_horizon - payment
                    )
                )
                self.impressions[k] += 1
                if k == winner:
                    self.click_counts[k] += 1

        return results
    

    def calculate_liquid_welfare(self, results):
        agent_impressions = np.zeros(self.num_agents)

        for _, winner, _ in results:
            agent_impressions[winner] += 1

        liquid_welfare = 0
        for agent in range(self.num_agents):
            total_value = agent_impressions[agent] * self.valuations[agent]
            liquid_welfare += min(self.budgets[agent], total_value)

        return liquid_welfare

def plot_results(results):
    times, _, payments = zip(*results)

    plt.figure(figsize=(8, 6))
    plt.plot(times, payments, label="Payments", marker='o')
    plt.xlabel("Time")
    plt.ylabel("Payments")
    plt.title("Auction Payments Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("results.png", format="png", dpi=300)
    plt.show()


n_agents = 3
budgets = [100, 150, 200]
valuations = [0.8, 0.6, 0.9]  
step_size = 0.5
time_horizon = 100
upper_bound = 1.0

budget_pacing = BudgetPacingUCBCTR(n_agents, budgets, valuations, step_size, time_horizon, upper_bound)
results = budget_pacing.run()
plot_results(results)
liquid_welfare = budget_pacing.calculate_liquid_welfare(results)
print("Liquid Welfare:", liquid_welfare)