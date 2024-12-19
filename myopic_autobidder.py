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

        self.liquid_welfare = []
        self.allocations = np.zeros((self.num_agents, self.time_horizon)) 
        self.payments = np.zeros((self.num_agents, self.time_horizon)) 

    def run(self):
        bids = np.zeros(self.num_agents)

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
            self.allocations[winner, t] = 1  
            self.payments[winner, t] = payment  

            self.liquid_welfare.append(self.calculate_objective(self.allocations)[0])

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

        return self.allocations

    def calculate_objective(self, allocations, λ=0):
        total_utility = 0
        utilities = np.zeros(self.num_agents)
        liquid_welfare = 0
        for agent in range(self.num_agents):
            total_value = sum(allocations[agent] * valuations[agent])
            liquid_welfare += min(budgets[agent], total_value)
            payments = sum(allocations[agent])  
            total_utility += total_value - payments
        return (1-λ) * liquid_welfare + (λ * total_utility), utilities


num_agents = 3
budgets = [100, 150, 200]
valuations = [0.8,0.7,0.6]
step_size = 0.025
time_horizon = 2500
upper_bound = 1.0

budget_pacing = BudgetPacingUCBCTR(num_agents, budgets, valuations, step_size, time_horizon, upper_bound)
results = budget_pacing.run()
objective, utilities = budget_pacing.calculate_objective(results, λ=1)

print("Final CTR Estimates:", budget_pacing.ctr_estimates)
print("Remaining Budgets:", budget_pacing.remaining_budgets)
print("Liquid Welfare:", budget_pacing.calculate_objective(results, λ=0))

# Liquid Welfare Plots 
liquid_welfare_values = budget_pacing.liquid_welfare
plt.figure(figsize=(10, 6))
plt.plot(range(len(liquid_welfare_values)), liquid_welfare_values, label="Liquid Welfare", color="blue")
plt.xlabel("Iterations (Time Steps)")
plt.ylabel("Liquid Welfare")
plt.title("Liquid Welfare Over Time")
plt.grid(True)
plt.legend()
plt.savefig(f"myopic_liquid_welfare_{time_horizon}_steps")

# Payment Plots 
cumulative_payments = np.cumsum(budget_pacing.payments, axis=1)  
plt.figure(figsize=(10, 6))
for agent in range(num_agents):
    plt.plot(cumulative_payments[agent], label=f"Agent {agent}")

plt.xlabel("Rounds")
plt.ylabel("Cumulative Payments")
plt.title("Cumulative Payments Over Time")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig("myopic_cumulative_payments_over_time")  

# CTR Estimate Plots 
labels = [f"Agent {i}" for i in range(num_agents)]
width = 0.35  

plt.bar(range(num_agents), valuations, width, label='Valuations', alpha=0.7)
plt.bar([i + width for i in range(num_agents)], budget_pacing.ctr_estimates, width, label='CTR Estimates', alpha=0.7)

plt.xticks([i + width / 2 for i in range(num_agents)], labels, rotation=45)
plt.ylabel('Values')
plt.title('Comparison of Valuations and CTR Estimates (Myopic Algorithm)')
plt.legend()
plt.tight_layout()
plt.savefig('valuation_ctr_myopic')
print(budget_pacing.ctr_estimates)
