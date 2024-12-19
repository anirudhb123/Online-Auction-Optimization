import numpy as np
import matplotlib.pyplot as plt

class BudgetPacingCTRPredictor:
    def __init__(self, num_agents, budgets, valuations, step_size, time_horizon, upper_bound):
        self.num_agents = num_agents
        self.budgets = budgets
        self.valuations = valuations
        self.step_size = step_size
        self.time_horizon = time_horizon
        self.upper_bound = upper_bound
        self.liquid_welfares = []

        self.exploration_clicks = []
        self.exploitation_clicks = []
        self.agent_clicks = {i: {"exploration": 0, "exploitation": 0} for i in range(num_agents)}

    def run(self):
        # Phase 1: Exploration
        ctr_estimates = np.zeros(self.num_agents)
        impressions = np.ones(self.num_agents) 
        clicks = np.zeros(self.num_agents)     
        t = 1

        bound = 0.01
        while True:
            lower_bounds = ctr_estimates - np.sqrt(3 * np.log(self.time_horizon) / (2 * impressions))
            upper_bounds = ctr_estimates + np.sqrt(3 * np.log(self.time_horizon) / (2 * impressions))

            best_agent = np.argmax(lower_bounds)
            max_lower_bound = lower_bounds[best_agent]

            max_upper_bound = max(upper_bounds[j] for j in range(self.num_agents) if j != best_agent)

            if abs(max_lower_bound - max_upper_bound) < bound:
                break

            for agent in range(self.num_agents):
                clicked = np.random.rand() < self.valuations[agent]  
                clicks[agent] += clicked
                impressions[agent] += 1
                self.exploration_clicks.append(clicked)  
                self.agent_clicks[agent]["exploration"] += clicked

            ctr_estimates = clicks / impressions
            t += 1

        # Phase 2: Exploitation
        pacing_multipliers = np.zeros(self.num_agents)  
        remaining_budgets = self.budgets.copy()
        allocations = np.zeros((self.num_agents, self.time_horizon))

        payments = np.zeros((self.num_agents, self.time_horizon))  

        ctr_lower_bounds = ctr_estimates - np.sqrt(3 * np.log(self.time_horizon) / (2 * impressions))

        for round in range(self.time_horizon):
            bids = np.zeros(self.num_agents)

            for agent in range(self.num_agents):
                adjusted_valuation = self.valuations[agent] * ctr_lower_bounds[agent]
                bids[agent] = min(adjusted_valuation / (1 + pacing_multipliers[agent]), remaining_budgets[agent])

            weighted_bids = bids * ctr_lower_bounds
            
            sorted_agents = np.argsort(weighted_bids)[::-1]
            winning_agent = None
            payment = 0

            for candidate in sorted_agents:
                Q2 = max(weighted_bids[j] for j in range(self.num_agents) if j != candidate)
                payment_candidate = Q2 / ctr_lower_bounds[candidate] if ctr_lower_bounds[candidate] > 0 else 0

                if remaining_budgets[candidate] >= payment_candidate:
                    winning_agent = candidate
                    payment = payment_candidate
                    break

            if winning_agent is None:
                continue

            clicked = np.random.rand() < self.valuations[winning_agent]
            if clicked:
                remaining_budgets[winning_agent] -= payment
                self.exploitation_clicks.append(1)
                self.agent_clicks[winning_agent]["exploitation"] += 1
            else:
                self.exploitation_clicks.append(0)

            pacing_multipliers[winning_agent] = np.clip(
                pacing_multipliers[winning_agent] - self.step_size * (self.budgets[winning_agent] / self.time_horizon - payment),
                0, self.upper_bound
            )

            allocations[winning_agent, round] = 1
            payments[winning_agent, round] = payment

            self.liquid_welfares.append(self.calculate_objective(self.valuations, allocations, self.budgets))

        return {
            "ctr_estimates": ctr_estimates,
            "pacing_multipliers": pacing_multipliers,
            "remaining_budgets": remaining_budgets,
            "allocations": allocations,
            "liquid_welfare": self.liquid_welfares[-1],
            "payments": payments,
            'exploration_clicks': self.exploration_clicks,
            'exploitation_clicks': self.exploitation_clicks
        }
    
    def calculate_objective(self, valuations, allocations, budgets, λ=0):
        liquid_welfare = 0
        total_utility = 0
        for agent in range(self.num_agents):
            total_value = sum(allocations[agent] * valuations[agent])
            liquid_welfare += min(budgets[agent], total_value)
            payments = sum(allocations[agent])  
            total_utility += total_value - payments
        return (1-λ) * liquid_welfare + (λ * total_utility)

num_agents = 3
budgets = [100, 150, 200]
valuations = [0.8,0.7,0.6]
step_size = 0.025
time_horizon = 2500
upper_bound = 1.0

predictor = BudgetPacingCTRPredictor(num_agents, budgets, valuations, step_size, time_horizon, upper_bound)
results = predictor.run()

print("Final CTR Estimates:", results["ctr_estimates"])
print("Final Pacing Multipliers:", results["pacing_multipliers"])
print("Remaining Budgets:", results["remaining_budgets"])
print("Liquid Welfare:", results["liquid_welfare"])


# Payment Plots 
payments = results["payments"]
cumulative_payments = np.cumsum(payments, axis=1)

plt.figure(figsize=(10, 6))
for agent in range(num_agents):
    plt.plot(cumulative_payments[agent], label=f"Agent {agent}")

plt.xlabel("Rounds")
plt.ylabel("Cumulative Payments")
plt.title("Cumulative Payments Over Time")
plt.legend()
plt.grid(True)
plt.savefig("cumulative_payments_over_time_non_myopic.png")  

# Liquid Welfare Plots
liquid_welfare_values = predictor.liquid_welfares

plt.figure(figsize=(10, 6))
plt.plot(range(len(liquid_welfare_values)), liquid_welfare_values, label="Liquid Welfare", color="blue")
plt.xlabel("Iterations (Time Steps)")
plt.ylabel("Liquid Welfare")
plt.title("Liquid Welfare Over Time")
plt.grid(True)
plt.legend()
plt.savefig(f"non_myopic_liquid_welfare_{time_horizon}_steps")

# Exploration vs. Exploitation Phase Plots
agents = np.arange(num_agents)
exploration = [predictor.agent_clicks[i]["exploration"] for i in range(num_agents)]
exploitation = [predictor.agent_clicks[i]["exploitation"] for i in range(num_agents)]

plt.figure(figsize=(10, 6))
bar_width = 0.35

bars1 = plt.bar(agents, exploration, width=bar_width, label='Exploration')
bars2 = plt.bar(agents + bar_width, exploitation, width=bar_width, label='Exploitation')

for bar in bars1:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{int(bar.get_height())}',
             ha='center', va='bottom', fontsize=9)

for bar in bars2:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{int(bar.get_height())}',
             ha='center', va='bottom', fontsize=9)

plt.xlabel('Agent')
plt.ylabel('Total Clicks')
plt.title(f"Total Clicks Per Agent: Exploration vs Exploitation - {time_horizon} Steps")
plt.xticks(agents + bar_width / 2, [f'Agent {i}' for i in agents])
plt.yscale('log')  
plt.legend()
plt.savefig(f"clicks_per_agent_{time_horizon}_steps")

exploration_clicks = np.cumsum(predictor.exploration_clicks)
exploitation_clicks = np.cumsum(predictor.exploitation_clicks)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(exploration_clicks) + 1), exploration_clicks, label='Exploration Phase')
plt.plot(range(len(exploration_clicks) + 1, len(exploration_clicks) + len(exploitation_clicks) + 1),
         exploration_clicks[-1] + exploitation_clicks, label='Exploitation Phase')
plt.axvline(x=len(exploration_clicks), color='r', linestyle='--', label='Phase Transition')
plt.text(len(exploration_clicks) + 50, plt.ylim()[1] * 0.5, f"{len(exploration_clicks)}", 
         ha='left', va='center', fontsize=10, color='black')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Clicks')
plt.title(f"Cumulative Clicks Over Time: Exploration vs Exploitation - Step Size: {step_size}")
plt.legend()
print(f"exploration_vs_exploitation_{time_horizon}_steps_{step_size * 1000}_size")
plt.savefig(f"exploration_vs_exploitation_{time_horizon}_steps_{int(step_size * 1000)}_size")


# CTR Estimate Plots 
labels = [f"Agent {i}" for i in range(num_agents)]
width = 0.35  
plt.figure(figsize=(10, 6))
plt.bar(range(num_agents), valuations, width, label='Valuations', alpha=0.7)
plt.bar([i + width for i in range(num_agents)], results['ctr_estimates'], width, label='CTR Estimates', alpha=0.7)

plt.xticks([i + width / 2 for i in range(num_agents)], labels, rotation=45)
plt.ylabel('Values')
plt.title('Comparison of Valuations and CTR Estimates (Non-Myopic Algorithm)')
plt.legend()
plt.tight_layout()
plt.savefig('valuation_ctr_non_myopic')
print(results['ctr_estimates'])
