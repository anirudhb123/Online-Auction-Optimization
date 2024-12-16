import numpy as np

class BudgetPacingCTRPredictor:
    def __init__(self, n_agents, budgets, valuations, step_size, time_horizon, upper_bound):
        self.n_agents = n_agents
        self.budgets = budgets
        self.valuations = valuations
        self.step_size = step_size
        self.time_horizon = time_horizon
        self.upper_bound = upper_bound

    def run(self):
        # Phase 1: Exploration
        ctr_estimates = np.zeros(self.n_agents)
        impressions = np.ones(self.n_agents) 
        clicks = np.zeros(self.n_agents)     
        t = 1

        while True:
            max_lower_bound = max(ctr_estimates - np.sqrt(3 * np.log(self.time_horizon) / (2 * impressions)))
            max_upper_bound = max(ctr_estimates + np.sqrt(3 * np.log(self.time_horizon) / (2 * impressions)))

            print(max_lower_bound, max_upper_bound)

            if max_lower_bound > max_upper_bound:
                break

            for agent in range(self.n_agents):
                clicked = np.random.rand() < self.valuations[agent]  
                clicks[agent] += clicked
                impressions[agent] += 1

            ctr_estimates = clicks / impressions
            t += 1

        # Phase 2: Exploitation
        pacing_multipliers = np.zeros(self.n_agents)
        remaining_budgets = self.budgets.copy()
        allocations = np.zeros((self.n_agents, self.time_horizon))  

        for round in range(self.time_horizon):
            bids = np.zeros(self.n_agents)
            for agent in range(self.n_agents):
                adjusted_valuation = self.valuations[agent] * ctr_estimates[agent]
                bids[agent] = min(adjusted_valuation / (1 + pacing_multipliers[agent]), remaining_budgets[agent])

            sorted_agents = np.argsort(bids)[::-1]
            winning_agent = sorted_agents[0]
            second_highest_bid = bids[sorted_agents[1]] if len(sorted_agents) > 1 else 0

            clicked = np.random.rand() < self.valuations[winning_agent]
            payment = second_highest_bid / ctr_estimates[winning_agent] if ctr_estimates[winning_agent] > 0 else 0

            if clicked:
                remaining_budgets[winning_agent] -= payment

            pacing_multipliers[winning_agent] = max(0, pacing_multipliers[winning_agent] - self.step_size * (self.budgets[winning_agent] / self.time_horizon - payment))

            allocations[winning_agent, round] = 1  

        liquid_welfare = self.calculate_liquid_welfare(self.valuations, allocations, self.budgets)

        return {
            "ctr_estimates": ctr_estimates,
            "pacing_multipliers": pacing_multipliers,
            "remaining_budgets": remaining_budgets,
            "allocations": allocations,
            "liquid_welfare": liquid_welfare
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

n_agents = 3
budgets = [100, 150, 200]
valuations = [0.8, 0.6, 0.9]  
step_size = 0.5
time_horizon = 100
upper_bound = 1.0

predictor = BudgetPacingCTRPredictor(n_agents, budgets, valuations, step_size, time_horizon, upper_bound)
results = predictor.run()

print("Final CTR Estimates:", results["ctr_estimates"])
print("Final Pacing Multipliers:", results["pacing_multipliers"])
print("Remaining Budgets:", results["remaining_budgets"])
print("Liquid Welfare:", results["liquid_welfare"])
