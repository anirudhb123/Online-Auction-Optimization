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
