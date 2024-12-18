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
    

    def calculate_objective(self, results, λ=0):
        agent_impressions = np.zeros(self.num_agents)
        total_utility = 0
        for _, winner, payment in results:
            agent_impressions[winner] += 1
            total_utility += self.valuations[winner] - payment  

        liquid_welfare = 0
        for agent in range(self.num_agents):
            total_value = agent_impressions[agent] * self.valuations[agent]
            liquid_welfare += min(self.budgets[agent], total_value)

        return (1-λ) * liquid_welfare + (λ * total_utility)
    
    def calculate_regret(self, results): 
        # Calculate optimal fixed strategy value 
        optimal_value = self.calculate_optimal_fixed_strategy()
        print(f"Optimal value: {optimal_value}")  # Debug print
    
        # Calculate cumulative value achieved by algorithm
        cumulative_value = 0
        regret_over_time = []
        
        for t, winner, payment in results:
            # Calculate value achieved in this round
            round_value = self.valuations[winner] - payment
            cumulative_value += round_value
            
            # Calculate regret up to this point
            optimal_value_until_t = optimal_value * (t + 1)
            regret = max(optimal_value_until_t - cumulative_value, 1e-10)  # Ensure positive
            regret_over_time.append(regret)
            # Debug prints for first few rounds
            if t < 5:
                print(f"Round {t}: value={round_value}, cumul={cumulative_value}, regret={regret}")

        return regret_over_time

    def calculate_optimal_fixed_strategy(self):
        # Grid search over possible fixed pacing multipliers
        best_value = float('-inf')
        for multiplier in np.linspace(0, self.upper_bound, 100):
            value = self.simulate_fixed_strategy(multiplier)
            best_value = max(best_value, value)
        return best_value

    def simulate_fixed_strategy(self, fixed_multiplier):
        # Simulate auction outcomes with a fixed pacing multiplier
        total_value = 0
        remaining_budget = self.budgets.copy()
        
        for t in range(self.time_horizon):
            # Simulate one round with fixed multiplier
            round_value = self.simulate_round(fixed_multiplier, remaining_budget)
            total_value += round_value
        
        return total_value / self.time_horizon

    def simulate_round(self, fixed_multiplier, remaining_budget):
        bids = np.zeros(self.num_agents)
        # Use true CTRs for simulation (could be estimated from historical data)
        true_ctrs = self.click_counts / np.maximum(self.impressions, 1)
        
        for k in range(self.num_agents):
            adjusted_value = self.valuations[k] * true_ctrs[k]
            bids[k] = min(
                adjusted_value / (1 + fixed_multiplier),
                remaining_budget[k]
            )
        
        sorted_agents = np.argsort(bids * true_ctrs)[::-1]
        winner = sorted_agents[0]
        second_price = bids[sorted_agents[1]] * true_ctrs[sorted_agents[1]]
        
        payment = second_price / true_ctrs[winner] if true_ctrs[winner] > 0 else 0
        remaining_budget[winner] -= payment
        
        return self.valuations[winner] - payment  # Return utility (value - payment)

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


def plot_regret(regret_over_time):
    plt.figure(figsize=(10, 6))
    
    # Plot in log-log scale
    plt.loglog(range(1, len(regret_over_time) + 1), regret_over_time, 'b-', label='Empirical Regret')
    
    # Add theoretical bound line (T^(3/4))
    T = len(regret_over_time)
    theoretical = [t**0.75 for t in range(1, T+1)]
    plt.loglog(range(1, T+1), theoretical, 'r--', label='T^(3/4) Bound')
    
    plt.xlabel('Number of Auctions (log scale)')
    plt.ylabel('Regret (log scale)')
    plt.title('Budget Pacing with UCB-CTR Regret Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('regret_analysis.png', dpi=300)
    plt.show()


def run_multiple_simulations(n_simulations=100):
    all_regrets = []
    
    for sim in range(n_simulations):
        # Randomize parameters slightly for each simulation
        budgets = np.random.uniform(100, 1000, n_agents)  
        valuations = np.random.uniform(0.1, 10, n_agents)   

        # Debug print to see the ranges
        print(f"Simulation {sim}:")
        print(f"Budget range: [{min(budgets):.1f}, {max(budgets):.1f}]")
        print(f"Value range: [{min(valuations):.1f}, {max(valuations):.1f}]")
        
        budget_pacing = BudgetPacingUCBCTR(n_agents, budgets, valuations, step_size, time_horizon, upper_bound)
        results = budget_pacing.run()
        regret = budget_pacing.calculate_regret(results)
        
        # Filter out any negative regrets as done in paper
        regret = np.maximum(regret, 1e-10)  # Replace negatives with small positive
        all_regrets.append(regret)
    
    return all_regrets

def plot_regret_analysis(all_regrets):
    plt.figure(figsize=(10, 6))
    
    # Plot each simulation's regret curve
    x = np.arange(1, time_horizon + 1)
    for regret in all_regrets:
        plt.loglog(x, regret, alpha=0.3, linewidth=0.5)
    
    # Calculate and print average slope (α)
    log_x = np.log(x)
    slopes = []
    for regret in all_regrets:
        log_y = np.log(regret)
        slope, _ = np.polyfit(log_x, log_y, 1)
        slopes.append(slope)
    
    avg_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    print(f"Estimated α = {avg_slope:.3f} (±{std_slope:.3f})")
    
    # Set x-axis limits to start from 1000
    plt.xlim(1000, time_horizon)
    
    plt.xlabel('Number of Auctions (Log)')
    plt.ylabel('Total Regret (Log)')
    plt.title('Budget Pacing with UCB-CTR Regret Analysis')
    plt.grid(True)
    plt.savefig('regret_analysis.png', dpi=300)
    plt.show()

# Parameters
n_agents = 3
step_size = 0.1
time_horizon = 10000
upper_bound = 5.0

# Run simulations and plot
all_regrets = run_multiple_simulations(n_simulations=100)
plot_regret_analysis(all_regrets)
