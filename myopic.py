import numpy as np
from numpy import log, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tqdm import tqdm

np.random.seed(42)

class BudgetPacingUCBCTR:
    def __init__(self, num_agents, budgets, valuations, epsilon, T, mu_bar, true_ctrs):
        """
        Inputs:
          num_agents: number of agents n
          budgets: list or array of budgets [B₁, ..., Bₙ]
          valuations: list or array of valuations [v₁, ..., vₙ]
          epsilon: pacing step-size (scalar for all agents)
          T: time horizon (number of rounds)
          mu_bar: upper bound on pacing multipliers
          true_ctrs: list or array of true CTRs [ρ₁, ..., ρₙ]
        """
        self.n = num_agents
        self.budgets = np.array(budgets, dtype=float)
        self.valuations = np.array(valuations, dtype=float)
        self.T = T
        self.epsilon = epsilon
        self.mu_bar = mu_bar
        self.true_ctrs = np.array(true_ctrs, dtype=float)

        self.mu = np.zeros(self.n)
        self.remaining_budget = self.budgets.copy()

        self.N = np.ones(self.n)
        self.rho_hat = np.zeros(self.n)
        for k in range(self.n):
            click = np.random.rand() < self.true_ctrs[k]
            self.rho_hat[k] = 1.0 if click else 0.0

        self.allocations = np.zeros((self.n, T))
        self.payments = np.zeros((self.n, T))
        self.mu_history = np.zeros((self.n, T))
        self.liquid_welfare = []

        self.rho_hat_history = np.zeros((self.n, T))
        self.N_history = np.zeros((self.n, T))

    def run(self):
        for t in tqdm(range(self.T), desc="Time Horizon", ncols=100, unit="round", leave=True):
            tilde_rho = np.minimum(self.rho_hat + sqrt(3 * log(self.T) / (2 * self.N)), 1.0)

            self.rho_hat_history[:, t] = self.rho_hat
            self.N_history[:, t] = self.N

            bids = np.zeros(self.n)
            for k in range(self.n):
                v_prime = self.valuations[k] * tilde_rho[k]
                bids[k] = min(v_prime / (1 + self.mu[k]), self.remaining_budget[k])

            effective_bids = tilde_rho * bids
            sorted_idx = np.argsort(effective_bids)[::-1]
            winner = sorted_idx[0]
            if len(sorted_idx) > 1:
                Q2 = effective_bids[sorted_idx[1]]
            else:
                Q2 = effective_bids[winner]

            z = np.zeros(self.n)
            for k in range(self.n):
                if k == winner:
                    clicked = (np.random.rand() < self.true_ctrs[k])
                    z[k] = (Q2 / tilde_rho[k]) if clicked else 0.0
                    self.N[k] += 1
                    self.rho_hat[k] = (1 - 1/self.N[k]) * self.rho_hat[k] + ((1.0 if clicked else 0.0) / self.N[k])
                else:
                    z[k] = 0.0

            for k in range(self.n):
                update = self.mu[k] - self.epsilon * (self.budgets[k] / self.T - z[k])
                self.mu[k] = np.clip(update, 0, self.mu_bar)
                self.mu_history[k, t] = self.mu[k]
                self.remaining_budget[k] -= z[k]
                if self.remaining_budget[k] < 0:
                    self.remaining_budget[k] = 0.0
                if z[k] > 0:
                    self.allocations[k, t] = 1
                    self.payments[k, t] = z[k]

            lw = 0.0
            for k in range(self.n):
                impressions_allocated = np.sum(self.allocations[k, :t+1])
                value_k = impressions_allocated * (self.true_ctrs[k] * self.valuations[k])
                lw += min(self.budgets[k], value_k)
            self.liquid_welfare.append(lw)

        return self.allocations

    def compute_optimal_revenue(self):
        products = self.true_ctrs * self.valuations
        sorted_products = np.sort(products)[::-1]
        second_highest = sorted_products[1] if len(sorted_products) > 1 else sorted_products[0]
        return self.T * second_highest

    def compute_optimal_liquid_welfare(self):
        ctr = self.true_ctrs  
        valuations = self.valuations
        budgets = self.budgets
        T = self.T
        n = self.n

        value_per_impression = ctr * valuations
        c = -value_per_impression

        A_ub = []
        b_ub = []
        for i in range(n):
            constraint = np.zeros(n)
            constraint[i] = value_per_impression[i]
            A_ub.append(constraint)
            b_ub.append(budgets[i])
        A_ub.append(np.ones(n))
        b_ub.append(T)

        bounds = [(0, T) for _ in range(n)]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x_opt = result.x
            w_star = -result.fun
            return w_star, x_opt
        else:
            raise Exception("Offline LP did not converge.")

    def compute_theoretical_regret_bound(self):
        sorted_prod = np.sort(self.true_ctrs * self.valuations)
        smax = sorted_prod[-2] if self.n >= 2 else sorted_prod[-1]
        term1 = sum(np.sqrt((2 * 3 * self.T * np.log(2 * self.n * self.T)) / self.true_ctrs))
        term2 = self.n / self.epsilon
        theoretical_bound = smax * (term1 + self.mu_bar * term2 + 1 / self.T)
        return theoretical_bound

# Parameters
num_agents = 20
valuations = np.random.uniform(0.5, 1.0, num_agents)
epsilon = 0.025
T = 5000
budgets = np.random.uniform(100, 200, num_agents)
mu_bar = 1.0
true_ctrs = np.random.uniform(0.1, 0.9, num_agents)

# Run the algorithm
pacing_algo = BudgetPacingUCBCTR(num_agents, budgets, valuations, epsilon, T, mu_bar, true_ctrs)
allocations = pacing_algo.run()

# Output results
print("Final CTR Estimates:", pacing_algo.rho_hat)
w_star, x_opt = pacing_algo.compute_optimal_liquid_welfare()
print("Achieved Liquid Welfare:", pacing_algo.liquid_welfare[-1])
print("Offline Optimal Liquid Welfare:", w_star)
total_revenue = np.sum(pacing_algo.payments)
opt = pacing_algo.compute_optimal_revenue()
actual_regret = opt - total_revenue
print("Actual Revenue Regret:", actual_regret)
theoretical_bound = pacing_algo.compute_theoretical_regret_bound()
print("Theoretical Regret Bound:", theoretical_bound)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(len(pacing_algo.liquid_welfare)), pacing_algo.liquid_welfare, label="Achieved Liquid Welfare (Gross Value)", color="blue")
plt.axhline(y=w_star, color="red", linestyle="--", label="Offline Optimal Liquid Welfare (w*)")
plt.xlabel("Rounds")
plt.ylabel("Liquid Welfare (Gross Expected Value)")
plt.title("Liquid Welfare Over Time")
plt.legend()
plt.grid(True)
plt.savefig("liquid_welfare_over_time.png")
plt.show()

cumulative_payments = np.cumsum(pacing_algo.payments, axis=1)
plt.figure(figsize=(10, 6))
for agent in range(num_agents):
    plt.plot(cumulative_payments[agent], label=f"Agent {agent}")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Payments")
plt.title("Cumulative Payments Over Time")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("cumulative_payments_over_time.png")
plt.show()

plt.figure(figsize=(10, 6))
for k in range(num_agents):
    final_N = pacing_algo.N_history[k, -1]
    final_rho_hat = pacing_algo.rho_hat_history[k, -1]
    ci_width = sqrt(3 * log(T) / (2 * final_N))
    plt.errorbar(k, final_rho_hat, yerr=ci_width, fmt='o', color='blue', label="Estimated CTR" if k==0 else "")
    plt.plot(k, pacing_algo.true_ctrs[k], 'ro', label="True CTR" if k==0 else "")
plt.xlabel("Agent")
plt.ylabel("CTR")
plt.title("Final Estimated CTRs with Confidence Intervals vs. True CTRs")
plt.legend()
plt.grid(True)
plt.savefig("ctr_confidence_intervals.png")
plt.show()