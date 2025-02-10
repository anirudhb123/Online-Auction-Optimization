# Online-Auction-Optimization (CIS 6200 Final Project)

## Introduction

In this project, we implement and simulate two budget-pacing algorithms for online auctions that are defined in our paper:

1. **Myopic Algorithm:** Uses a UCB-style CTR prediction approach to optimize bids.
2. **Non-Myopic Algorithm:** Balances exploration and exploitation in CTR prediction to optimize bids over a longer horizon. (*Note: This implementation is currently a work in progress.*)

## Usage

### Running Simulations

The myopic algorithm can be run using the command ```python myopic.py```, while the non-myopic algorithm can be run using ```python non_myopic.py```.

### Parameters

We've preset parameters for each algorithm (lines 80-85 in the myopic algorithm and 121-126 in the non-myopic algorithm) but these can be modified as necessary.

## Output

Simulation results will be output as console messages, detailing metrics such as:

- Total Liquid Welfare
- Remaining Budgets
- CTR Estimates

You can also visualize additional results using the provided plots.
