import numpy as np


def simulate_stock_prices(num_paths, num_periods, S, dT, sigma, r, y):

	shocks = np.random.normal(size=(num_paths, num_periods))
	stock_prices = np.zeros((num_paths, num_periods))
	stock_prices[:, 0] = S

	for t_idx in range(1, num_periods):
		stock_prices[:, t_idx] = stock_prices[:, t_idx-1] * np.exp((r - y - (sigma ** 2) / 2) * dT + sigma * np.sqrt(dT) * shocks[:, t_idx])

	return stock_prices


if __name__ == '__main__':

	# Parameters
	S = 2940
	dT = 1/12
	sigma = 0.157
	r = 0.08
	y = 0.0198

	# Simulation
	stocks_processes = simulate_stock_prices(1000, 42, S, dT, sigma, r, y)

	print(stocks_processes.shape)
	print(stocks_processes[:5,:5])
