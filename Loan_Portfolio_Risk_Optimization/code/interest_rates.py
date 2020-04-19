import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta


def B(t, T, kappa):
	delta_time = (T - t) / 12
	return (1 - np.exp(-kappa * delta_time)) / kappa


def B_theta_Riemann_sum(t, T, theta, kappa):
	riemann_sum = 0
	ds = 1.0 / 12
	for s in range(t, T):
		riemann_sum += B(s, T, kappa) * theta * ds
	return riemann_sum

def A(t, T, theta, kappa, sigma):
	delta_time = (T - t) / 12
	a_1 = B_theta_Riemann_sum(t, int(T), theta, kappa)
	a_2 = sigma ** 2 / (2 * kappa ** 2) * (delta_time + (1 - np.exp(- 2 * kappa * delta_time)) / (2 * kappa) - 2 * B(t, T, kappa))
	return -a_1 + a_2


def calc_future_spot_rates(tau_years, short_rates, t, theta, kappa, sigma):
	"""Calculate the future spot rates. For example, in t months what is the tau_years spot rate."""
	tau_months = 12.0 * tau_years
	ten_year_rate = -A(t, t + tau_months, theta, kappa, sigma) / tau_years + (1 / kappa) * ((1 - np.exp(-kappa * tau_years)) / tau_years) * short_rates
	return ten_year_rate


def construct_future_spot_rate_matrix(tau_years, spot_rates, theta, kappa, sigma, num_periods):
	num_simulations = spot_rates.shape[0]
	tau_year_rates = np.zeros((num_simulations, num_periods))
	for time_idx in range(num_periods):
		tau_year_rates[:, time_idx] = calc_future_spot_rates(tau_years, spot_rates[:, time_idx], time_idx, theta, kappa, sigma)
	return tau_year_rates


def simulate_antithetic_short_rates(num_paths, num_months, theta, kappa, sigma, inital_r=0, seed=102):
	np.random.seed(seed)
	draws = np.random.normal(size=(num_paths, num_months))
	dt = 1/12
	dr_pos = np.zeros((num_paths, num_months+1))
	r_pos = np.zeros((num_paths, num_months+1))
	dr_neg = np.zeros((num_paths, num_months+1))
	r_neg = np.zeros((num_paths, num_months+1))
	r_pos[:, 0] = inital_r
	r_neg[:, 0] = inital_r

	for i in range(num_months):
		dr_pos[:,i] = (theta - kappa*r_pos[:,i])*dt + sigma*dt**0.5*draws[:,i]
		dr_neg[:,i] = (theta - kappa*r_neg[:,i])*dt - sigma*dt**0.5*draws[:,i]
		r_pos[:,i+1] = r_pos[:,i] + dr_pos[:,i]
		r_neg[:,i+1] = r_neg[:,i] + dr_neg[:,i]

	return  r_pos, r_neg


def fixed_rates_per_loan(fixed_rates, num_loans=30, months=360):
	fixed_final = []
	for i in list(fixed_rates):
		fixed = []
		for j in range(0, months):
			fixed.append(i)
		fixed_final.append(fixed)

	return np.array(fixed_final)


def coupon_gap_per_user(fixed_rates_per_loan, future_spot_rates, origination_dates):
	"""fixed loan rate - 10 yr rate"""

	dataframes = []

	for user in range(0, fixed_rates_per_loan.shape[0]):
		start_date = origination_dates[user]

		coupon_gap = []

		for int_path in range(0, future_spot_rates.shape[0]):
			coupon_gap.append(list(fixed_rates_per_loan[user] - future_spot_rates[int_path]))

		coupon_gap = np.array(coupon_gap).T
		num_periods = coupon_gap.shape[0]
		dates = [start_date + relativedelta(months=timestep + 1) for timestep in range(num_periods)]

		dataframes.append(pd.DataFrame(coupon_gap, index=dates))

	return dataframes

# def date_coupon_gaps(list_of_user_frames, start_date, num_months):
# 	dates = [start_date + relativedelta(months=timestep + 1) for timestep in range(num_months)]
#
# 	end_yr = str(int(start_date.split("-")[0]) + years)
# 	end_date = end_yr + "-01-01"
# 	dates = pd.date_range(start_date, end_date, freq='MS')
#
# 	for user in range(0, len(list_of_user_frames)):
# 		list_of_user_frames[user].index = dates[:-1]
#
# 	return list_of_user_frames


if __name__ == '__main__':

	# Simulation Parameters
	N = 10000  # num simulations
	T = 360  # months
	COUPON_GAP_YEARS = 10  # years

	# Hull-White Parameters
	short_rate_on_august_30_2004 = 1.45 / 100
	theta = 0.004624
	kappa = 0.114676
	sigma = 0.01456

	# calculate discount factors
	short_rates, short_rates_anti = simulate_antithetic_short_rates(N, T, theta, kappa, sigma,
																	short_rate_on_august_30_2004)
	discount_factors = np.cumprod(np.exp(-short_rates / 12), axis=1)
	discount_factors_anti = np.cumprod(np.exp(-short_rates_anti / 12), axis=1)

	# calculate 10 year rates
	future_spot_rates = construct_future_spot_rate_matrix(COUPON_GAP_YEARS, short_rates, theta, kappa, sigma, T)
	future_spot_rates_anti = construct_future_spot_rate_matrix(COUPON_GAP_YEARS, short_rates_anti, theta, kappa, sigma,
															   T)

	print("Short Rates:\n")
	print(short_rates)

	print("Discount Factors: \n")
	print(discount_factors)

	print("Future 10 year spot rates: \n")
	print(future_spot_rates)