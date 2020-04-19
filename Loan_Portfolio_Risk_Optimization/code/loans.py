import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Loan:
	"""Emulates the behaviour of a fixed rate loan.

	Parameters
	----------
	balance : float
		The initial balance of the loan.
	origination_date : datetime
		The origination date of the loan. First payment is one month later.
	apr : float
		Annulized payment rate as percentage. An input of 5.21 is interpreted as 5.21% annulized.
	monthly_payment : float
		Monthly amortization payment made by borrower.
	"""

	def __init__(self, balance, origination_date, apr, monthly_payment):

		self.original_balance = balance
		self.origination_date = origination_date
		self.monthly_payment = monthly_payment
		self.apr = apr / 100.0
		self.monthly_apr = self.apr / 12

		loan_balance, principal_cashflows, interest_cashflows = self._break_down_cashflows()
		self.cashflow_dates = [origination_date + relativedelta(months=timestep + 1) for timestep in range(len(principal_cashflows))]
		self.final_payment_date = max(self.cashflow_dates)
		self.balance_ts = pd.Series(loan_balance, index=self.cashflow_dates)
		self.principal_ts = pd.Series(principal_cashflows, index=self.cashflow_dates)
		self.interest_ts = pd.Series(interest_cashflows, index=self.cashflow_dates)
		self.amortization_ts = self.principal_ts + self.interest_ts


	def _break_down_cashflows(self):

		current_balance = self.original_balance
		principal_cashflows, interest_cashflows, loan_balance = [], [], []
		while current_balance > 0:
			interest_cf = current_balance * self.monthly_apr
			residual = max(0, interest_cf - self.monthly_payment)
			principal_cf = min(current_balance, max(0, self.monthly_payment - interest_cf))
			next_balance = current_balance - principal_cf + residual

			if next_balance > current_balance:
				raise(Warning("User isn't paying loan off fast enough."))

			current_balance = next_balance
			loan_balance.append(current_balance)
			principal_cashflows.append(principal_cf)
			interest_cashflows.append(interest_cf)

		return loan_balance, principal_cashflows, interest_cashflows

	def calc_unrealized_value(self, short_rates, default_probas, survivors=None, cutoff_date=None, value_date=None):
		"""Calculate the distribution of present values for many monte carlo samples of the world.

		Parameters
		----------
		short_rates : DataFrame
			Data frame with datetime index corresponding to dates and their corresponding short rates.
			Each column corresponds to a simulation path.
		default_probas : DataFrame
			Data frame with datetime index corresponding to the date for which we have the default probability.
			Each column corresponds to a simulation path.
		survivors : array
			Array of length of number of simulations with True / False entries to indicate if this loan is still active.
		cutoff_date : datetime
			Will only include cashflows after this date.
		value_date : datetime
			Date for which the current value is calculated.

		Returns
		-------
		array : Array of present value per simulation path.
		"""

		num_simulations = short_rates.shape[1]

		if cutoff_date is None:
			cutoff_date = self.origination_date
		if value_date is None:
			value_date = self.origination_date
		if survivors is None:
			survivors = np.ones(num_simulations)

		interest_discount_factors = np.exp(-short_rates[(short_rates.index > cutoff_date) & (short_rates.index <= self.final_payment_date)] / 12).cumprod()
		cum_surviving_proba = (1 -  default_probas[(default_probas.index > cutoff_date) & (default_probas.index <= self.final_payment_date)]).cumprod()
		discount_factors = interest_discount_factors * cum_surviving_proba
		present_values = discount_factors.multiply(self.amortization_ts[self.amortization_ts.index > cutoff_date], axis=0)
		present_values = np.multiply(survivors, present_values.sum().values)

		return present_values

	def simulate_realized_cashflows(self, default_probas, cutoff_date):

		if cutoff_date is None:
			cutoff_date = self.origination_date

		realized_default_probas = default_probas[default_probas.index <= cutoff_date]
		num_periods, num_simulations = realized_default_probas.shape

		uniform_samples = pd.DataFrame(np.random.uniform(0, 1, (num_periods, num_simulations)), index=realized_default_probas.index)
		has_survived = (uniform_samples < realized_default_probas).cumsum() == 0

		candidate_payments = self.amortization_ts[self.amortization_ts.index <= cutoff_date]
		realized_cashflows = has_survived.mul(candidate_payments, axis=0)

		has_defaulted = ((has_survived == False).sum() > 0).values

		return realized_cashflows, has_defaulted

	def calc_value_distribution(self, value_date, short_rates, default_probas):

		if value_date is None:
			value_date = self.origination_date

		realized_cashflows, has_defaulted = self.simulate_realized_cashflows(default_probas, value_date)
		past_cashflows = realized_cashflows.sum().values
		has_survived = has_defaulted == 0
		future_discounted_cashflows = self.calc_unrealized_value(short_rates, default_probas, has_survived, cutoff_date=value_date)
		total_value = past_cashflows + future_discounted_cashflows

		return total_value


if __name__ == '__main__':

	import matplotlib.pyplot as plt

	def calc_psa_cpr(speed, period):
		return (speed / 100) * 0.06 * min(1, float(period) / 30)

	def calc_psa_smm(speed, period):
		cpr = calc_psa_cpr(speed, period)
		return 1 - (1 - cpr) ** (1.0 / 12)

	balance = 1000.0  # dollars
	start_date = datetime(2017, 6, 1)
	cutoff = datetime(2017, 11, 1)
	monthly_payment = 50.00  # dollars
	num_periods = 36
	apr = 5.31  # percent
	cashflow_dates = [start_date + relativedelta(months=timestep + 1) for timestep in range(num_periods)]
	rates = pd.DataFrame(np.random.uniform(0, 0.05, (num_periods, 1000)), index=cashflow_dates)
	mortalities = [calc_psa_smm(100, timestep + 1) for timestep in range(num_periods)]
	default_df = pd.DataFrame(np.repeat(mortalities, 1000).reshape(-1, 1000) * np.random.uniform(0.5, 1.5, (num_periods, 1000)), index=cashflow_dates)
	survivors = np.ones(1000)

	loan = Loan(balance, start_date, apr, monthly_payment)
	value_distribution = loan.calc_value_distribution(cutoff, rates, default_df)

	print(value_distribution)
	print(value_distribution.shape)

	plt.hist(value_distribution)
	plt.show()
