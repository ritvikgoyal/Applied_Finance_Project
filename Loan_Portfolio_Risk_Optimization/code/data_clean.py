#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
from datetime import datetime
import pandas_datareader.data as web


start_date = '2010-01-01'
end_date = '2016-12-31'

loan_f = pd.read_csv("~/Downloads/lending-club/all_loans.csv")
loan_f = loan_f.drop(["desc"], axis = 1)

loan_f = loan_f[['id','issue_d','funded_amnt','term', 'int_rate', 'installment', 'emp_length', 'home_ownership', 'annual_inc','verification_status','purpose', 'zip_code', 'addr_state','dti','delinq_2yrs', 'earliest_cr_line','fico_range_low','fico_range_high','inq_last_6mths', 'mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','out_prncp','out_prncp_inv', 'total_pymnt', 'total_pymnt_inv','total_rec_prncp', 'total_rec_int', 'last_credit_pull_d','total_rec_late_fee',
'application_type', 'annual_inc_joint', 'dti_joint',
'verification_status_joint','last_fico_range_high', 'last_fico_range_low','mths_since_recent_bc', 'mths_since_recent_bc_dlq',
'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
'num_tl_op_past_12m','last_pymnt_d','last_pymnt_amnt', 'next_pymnt_d','loan_status',]]

loan_f = loan_f[pd.notnull(loan_f['id'])]
loan_f = loan_f[pd.notnull(loan_f['term'])]
loan_f = loan_f[loan_f.term != "nan"]

null_zero_cols = ['delinq_2yrs','mths_since_last_delinq','mths_since_last_record','mths_since_recent_bc', 'mths_since_recent_bc_dlq',
       'mths_since_recent_inq', 'mths_since_recent_revol_delinq','num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
       'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
       'num_tl_op_past_12m']

for col in null_zero_cols:
    loan_f[col][pd.isnull(loan_f[col])] = 0


loan_f.verification_status_joint[loan_f.application_type == "Individual"] = loan_f.verification_status
loan_f.verification_status_joint[loan_f.application_type == "Individual"] = loan_f.verification_status
loan_f.annual_inc_joint[loan_f.application_type == "Individual"] = loan_f.annual_inc*2
loan_f.dti_joint[loan_f.application_type == "Individual"] = loan_f.dti/2

loan_f.emp_length[pd.isnull(loan_f.emp_length)] = loan_f["emp_length"].mode()[0]

loan_f.next_pymnt_d[pd.isnull(loan_f.next_pymnt_d)] = "Apr-2000"


loan_f = loan_f.dropna()


loan_f["term"] = loan_f.term.apply(lambda x : int(str(x).replace(' months','')))

cat_vars = ["emp_length","home_ownership","verification_status","verification_status_joint","purpose","addr_state","application_type"]


def create_dummy_cols(df,cat_cols):
    for col in cat_cols:
        temp_dummy_df = pd.get_dummies(df[col],prefix=col)
        temp_cols = temp_dummy_df.columns
        df_cols = df.columns
        df = pd.DataFrame(np.c_[df.values,temp_dummy_df.values])
        df.columns = np.r_[df_cols,temp_cols]
        df = df.drop([col],axis = 1)
        print(col)
    return(df)

df = create_dummy_cols(loan_f,cat_vars)


loan_f = df


loan_f["loan_date"] = loan_f.issue_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
loan_f["next_pymnt_d"] = loan_f.next_pymnt_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
loan_f["last_pymnt_d"] = loan_f.last_pymnt_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
loan_f["last_credit_pull_d"] = loan_f.last_credit_pull_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
