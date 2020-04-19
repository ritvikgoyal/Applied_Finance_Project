#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# In[2]:



market_data = pd.DataFrame()

start_date = '2010-01-01'
end_date = '2016-12-31'


# In[39]:


## S&P500 data
sp500_data = pd.read_csv('~/Desktop/sp500_data.csv')


# In[78]:


## Personal savings data
ps_rate = pd.read_csv('~/Desktop/PSAVERT.csv')
ps_rate_data = ps_rate[ps_rate['DATE']>start_date]


# In[9]:


# Treasury yields
ts_yield = pd.read_csv('~/Desktop/yield_data.csv')
ts_yield_data = ts_yield[ts_yield['Date']>start_date]


# In[10]:


loan_data = pd.read_csv("~/Desktop/all_loans.csv")


# In[11]:


loan1 = loan_data.sample(1000)


# In[12]:


np.array(loan1.columns)


# In[13]:


loan1 = loan1.drop(["desc"], axis = 1)


# In[14]:


loan_f = loan1[['id','issue_d','funded_amnt','term', 'int_rate', 'installment','emp_title', 'emp_length', 'home_ownership', 'annual_inc','verification_status','purpose', 'title', 'zip_code', 'addr_state','dti','delinq_2yrs', 'earliest_cr_line','fico_range_low','fico_range_high','inq_last_6mths', 'mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','out_prncp','out_prncp_inv', 'total_pymnt', 'total_pymnt_inv','total_rec_prncp', 'total_rec_int', 'last_credit_pull_d','total_rec_late_fee',
'application_type', 'annual_inc_joint', 'dti_joint',
'verification_status_joint','last_fico_range_high', 'last_fico_range_low','mths_since_recent_bc', 'mths_since_recent_bc_dlq',
'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
'num_tl_op_past_12m','last_pymnt_d','last_pymnt_amnt', 'next_pymnt_d','loan_status',]]


# In[15]:


loan_f.head()


# In[16]:


loan_f = loan_f.drop(["emp_title"],axis = 1)


# In[17]:


loan_f = loan_f.drop(["title"],axis = 1)


# In[18]:


np.array(loan_f.columns)


# In[19]:


loan_f = loan_f[pd.notnull(loan_f['id'])]
loan_f = loan_f[pd.notnull(loan_f['term'])]
loan_f = loan_f[loan_f.term != "nan"]


# In[20]:


null_zero_cols = ['delinq_2yrs','mths_since_last_delinq','mths_since_last_record','mths_since_recent_bc', 'mths_since_recent_bc_dlq',
       'mths_since_recent_inq', 'mths_since_recent_revol_delinq','num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
       'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
       'num_tl_op_past_12m']


# In[21]:


for col in null_zero_cols:
    loan_f[col][pd.isnull(loan_f[col])] = 0


# In[22]:


loan_f.verification_status_joint[loan_f.application_type == "Individual"] = loan_f.verification_status
loan_f.verification_status_joint[loan_f.application_type == "Individual"] = loan_f.verification_status
loan_f.annual_inc_joint[loan_f.application_type == "Individual"] = loan_f.annual_inc*2
loan_f.dti_joint[loan_f.application_type == "Individual"] = loan_f.dti/2


# In[23]:


loan_f.emp_length[pd.isnull(loan_f.emp_length)] = loan_f["emp_length"].mode()[0]


# In[24]:


loan_f.next_pymnt_d[pd.isnull(loan_f.next_pymnt_d)] = "Apr-2000"


# In[25]:


loan_f = loan_f.dropna()


# In[26]:


loan_f["term"] = loan_f.term.apply(lambda x : int(str(x).replace(' months','')))


# In[27]:


cat_vars = ["emp_length","home_ownership","verification_status","verification_status_joint","purpose","addr_state","application_type"]


# In[28]:


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


# In[29]:


df = create_dummy_cols(loan_f,cat_vars)


# In[30]:


loan_f = df


# In[31]:


import pickle


# In[32]:


loan_f["loan_date"] = loan_f.issue_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
loan_f["next_pymnt_d"] = loan_f.next_pymnt_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
loan_f["last_pymnt_d"] = loan_f.last_pymnt_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
loan_f["last_credit_pull_d"] = loan_f.last_credit_pull_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))


# In[33]:


with open("loan_file.pickle","wb") as handle:
    pickle.dump(loan_f, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[34]:


loan_subset = loan_f[loan_f.loan_date >= pd.to_datetime("2015-01-01")]


# In[35]:


from dateutil.relativedelta import relativedelta


# In[76]:


sp500_close = pd.DataFrame(sp500_data.Close)
sp500_close.index = sp500_data.Date
sp500_close.columns = ["sp500"]
sp500_close["date_sp"] = sp500_close.index
sp500_close.date_sp = sp500_close.date_sp.apply(lambda x : datetime.strptime(x,"%Y-%m-%d"))


# In[79]:


ps_rate.index = ps_rate.DATE


# In[80]:


ps_rate = ps_rate.drop(["DATE"],axis = 1)
ps_rate["date_ps"] = ps_rate.index
ps_rate.date_ps = ps_rate.date_ps.apply(lambda x : datetime.strptime(x,"%Y-%m-%d"))


# In[81]:


ts_yield_data = ts_yield_data.sort_values(by = "Date")
ts_yield_data["Date"] = ts_yield_data.Date.apply(lambda x: (datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S")))
ts_yield_data["date_req"] = pd.to_datetime(ts_yield_data['Date']).dt.to_period('M')
ts_yield_df = pd.DataFrame(ts_yield_data.groupby("date_req").first()["10 YR"])


# In[82]:


len(loan_subset.id)


# In[96]:


def single_loan_cash_split(df,id_num,sp500_df,psrate_df,ts_yield_df):
    try:
        df_req = df.iloc[np.where(df.id == id_num)[0][0],:]
        def_flag = 1 if df_req.total_rec_late_fee > 0 else 0
        loan_date_difference = (df_req.loan_date +relativedelta(months = df_req.term+1)) - df_req.last_pymnt_d
        tenure = df_req.term if (loan_date_difference == 0) else ((df_req.last_pymnt_d.year-df_req.loan_date.year)*12 +df_req.last_pymnt_d.month -df_req.last_pymnt_d.month-1)
        df_new = pd.DataFrame([df_req]*tenure)
        df_new["installment_num"] = np.arange(1,df_new.shape[0]+1)
        #df_new.loan_date = df_new.loan_date.apply(lambda x: datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
        df_new.index = np.arange(0,df_new.shape[0])
        inst_ar = []
        date_inst_ar = df_new[["loan_date","installment_num"]].values
        for i in range(df_new.shape[0]):
            inst_ar = np.r_[inst_ar,date_inst_ar[i,0]+relativedelta(months = date_inst_ar[i,1])]
        df_new["inst_date"] = inst_ar

        df_new = df_new.merge(sp500_df,left_on = "inst_date",right_on = sp500_close.date_sp)

        df_new.inst_date = df_new.inst_date.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
        df_new = df_new.merge(ps_rate,left_on = "inst_date", right_on = ps_rate.date_ps)
        df_new["month_year"] = pd.to_datetime(df_new['inst_date']).dt.to_period('M')

        df_new = df_new.merge(ts_yield_df, left_on = "month_year", right_on = "date_req")
        df_new = df_new.drop(["month_year"],axis = 1)

        df_new["loan_amt_left"] = df_new["funded_amnt"]-(df_new["installment"]*df_new["installment_num"])
        df_new["loan_amt_paid"] = df_new["installment"]*df_new["installment_num"]
        df_new["default_flag"] = 0
        df_new.default_flag[tenure-1] = def_flag
        return(df_new)
    except:
        pass


# In[99]:


def create_split_scenarios_df(loan_status_req,df = loan_subset,sp500 = sp500_close,psrate = ps_rate,ts_yield_data = ts_yield_df):
        df_subset = df[df.loan_status == loan_status_req]
        ids = np.array(df_subset.id)
        #print(ids)
        split_df = single_loan_cash_split(df_subset,ids[0],sp500,psrate,ts_yield_data)
        for i in range(1,len(ids)):
            temp_df = single_loan_cash_split(df_subset,ids[i],sp500,psrate,ts_yield_data)
            print(ids[i])
            if(temp_df is not None):
                split_df = split_df.append(temp_df)
                split_df.head()
                print(ids[i],loan_status_req)
        return(split_df)
        


# In[ ]:



import multiprocessing as mp

num_cpus = mp.cpu_count()

pool = mp.Pool(processes=num_cpus-1)
pool_results = pool.map(create_split_scenarios_df, ['Fully Paid', 'Current', 'Charged Off', 'In Grace Period',
       'Late (31-120 days)', 'Late (16-30 days)', 'Default'])
pool.close


# In[ ]:



with open('pool_results2.pickle', 'wb') as handle:
    pickle.dump(pool_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




