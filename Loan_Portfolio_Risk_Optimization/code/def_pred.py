#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from datetime import datetime
import pandas_datareader.data as web
import warnings
warnings.filterwarnings('ignore')


# In[2]:


sp = pd.read_csv("snp500_sim.csv")


# In[28]:


lc = pd.read_csv("lc_stats.csv")


# In[31]:


import pickle


# In[ ]:





# In[233]:


with open("loan_file.pickle", "rb") as input_file2:
    loan_subset = pickle.load(input_file2)


# In[ ]:





# In[235]:


y = loan_subset[["id","installment_num"]]


# In[98]:


with open("snp500_simulation_2.p", "rb") as input_file2:
    sp500df = pickle.load(input_file2)


# In[141]:


sim_psr = pd.read_csv("sim_psr.csv")


# In[130]:


with open("future_spot_rates_2.p", "rb") as input_file2:
    future_spot = pickle.load(input_file2)


# In[131]:


sp500_df = sp500df
cgap = future_spot
num_sim = sp500df.shape[1]


# In[149]:


loan_subset_req = loan_subset[loan_subset.loan_date >= pd.to_datetime("2018-01-01")].sample(500)


# In[103]:


loan_subset_req = loan_subset_req


# In[104]:


sp500_df["date_sp"] = sp500_df.index
sp500_df = sp500_df.iloc[:,1:]


# In[142]:


sim_psr = sim_psr.iloc[:,1:]


# In[108]:


sp500_df.shape


# In[143]:


sim_psr.index = sp500_df.index


# In[144]:


sim_psr


# In[125]:


num_sim


# In[128]:


cgap.shape


# In[132]:


dict_mkt = {}

sp500_df["date_sp"] = sp500_df.index



#   ps_rate.index = ps_rate.iloc[:,0]
#   ps_rate["date_ps"] = ps_rate.iloc[:,0]
#   ps_rate = ps_rate.iloc[:,1:]
sim_psr["date_ps"] = sim_psr.index
cgap["date_cg"] = cgap.index


for num in range(num_sim):
    spi = pd.concat([sp500_df.iloc[:,num],sp500_df.date_sp],axis =1)
    spi.index.name = "date"
    spi.columns = ["sp500","date_sp"]
    spi.date_sp = spi.date_sp.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))

    psi = pd.concat([sim_psr.iloc[:,num],sim_psr.date_ps],axis =1)
    psi.index.name = "date"
    psi.columns = ["psr","date_ps"]
    psi.date_ps = psi.date_ps.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
   # psi = pd.concat([ps_rate.iloc[:,num_sim],ps_rate.date_ps],axis =1)
   # psi.index.name = "date"
   # psi.columns = ["PSAVERT","date_ps"]
   # psi.index.name = "date"
   # psi.date_ps = psi.date_ps.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))

    cgi = pd.concat([cgap.iloc[:,num],cgap.date_cg],axis =1)
    cgi.index.name = "date"
    cgi.columns = ["coupon_gap","date_cg"]
    cgi.date_cg = cgi.date_cg.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))

    df = pd.concat([spi,cgi],axis =1)
    df = pd.concat([df,psi],axis = 1)
    dict_mkt[num_sim] = df


# In[151]:


dict1[0].head()


# In[154]:


with open("pool_2018.pickle", "rb") as input_file2:
    pool_res = pickle.load(input_file2)


# In[156]:


df = pool_res[1]
for i in range(2,len(pool_res)):
    df = df.append(pool_res[i])


# In[157]:


df.shape


# In[159]:


df = df.drop(["sp500","PSAVERT","10 YR"],axis =1)


# In[160]:


d1 = df.merge(dict1[0],left_on=  "loan_date",right_on = "date_sp")


# In[234]:


loan_data.groupbypby(["grade"]).count()


# In[163]:


d1.shape


# In[245]:


x1 =df


# In[246]:


ldf = loan_data[["id","grade","sub_grade"]]


# In[247]:


x2 = x1.merge(ldf, on = "id")


# In[220]:


loan_data = pd.read_csv("~/Downloads/lending-club/all_loans.csv")


# In[166]:


dict_df = {}


# In[167]:


for i in range(1001):
    dict_df[i] = df.merge(dict1[i],left_on=  "loan_date",right_on = "date_sp")
    #d1 = d1.append(d_temp)
    print(i)


# In[169]:


with open("log_2018.pickle", "rb") as input_file2:
    logreg = pickle.load(input_file2)


# In[ ]:





# In[177]:


for i in range(1001):
    dfi = dict_df[i]
    dfi["10 YR"] = dfi["coupon_gap"]
    dfi["PSAVERT"] = dfi["psr"]
    dfi = dfi[['funded_amnt', 'term', 'int_rate', 'installment', 'annual_inc',
       'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
       'inq_last_6mths', 'mths_since_last_delinq',
       'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
       'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
       'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_int', 'total_rec_late_fee', 'annual_inc_joint',
       'dti_joint', 'last_fico_range_high', 'last_fico_range_low',
       'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
       'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
       'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
       'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
       'num_tl_op_past_12m', 'last_pymnt_amnt', 'emp_length_1 year',
       'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years',
       'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years',
       'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years',
       'emp_length_< 1 year', 'home_ownership_ANY',
       'home_ownership_MORTGAGE', 'home_ownership_NONE',
       'home_ownership_OTHER', 'home_ownership_OWN',
       'home_ownership_RENT', 'verification_status_Not Verified',
       'verification_status_Source Verified',
       'verification_status_Verified',
       'verification_status_joint_Not Verified',
       'verification_status_joint_Source Verified',
       'verification_status_joint_Verified', 'purpose_car',
       'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_educational', 'purpose_home_improvement', 'purpose_house',
       'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
       'purpose_other', 'purpose_renewable_energy',
       'purpose_small_business', 'purpose_vacation', 'purpose_wedding',
       'addr_state_AK', 'addr_state_AL', 'addr_state_AR', 'addr_state_AZ',
       'addr_state_CA', 'addr_state_CO', 'addr_state_CT', 'addr_state_DC',
       'addr_state_DE', 'addr_state_FL', 'addr_state_GA', 'addr_state_HI',
       'addr_state_IA', 'addr_state_ID', 'addr_state_IL', 'addr_state_IN',
       'addr_state_KS', 'addr_state_KY', 'addr_state_LA', 'addr_state_MA',
       'addr_state_MD', 'addr_state_ME', 'addr_state_MI', 'addr_state_MN',
       'addr_state_MO', 'addr_state_MS', 'addr_state_MT', 'addr_state_NC',
       'addr_state_ND', 'addr_state_NE', 'addr_state_NH', 'addr_state_NJ',
       'addr_state_NM', 'addr_state_NV', 'addr_state_NY', 'addr_state_OH',
       'addr_state_OK', 'addr_state_OR', 'addr_state_PA', 'addr_state_RI',
       'addr_state_SC', 'addr_state_SD', 'addr_state_TN', 'addr_state_TX',
       'addr_state_UT', 'addr_state_VA', 'addr_state_VT', 'addr_state_WA',
       'addr_state_WI', 'addr_state_WV', 'addr_state_WY',
       'application_type_Individual', 'application_type_Joint App',
       'installment_num', 'sp500', 'PSAVERT', '10 YR', 'loan_amt_left',
       'loan_amt_paid','default_flag']]
    y_pred = logreg.predict_proba(dfi.loc[:,dfi.columns != "default_flag"])
    y_pred = [x[0] for x in y_pred]
    dfi["preds"] = y_pred
    dict_df[i] = dfi
    print(i)


# In[180]:


pred_ar = np.zeros((dict_df[1].shape[0],1))


# In[181]:


pred_ar


# In[ ]:


loan


# In[221]:


pred_df = df[["id"]].values


# In[198]:


pred_df


# In[222]:


for i in range(1,1001):
    pred_vals = dict_df[i].preds
    pred_df = np.c_[pred_df,pred_vals]
    print(i)


# In[225]:


loan_data


# In[231]:


l1 = loan_data.merge(x1,on = "id")


# In[232]:


l1


# In[230]:


x1


# In[223]:


x1 = pd.DataFrame(pred_df)


# In[227]:


col = ["id","installment_num"]
for i in range(1000):
    str1 = "sim_"+str(i+1)
    col.append(str1)


# In[229]:


x1.columns = np.array(col[:1000])


# In[189]:


pd.DataFrame(pred_ar).to_csv("pred_vals.csv")


# In[248]:



with open('pred2.pickle', 'wb') as handle:
    pickle.dump(x2, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[244]:


x2.shape


# In[ ]:


y_pred = logreg.predict_proba(X_test)


# In[150]:


loan1 = loan_subset_req.merge()


# In[145]:


def prep_market_vars(sp500_df,cgap,sim_psr,num_sim):
    dict_mkt = {}

    sp500_df["date_sp"] = sp500_df.index



    #   ps_rate.index = ps_rate.iloc[:,0]
    #   ps_rate["date_ps"] = ps_rate.iloc[:,0]
    #   ps_rate = ps_rate.iloc[:,1:]
    sim_psr["date_ps"] = sim_psr.index
    cgap["date_cg"] = cgap.index


    for num in range(num_sim):
        spi = pd.concat([sp500_df.iloc[:,num],sp500_df.date_sp],axis =1)
        spi.index.name = "date"
        spi.columns = ["sp500","date_sp"]
        spi.date_sp = spi.date_sp.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))

        psi = pd.concat([sim_psr.iloc[:,num],sim_psr.date_ps],axis =1)
        psi.index.name = "date"
        psi.columns = ["psr","date_ps"]
        psi.date_ps = psi.date_ps.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
       # psi = pd.concat([ps_rate.iloc[:,num_sim],ps_rate.date_ps],axis =1)
       # psi.index.name = "date"
       # psi.columns = ["PSAVERT","date_ps"]
       # psi.index.name = "date"
       # psi.date_ps = psi.date_ps.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))

        cgi = pd.concat([cgap.iloc[:,num],cgap.date_cg],axis =1)
        cgi.index.name = "date"
        cgi.columns = ["coupon_gap","date_cg"]
        cgi.date_cg = cgi.date_cg.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))

        df = pd.concat([spi,cgi],axis =1)
        df = pd.concat([df,psi],axis = 1)
        dict_mkt[num] = df
    return(dict_mkt)


# In[146]:


dict1 = prep_market_vars(sp500_df,cgap,sim_psr,num_sim)


# In[ ]:


loan


# In[148]:


dict1[0]


# In[21]:


def pred_def(model,loan_data,dict_mkt):
    loan_data = loan_data.drop(["desc"], axis = 1)
    loan_f = loan1[['id','issue_d','funded_amnt','term', 'int_rate', 'installment','emp_title', 'emp_length', 'home_ownership', 'annual_inc','verification_status','purpose', 'title', 'zip_code', 'addr_state','dti','delinq_2yrs', 'earliest_cr_line','fico_range_low','fico_range_high','inq_last_6mths', 'mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','out_prncp','out_prncp_inv', 'total_pymnt', 'total_pymnt_inv','total_rec_prncp', 'total_rec_int', 'last_credit_pull_d','total_rec_late_fee',
'application_type', 'annual_inc_joint', 'dti_joint',
'verification_status_joint','last_fico_range_high', 'last_fico_range_low','mths_since_recent_bc', 'mths_since_recent_bc_dlq',
'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
'num_tl_op_past_12m','last_pymnt_d','last_pymnt_amnt', 'next_pymnt_d','loan_status',]]
    
    loan_f = loan_f.drop(["emp_title"],axis = 1)
    loan_f = loan_f.drop(["title"],axis = 1)
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

    loan_f = create_dummy_cols(loan_f,cat_vars)
    
    loan_f["loan_date"] = loan_f.issue_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
    loan_f["next_pymnt_d"] = loan_f.next_pymnt_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
    loan_f["last_pymnt_d"] = loan_f.last_pymnt_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
    loan_f["last_credit_pull_d"] = loan_f.last_credit_pull_d.apply(lambda x: datetime.strptime(x,"%b-%Y"))
    
    
    #sp500_close = pd.DataFrame(sp500_data.Close)
    #sp500_close.columns = ["sp500"]
    sp.index = sp.iloc[:,0]
    sp["date_sp"] = sp.iloc[:,0]
    sp = sp.iloc[:,1:]
    sp =pd.concat([sp.iloc[:,0],sp.date_sp],axis =1)
    sp.index.name = "one"
    sp.columns = ["sp500","date_sp"]
    sp500_close["date_sp"] = sp500_close.index
    sp500_close.date_sp = sp500_close.date_sp.apply(lambda x : datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
    
    #ps_rate.index = ps_rate.DATE
    #ps_rate = ps_rate.drop(["DATE"],axis = 1)
    ps_rate["date_ps"] = ps_rate.index
    ps_rate.date_ps = ps_rate.date_ps.apply(lambda x : datetime.strptime(x,"%Y-%m-%d"))
    
    ts_yield_data = ts_yield_data.sort_values(by = "Date")
    ts_yield_data["Date"] = ts_yield_data.Date.apply(lambda x: (datetime.strptime(x,"%Y-%m-%d")))
    ts_yield_data["date_req"] = pd.to_datetime(ts_yield_data['Date']).dt.to_period('M')
    #ts_yield_df = pd.DataFrame(ts_yield_data.groupby("date_req").first()["10 YR"])
    

    def single_loan_cash_split(df,id_num,dict_mkt,num_sim,cols):
        df_req = loan_fp.iloc[np.where(df.id == id_num)[0][0],:]
        #def_flag = 1 if df_req.total_rec_late_fee > 0 else 0
        loan_date_difference = (df_req.loan_date +relativedelta(months = df_req.term+1)) - df_req.last_pymnt_d
        tenure = df_req.term 
        df_new = pd.DataFrame([df_req]*tenure)
        df_new["installment_num"] = np.arange(1,df_new.shape[0]+1)
        #df_new.loan_date = df_new.loan_date.apply(lambda x: datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
        df_new.index = np.arange(0,df_new.shape[0])
        inst_ar = []
        date_inst_ar = df_new[["loan_date","installment_num"]].values
        for i in range(df_new.shape[0]):
            inst_ar = np.r_[inst_ar,date_inst_ar[i,0]+relativedelta(months = date_inst_ar[i,1])]
        df_new["inst_date"] = inst_ar
        
        temp_df = df_new
        for sim in range(num_sim):
            
            mkt_df = dict_mkt[sim]
            
            temp_df = pd.concat([temp_df,mkt_df],axis =1)
            temp_df = temp_df[[cols]]
         
            y_pred = model.predict_proba(temp_df)
            y_pred = [x[0] for x in y_pred]
            
            temp_df["sim_"+str(num_sim)] = y_pred

        df_new = temp_df
        df_new["loan_amt_left"] = df_new["funded_amnt"]-(df_new["installment"]*df_new["installment_num"])
        df_new["loan_amt_paid"] = df_new["installment"]*df_new["installment_num"]
        df_new["default_flag"] = 0
        df_new.default_flag[tenure-1] = def_flag
        return(df_new)
    
    #df_subset = df[df.loan_status == loan_status_req]
    ids = np.array(df_new.id)
    split_df = single_loan_cash_split(df_new,ids[0],dict_mkt,num_sim)
    for i in range(1,len(ids)):
        temp_df = single_loan_cash_split(df_subset,ids[i],dict_mkt,cols)

        split_df = split_df.append(temp_df)
        split_df.head()
        print(ids[i],loan_status_req)
    return(split_df)
        


# In[ ]:




