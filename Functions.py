######################################################### packages ############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

######################################################### Bin function ############################################################
#takes a df and bins it by code health (1 to 1.25 is 1, 1.25 to 1.75 is 1.5, 1.75 to 2.25 is 2 etc)
def bin_ch(df):
    df.rename(columns={"code_health_now": "code_health", "mean_time_for_issue": "lead_time_minutes"},inplace=True)
    df2=df.copy()
    
    for i in range(df2.shape[0]):
        h=df2['code_health'].iloc[i]
        if 1<=h<1.25:
            df2['code_health'].iloc[i]=1
        elif 1.25<=h<1.75:
            df2['code_health'].iloc[i]=1.5
        elif 1.75<=h<2.25:
            df2['code_health'].iloc[i]=2
        elif 2.25<=h<2.75:
            df2['code_health'].iloc[i]=2.5
        elif 2.75<=h<3.25:
            df2['code_health'].iloc[i]=3
        elif 3.25<=h<3.75:
            df2['code_health'].iloc[i]=3.5
        elif 3.75<=h<4.25:
            df2['code_health'].iloc[i]=4
        elif 4.25<=h<4.75:
            df2['code_health'].iloc[i]=4.5
        elif 4.75<=h<5.25:
            df2['code_health'].iloc[i]=5
        elif 5.25<=h<5.75:
            df2['code_health'].iloc[i]=5.5
        elif 5.75<=h<6.25:
            df2['code_health'].iloc[i]=6
        elif 6.25<=h<6.75:
            df2['code_health'].iloc[i]=6.5
        elif 6.75<=h<7.25:
            df2['code_health'].iloc[i]=7
        elif 7.25<=h<7.75:
            df2['code_health'].iloc[i]=7.5
        elif 7.75<=h<8.25:
            df2['code_health'].iloc[i]=8
        elif 8.25<=h<8.75:
            df2['code_health'].iloc[i]=8.5
        elif 8.75<=h<9.25:
            df2['code_health'].iloc[i]=9
        elif 9.25<=h<9.75:
            df2['code_health'].iloc[i]=9.5
        elif 9.75<=h<10:
            df2['code_health'].iloc[i]=10
    return df2

##################################################### Linear Model functions ######################################################
# takes the df, makes linear regression for bugs in fct of CH and returns the regresssion coefficients
def bug_Lin_Regression(df):
   # Definitions ############################################
    x_bugs=df['code_health'].to_numpy()
    y_bugs=df['total_defects'].to_numpy()

    # Model ###############################
    p=np.polyfit(x_bugs,y_bugs,1)
    LinReg_bugs_a,LinReg_bugs_b=p
    LinReg_bugs_a,LinReg_bugs_b=np.round(LinReg_bugs_a,3),np.round(LinReg_bugs_b,3)
    
    return [LinReg_bugs_a,LinReg_bugs_b]

# takes the df, makes linear regression for time-in-dev in fct of CH and returns the regresssion coefficients
def time_Lin_Regression(df):
    # Definitions ############################################
    x_time=df[df['lead_time_minutes'].notna()]['code_health'].to_numpy()
    y_time=df[df['lead_time_minutes'].notna()]['lead_time_minutes'].to_numpy()
    
    # Model ###############################
    p=np.polyfit(x_time,y_time,1)
    LinReg_time_a,LinReg_time_b=p
    LinReg_time_a,LinReg_time_b=np.round(LinReg_time_a,3),np.round(LinReg_time_b,3)
    
    return [LinReg_time_a,LinReg_time_b]


# calculates the added value in fucntion of the parameters with a linear model
def lin_Added_Value(value_creation_old,CH_old,CH_new,unplanned_work_old,bug_coeff_list,time_coeff_list):
    LinReg_bugs_a,LinReg_bugs_b=bug_coeff_list
    LinReg_time_a,LinReg_time_b=time_coeff_list
    
    # New capacity ################################
    capacity_old=value_creation_old/(1-unplanned_work_old)
    
    time_old=LinReg_time_a*CH_old+LinReg_time_b
    time_new=LinReg_time_a*CH_new+LinReg_time_b
    
    time_improvement=(time_old-time_new)/time_old
    
    capacity_new=capacity_old*(1+time_improvement)
    
    # Bug Reduction ################################
    bugs_old=LinReg_bugs_a*CH_old+LinReg_bugs_b
    bugs_new=LinReg_bugs_a*CH_new+LinReg_bugs_b
    
    bug_reduction=(bugs_old-bugs_new)/bugs_old
    
    # New Unplanned Work ################################
    unplanned_work_new=unplanned_work_old*(1-bug_reduction)
    
    # New Value Creation ################################
    value_creation_new=capacity_new*(1-unplanned_work_new)
    
    # Added Value ################################
    added_value=value_creation_new-value_creation_old
    
    return added_value

# plots the gain/loss in added value in fct of starting CH and parameters for linear model
def plot_Lin_Added_Value(value_creation_old,CH_old,unplanned_work_old,bug_coeff_list,time_coeff_list):
    X=np.arange(1,10.5,0.5)
    Y=np.zeros_like(X)
    for i in range(X.shape[0]):
        Y[i]= lin_Added_Value(value_creation_old,CH_old,X[i],unplanned_work_old,bug_coeff_list,time_coeff_list)
    
    plt.plot(X,Y)
    plt.plot(CH_old,0,color='red', marker='o', label='situation now')
    plt.legend()
    plt.xlabel('Code Health')
    plt.ylabel('added value in M')
    plt.title("Starting code health: "+str(CH_old))
    plt.show()

def Uncertainty_plot_Lin_Added_Value(value_creation_old,CH_old,unplanned_work_old,bug_coeff_list,time_coeff_list, type):
    #type 0= ligtblue 
    #type 1= main
    X=np.arange(1,10.5,0.5)
    Y=np.zeros_like(X)
    for i in range(X.shape[0]):
        Y[i]= lin_Added_Value(value_creation_old,CH_old,X[i],unplanned_work_old,bug_coeff_list,time_coeff_list)
    
    if type==0:
        plt.plot(X,Y,color='#3C6A88', alpha=0.3)
    elif type==1:
        plt.plot(X,Y,color='mediumspringgreen')
        plt.plot(CH_old,0,color='magenta', marker='o', label='situation now')
        plt.legend()
        plt.xlabel('Code Health')
        plt.ylabel('added value in M')

# big fuction that returns everything for the linear model
def linear_Model(df,value_creation_old,CH_old,unplanned_work_old):
    bugs=bug_Lin_Regression(df)
    time= time_Lin_Regression(df)
    plot_Bug_Regression(df,bugs)
    plot_Time_Regression(df,time)
    #plot_Lin_Added_Value(value_creation_old,CH_old,unplanned_work_old,bugs,time)

################################################## Polynomial Model functions #######################################################
# takes the df, makes 3rd degree polynomial regression for bugs in fct of CH and returns the regresssion coefficients
def bug_Poly_Regression(df):
    # Definitions ############################################
    x_bugs=df['code_health'].to_numpy()
    y_bugs=df['total_defects'].to_numpy()

    # Model ###############################
    p=np.polyfit(x_bugs,y_bugs,3)
    PolReg_bugs_a,PolReg_bugs_b,PolReg_bugs_c,PolReg_bugs_d=p
    PolReg_bugs_a,PolReg_bugs_b,PolReg_bugs_c,PolReg_bugs_d=np.round(PolReg_bugs_a,3),np.round(PolReg_bugs_b,3),np.round(PolReg_bugs_c,3),np.round(PolReg_bugs_d,3)

    return  [PolReg_bugs_a,PolReg_bugs_b,PolReg_bugs_c,PolReg_bugs_d]

# takes the df, makes 3rd degree polynomial regression for time-in-dev in fct of CH and returns the regresssion coefficients
def time_Poly_Regression(df):
    # Definitions ############################################
    x_time=df[df['lead_time_minutes'].notna()]['code_health'].to_numpy()
    y_time=df[df['lead_time_minutes'].notna()]['lead_time_minutes'].to_numpy()
    
    # Model ###############################
    p=np.polyfit(x_time,y_time,3)
    PolReg_time_a,PolReg_time_b,PolReg_time_c,PolReg_time_d=p
    PolReg_time_a,PolReg_time_b,PolReg_time_c,PolReg_time_d=np.round(PolReg_time_a,3),np.round(PolReg_time_b,3),np.round(PolReg_time_c,3),np.round(PolReg_time_d,3)
    
    return [PolReg_time_a,PolReg_time_b,PolReg_time_c,PolReg_time_d]


# calculates the added value in fucntion of the parameters with a 3rd degree polynomial model
def poly_Added_Value(value_creation_old,CH_old,CH_new,unplanned_work_old,bug_coeff_list,time_coeff_list):
    PolReg_bugs_a,PolReg_bugs_b,PolReg_bugs_c,PolReg_bugs_d=bug_coeff_list
    PolReg_time_a,PolReg_time_b,PolReg_time_c,PolReg_time_d=time_coeff_list
    
    # New capacity ################################
    capacity_old=value_creation_old/(1-unplanned_work_old)
    
    time_old=PolReg_time_a*(CH_old**3)+PolReg_time_b*(CH_old**2)+PolReg_time_c*CH_old+PolReg_time_d
    time_new=PolReg_time_a*(CH_new**3)+PolReg_time_b*(CH_new**2)+PolReg_time_c*CH_new+PolReg_time_d
    
    time_improvement=(time_old-time_new)/time_old
    
    capacity_new=capacity_old*(1+time_improvement)
    
    # Bug Reduction ################################
    bugs_old=PolReg_bugs_a*(CH_old**3)+PolReg_bugs_b*(CH_old**2)+PolReg_bugs_c*CH_old+PolReg_bugs_d
    bugs_new=PolReg_bugs_a*(CH_new**3)+PolReg_bugs_b*(CH_new**2)+PolReg_bugs_c*CH_new+PolReg_bugs_d
    
    bug_reduction=(bugs_old-bugs_new)/bugs_old
    
    # New Unplanned Work ################################
    unplanned_work_new=unplanned_work_old*(1-bug_reduction)
    
    # New Value Creation ################################
    value_creation_new=capacity_new*(1-unplanned_work_new)
    
    # Added Value ################################
    added_value=value_creation_new-value_creation_old
    
    return added_value

# plots the gain/loss in added value in fct of starting CH and parameters for 3rd degree polynomial model
def plot_Poly_Added_Value(value_creation_old,CH_old,unplanned_work_old,bug_coeff_list,time_coeff_list):
    X=np.arange(1,10.5,0.5)
    Y=np.zeros_like(X)
    for i in range(X.shape[0]):
        Y[i]= poly_Added_Value(value_creation_old,CH_old,X[i],unplanned_work_old, bug_coeff_list,time_coeff_list)

    plt.plot(X,Y)
    plt.plot(CH_old,0,color='red', marker='o', label='situation now')
    plt.legend()
    plt.xlabel('Code Health')
    plt.ylabel('added value in M')
    plt.title("Starting code health: "+str(CH_old))
    plt.show() 

def Uncertainty_plot_Poly_Added_Value(value_creation_old,CH_old,unplanned_work_old,bug_coeff_list,time_coeff_list,type):
    #type 0= ligtblue 
    #type 1= main
    X=np.arange(1,10.5,0.5)
    Y=np.zeros_like(X)
    for i in range(X.shape[0]):
        Y[i]= poly_Added_Value(value_creation_old,CH_old,X[i],unplanned_work_old, bug_coeff_list,time_coeff_list)

    if type==0:
        plt.plot(X,Y,color='#3C6A88', alpha=0.3)
    elif type==1:
        plt.plot(X,Y,color='mediumspringgreen')
        plt.plot(CH_old,0,color='magenta', marker='o', label='situation now')
        plt.legend()
        plt.xlabel('Code Health')
        plt.ylabel('added value in M')
    

# big fuction that returns everything for the 3rd degree polynomial model
def poly_Model(df,value_creation_old,CH_old,unplanned_work_old):
    bugs=bug_Poly_Regression(df)
    time= time_Poly_Regression(df)
    plot_Bug_Regression(df,bugs)
    plot_Time_Regression(df,time)
    #plot_Poly_Added_Value(value_creation_old,CH_old,unplanned_work_old,bugs,time)

########################################################## Plot functions ############################################################
# takes the df and the coefficients of the regression for bugs in fct of CH 
# plots the regression with errorbars and with the scatter plot
def plot_Bug_Regression(df, p):
    # Definitions ############################################
    df_binned=bin_ch(df)
    x_bugs=df['code_health'].to_numpy()
    y_bugs=df['total_defects'].to_numpy()
    
    # Mean & Standard Deviation ############################################
    mean_df=df_binned.groupby('code_health')['total_defects'].mean().reset_index()
    mean_df.rename(columns={"total_defects": "defects_mean"},inplace=True)
    std_df=df_binned.groupby('code_health')['total_defects'].std().reset_index()
    std_df.rename(columns={"total_defects": "defects_std"},inplace=True)
    
    mean=mean_df['defects_mean'].to_numpy()
    std=std_df['defects_std'].to_numpy() 

    # Repartition ############################################
    rep=df_binned.groupby('code_health')['code_health'].count().rename_axis('CH').reset_index()
    rep.rename(columns={"code_health": "nb_files", "CH": "code_health"},inplace=True)
    rep['distribution']=np.round(rep['nb_files']/df.shape[0],3)
    
    # Plot ###############################
    # Predicition ###############################
    x_new=np.arange(1,11)
    y_pred=np.polyval(p, x_new)
    
    
    plt.plot(x_new,y_pred)
    plt.errorbar(mean_df['code_health'].to_numpy(),mean,std, linestyle='None', marker='o')
    plt.xlabel('Code Health')
    plt.ylabel('#defects')
    plt.show()

    plt.plot(x_new,y_pred)#, label='prediction')
    #plt.bar(rep['code_health'],rep['distribution']*10,width=0.5, color='lightsteelblue', alpha=0.3, edgecolor='b', label='data repartition')
    #plt.legend()
    plt.xlabel('Code Health')
    plt.ylabel('#defects')
    plt.show()

    plt.plot(x_new,y_pred, color='r')
    plt.xlabel('Code Health')
    plt.ylabel('#bugs')
    plt.scatter(x_bugs, y_bugs, color='darkcyan')
    plt.errorbar(mean_df['code_health'].to_numpy(),mean,std, linestyle='None', color='y')
    plt.show()

# takes the df and the coefficients of the regression for time-in-dev in fct of CH 
# plots the regression with errorbars and with the scatter plot
def plot_Time_Regression(df, p):
    # Definitions ############################################
    df_binned=bin_ch(df)
    x=df[df['lead_time_minutes'].notna()]['code_health'].to_numpy()
    y=df[df['lead_time_minutes'].notna()]['lead_time_minutes'].to_numpy()
    x_reshaped=x.reshape(-1, 1)
    
    # Mean & Standard Deviation ############################################
    mean_df=df_binned.groupby('code_health')['lead_time_minutes'].mean().reset_index()
    mean_df.rename(columns={"lead_time_minutes": "time_mean"},inplace=True)
    std_df=df_binned.groupby('code_health')['lead_time_minutes'].std().reset_index()
    std_df.rename(columns={"lead_time_minutes": "time_std"},inplace=True)

    mean=mean_df['time_mean'].to_numpy()
    std=std_df['time_std'].to_numpy()

    # Repartition ############################################
    rep=df_binned.groupby('code_health')['code_health'].count().rename_axis('CH').reset_index()
    rep.rename(columns={"code_health": "nb_files", "CH": "code_health"},inplace=True)
    rep['distribution']=np.round(rep['nb_files']/df.shape[0],3)
    
    # Plot ###############################
    # Predicition ###############################
    x_new=np.arange(1,11)
    y_pred =np.polyval(p, x_new)
    
    plt.plot(x_new,y_pred)
    plt.errorbar(mean_df['code_health'].to_numpy(),mean,std, linestyle='None', marker='o')
    plt.xlabel('Code Health')
    plt.ylabel('average debugging time')
    plt.show()

    plt.plot(x_new,y_pred)#, label='prediction')
    #plt.bar(rep['code_health'],rep['distribution']*25000,width=0.5, color='lightsteelblue', alpha=0.3, edgecolor='b', label='data repartition')
    #plt.legend()
    plt.xlabel('Code Health')
    plt.ylabel('average debugging time')
    plt.show()

    plt.plot(x_new,y_pred, color='r')
    plt.xlabel('Code Health')
    plt.ylabel('average debugging time')
    plt.scatter(x, y, color='darkcyan')
    plt.errorbar(mean_df['code_health'].to_numpy(),mean,std, linestyle='None', color='y')
    plt.show()
    






