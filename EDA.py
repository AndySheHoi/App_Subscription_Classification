import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

class EDA:
    
    # =========================================================================
    # Exploratory Data Analysis
    # =========================================================================
    def eda(df): 
        # Viewing the Data
        df.size      # 600000
        df.head(5)
        
        # Distribution of Numerical Variables
        df.describe()
        
        # First set of Feature cleaning
        df["hour"] = df.hour.str.slice(1, 3).astype(int)
        
        df2 = df.copy().drop(columns = ['user', 'screen_list', 'enrolled_date',
                                                   'first_open', 'enrolled'])
        
        # Histograms
        plt.suptitle('Histograms of Numerical Columns', fontsize=20)
        
        for i in range(1, df2.shape[1] + 1):
            plt.subplot(3, 3, i)
            f = plt.gca()
            f.set_title(df2.columns.values[i - 1])
            vals = np.size(df2.iloc[:, i - 1].unique())    
            plt.hist(df2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.savefig('app_data_hist.jpg')
        
        ## Correlation with Response Variable
        df2.corrwith(df.enrolled).plot.bar(figsize=(20,10),
                          title = 'Correlation with Reposnse variable',
                          fontsize = 15, rot = 45, grid = True)
        
        
        # Correlation Matrix
        sn.set(style="white", font_scale=2)
        
        # Compute the correlation matrix
        corr = df2.corr()
        
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(18, 15))
        f.suptitle("Correlation Matrix", fontsize = 40)
        
        # Generate a custom diverging colormap
        cmap = sn.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        
        # =====================================================================
        # Feature Engineering
        # =====================================================================
        
        # Formatting Date Columns
        df.dtypes
        # 18926 rows of enrolled_date are null
        df.isnull().sum() 
        df["first_open"] = pd.to_datetime(df["first_open"])
        df["enrolled_date"] = pd.to_datetime(df["enrolled_date"])
        
        
        # Selecting Time For Response
        df["difference"] = (df.enrolled_date-df.first_open).astype('timedelta64[h]')
        response_hist = plt.hist(df["difference"].dropna(), color='#3F5D7D')
        plt.title('Distribution of Time-Since-Screen-Reached')
        plt.show()
        
        # the major population enrolled within the first 10 hours
        plt.hist(df["difference"].dropna(), color='#3F5D7D', range = [0, 100])
        plt.title('Distribution of Time-Since-Screen-Reached')
        plt.show()
        
        # change all the enrolled time which is larger than 48 hours to 0
        df.loc[df.difference > 48, 'enrolled'] = 0
        df = df.drop(columns=['enrolled_date', 'difference', 'first_open'])
        
        
        # Formatting the screen_list Field
        
        # Load Top Screens
        top_screens = pd.read_csv('top_screens.csv').top_screens.values
        
        # Mapping Screens to Fields
        df["screen_list"] = df.screen_list.astype(str) + ','
        
        # get_dummies
        # add all top_screens columns to the df
        for sc in top_screens:
            # if the screen_list contains the sc, df[sc] = 1, else = 0
            df[sc] = df.screen_list.str.contains(sc).astype(int)
            # remove the sc from the screen_list
            df['screen_list'] = df.screen_list.str.replace(sc+",", "")
        
        # after removed all top_screen, count the amount of other screen
        df['Other'] = df.screen_list.str.count(",")
        df = df.drop(columns=['screen_list'])
        
        # Feature fusion
        savings_screens = ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5",
                           "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"]
        df["SavingCount"] = df[savings_screens].sum(axis=1)
        df = df.drop(columns=savings_screens)
        
        cm_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"]
        df["CMCount"] = df[cm_screens].sum(axis=1)
        df = df.drop(columns=cm_screens)
        
        cc_screens = ["CC1", "CC1Category", "CC3"]
        df["CCCount"] = df[cc_screens].sum(axis=1)
        df = df.drop(columns=cc_screens)
        
        loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]
        df["LoansCount"] = df[loan_screens].sum(axis=1)
        df = df.drop(columns=loan_screens)
        
        #Saving Results
        df.head()
        df.describe()
        df.columns
        
        return df
