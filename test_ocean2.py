
import json
import pickle
import os, sys
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def get_input(local=False):
    if local:
        print("Reading local file dataset_61_iris.csv")

        return "dataset_61_iris.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename

    
    
# local=False
def catalonia_analysis(local=False):
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")

    filename = get_input(local)
    df = pd.read_csv(filename,index_col=None)
    df.DATA = pd.to_datetime(df.DATA, dayfirst = True)


    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    # df[['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
    #        '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
    #        '19h', '20h', '21h', '22h', '23h', '24h']] = pd.DataFrame(imp_mean.fit_transform(df[['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
    #        '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
    #        '19h', '20h', '21h', '22h', '23h', '24h']]))
    df['month'] = df['DATA'].dt.month_name()
    df['year'] = pd.to_datetime(df['DATA']).dt.year


    print(df.CONTAMINANT.unique())
    print(df['month'].unique())
    hours=['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
           '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
           '19h', '20h', '21h', '22h', '23h', '24h']
    df0 = df.groupby(['CONTAMINANT','MUNICIPI']).apply(lambda x: x[hours].mean().mean())
    df_=df0.reset_index()
    df_.columns = ['CONTAMINANT','City', 'Contamination_level']
    df_ = df_[df_['Contamination_level'] != df_.Contamination_level.max()] # let's remove outlier
    cities_rank = {}
    #pd.DataFrame()
    for contaminant in df_.CONTAMINANT.unique():
        if len(df_.loc[df_.CONTAMINANT == contaminant]) < 6: # do not make chart of non representative samples
            continue
        else:   
            fig, ax = plt.subplots(1, 1, figsize = (6, 3))#, dpi=300)
            sb.barplot(data=df_.loc[df_.CONTAMINANT == contaminant], x='City', y='Contamination_level').set_title("Contamination by "+contaminant)
            plt.xticks(rotation=90)
            ax.set_xlabel('')
            print('Top 5 most polluted cities by', contaminant, 'are: ',', '.join(df_.loc[df_.CONTAMINANT == contaminant].sort_values(by='Contamination_level')[-5:].City.tolist()))
            print('Top 5 lowest polluted cities by', contaminant, 'are: ',', '.join(df_.loc[df_.CONTAMINANT == contaminant].sort_values(by='Contamination_level')[:5].City.tolist()))
            plt.show()
            cities_rank[contaminant] = {}
            cities_rank[contaminant] = df_.loc[df_.CONTAMINANT == contaminant].sort_values(by='Contamination_level')[-5:]

    c=np.array([k[1]['City'] for k in cities_rank.items()]).ravel()
    lists = sorted(dict(zip(list(c),[list(c).count(i) for i in list(c)])).items()) # sorted by key, return a list of tuples
    x, y = zip(*lists)
    c=pd.DataFrame(lists)
    c.columns = ['City', 'Count']
    sb.pointplot(c.sort_values(by='Count',ascending=0 ),x='City',y='Count')
    plt.xticks(rotation=90)
    plt.show()
    print("Top most air polluted Cities are: ", ', '.join(c.sort_values(by='Count',ascending=0 ).City[:6].to_list()) )
    print("Less air polluted Cities are: ", ', '.join(c.sort_values(by='Count',ascending=0 ).City[-5:].to_list()) )

    df0 = df.groupby(['CONTAMINANT','ALTITUD']).apply(lambda x: x[hours].mean().mean())
    for contaminant in df.CONTAMINANT.unique():
        df_ = df0[df0.index.get_level_values('CONTAMINANT') == contaminant]
        df_ = df_.reset_index()[['ALTITUD',0]]
        df_.columns = ['ALTITUD', 'contamination_level']
        scaler = MinMaxScaler()
        if len(df_)<2:
            continue
        else:
            df_['scaled_altitude'] = scaler.fit_transform(df_[['ALTITUD']])
            sb.pairplot(df_[['scaled_altitude', 'contamination_level' ]],kind="kde")#.set_title("Contamination pairplot of "+ contaminant)
            print(contaminant)
            plt.show()

    df0 = df.groupby(['CONTAMINANT','AREA URBANA']).apply(lambda x: x[hours].mean().mean())
    for contaminant0, contaminant1  in zip(df.CONTAMINANT.unique()[0::2],df.CONTAMINANT.unique()[1::2]):
        fig, (ax1, ax2) = plt.subplots(figsize=(5, 3), ncols=2, sharex=False, sharey=False)
        labels = df0[df0.index.get_level_values('CONTAMINANT') == contaminant0].unstack(level=0).index#'rural', 'suburban', 'urban'
        sizes = df0[df0.index.get_level_values('CONTAMINANT') == contaminant0].unstack(level=1).values[0]
        ax1.set_title(contaminant0 +' areas share')
        ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('scaled')   
        labels = df0[df0.index.get_level_values('CONTAMINANT') == contaminant1].unstack(level=0).index#'rural', 'suburban', 'urban'
        sizes = df0[df0.index.get_level_values('CONTAMINANT') == contaminant1].unstack(level=1).values[0]
        # fig1, ax2 = plt.subplots(figsize=(4, 3))
        ax2.set_title(contaminant1 +' areas share')
        ax2.pie(sizes,  labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax2.axis('scaled')  
        plt.show()

    import seaborn as sb
    import seaborn.objects as so
    import matplotlib.pyplot as plt
    df0 = df.groupby(['CONTAMINANT','year']).apply(lambda x: x[hours].mean())
    sb.set(font_scale = .7)
    hour_statistics = {}
    month_statistics = {}
    for contaminant in df['CONTAMINANT'].unique():
        hour_statistics[contaminant] = {}
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2, sharex=False, sharey=False)
        sb.stripplot(df0[df0.index.get_level_values('CONTAMINANT') == contaminant],orient='h',ax=ax1).set_title("Contamination by hour Level "+ contaminant)
        # plt.ylabel('Contamination Level')
        plt.xlabel('Contamination Level')
        sb.kdeplot(df0[df0.index.get_level_values('CONTAMINANT') == contaminant],ax=ax2).set_title("Contamination by hour KDE "+ contaminant)
        plt.xlabel('Contamination Level')  
        hours_data = df0[df0.index.get_level_values('CONTAMINANT') == contaminant]
        hour_statistics[contaminant] = hours_data.mean()[hours_data.mean() == hours_data.mean().max()].index[0]
        print('Most contaminated Hour of '+contaminant+' Mean level is: ', hour_statistics[contaminant])
        plt.show()
    df0 = df.groupby(['CONTAMINANT','year','month']).apply(lambda x: x[hours].mean().mean())
    df0 = df0.unstack(level=2)
    for contaminant in df['CONTAMINANT'].unique():#df0.index.get_level_values('CONTAMINANT').unique():
        month_statistics[contaminant] = {}
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2, sharex=False, sharey=False)
        sb.stripplot(df0[df0.index.get_level_values('CONTAMINANT') == contaminant],orient='h',ax=ax1).set_title("Contamination by Month average Level "+ contaminant)
        plt.xlabel('Contamination Level')
        sb.kdeplot(df0[df0.index.get_level_values('CONTAMINANT') == contaminant],ax=ax2).set_title("Contamination by Month KDE "+ contaminant)
        # plt.ylabel('Contamination Level')  
        plt.xlabel('Contamination Level')  
        month_data = df0[df0.index.get_level_values('CONTAMINANT') == contaminant]
        month_statistics[contaminant] = month_data.mean()[month_data.mean() == month_data.mean().max()].index[0]
        print('Most contaminated Month '+contaminant+' is: ', month_statistics[contaminant])
        plt.show()

    c=[k for k in hour_statistics.items()]
    print(pd.DataFrame(np.vstack(c)[:,1]).value_counts()[:3])
    c=[k for k in month_statistics.items()]
    print(pd.DataFrame(np.vstack(c)[:,1]).value_counts()[:4])

if __name__ == "__main__":
    local=False
#     local = len(sys.argv) == 2 and sys.argv[1] == "local"
    catalonia_analysis(local)

