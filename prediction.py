import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filenames = ["season-0910_csv.csv","season-1011_csv.csv","season-1112_csv.csv","season-1213_csv.csv",
             "season-1314_csv.csv","season-1415_csv.csv","season-1516_csv.csv","season-1617_csv.csv",
             "season-1718_csv.csv","season-1819_csv.csv"]

df_all_games = pd.concat( [ pd.read_csv(f,index_col = "Date") for f in filenames ] )

# On filtre notre dataset pour n'afficher que les matchs du PSG

df_psg = df_all_games.query('HomeTeam == "Paris SG" | AwayTeam == "Paris SG"')
print("\n----------------------------------------------------------------------\n")
print(df_psg)
print("\n----------------------------------------------------------------------\n")



# On affiche tous les matchs du PSG, leur nombre de buts marqués et encaissés, indexés par date
df1 = df_psg.loc[:,["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
print(df1)
print("\n----------------------------------------------------------------------\n")


#c'est intéressant de regarder le nombre de victoires, de défaites et de matchs nuls entre 2009 et 2019
def displayWinsPSG(home,draw,away):

    for m in range(380):
        if df_psg['HomeTeam'][m] == 'Paris SG':
            if df_psg['FTR'][m] == "H":
                home = home+1
            if df_psg['FTR'][m] == "D":
                draw = draw+1
        else:
            if df_psg['AwayTeam'][m] == 'Paris SG':
                if df_psg['FTR'][m] == "A":
                    away = away+1
                if df_psg['FTR'][m] == "D":
                    draw = draw+1
    print("Résultats du PSG : \n")
    print("Victoire à domicile : ", home, " Victoire à l'extérieur : ",away, " Matchs nuls : ", draw)
    return home, away, draw


home = 0
draw = 0
away = 0
home,away,draw = displayWinsPSG(home,draw,away)

names = ['Home', 'Away', 'Draw']
values = [home, away, draw]



plt.bar(names, values)
plt.suptitle('Nombre de victoires à domicile, à l\'extérieur et nombre de matchs nuls entre 2009-2010 et 2018-2019')
plt.show()
print("\n----------------------------------------------------------------------\n")



#L'idée est qu'on voudrait utiliser une loi de poisson pour prédire 
# le score du prochain match du PSG

#On commence par chercher la moyenne de buts en 90 minutes dans un match de ligue 1

df_all_games['total_goals']=df_all_games['FTHG']+df_all_games['FTAG']

print("Moyenne de buts par match en ligue 1 entre 2009 et 2019 : ", df_all_games.total_goals.mean())
print("\n----------------------------------------------------------------------\n")

# on affiche les données pour mieyux visualiser le nombre de buts


# nombre de buts par l'équipe à domicile
import matplotlib.pyplot as plt
#%matplotlib inline
fig = plt.figure(figsize = (10,5))
ax = fig.gca()
plt.hist(df_all_games.FTHG,edgecolor='black')
plt.xticks(range(15))
print("\n----------------------------------------------------------------------\n")


#nombre total de buts dans un match
import matplotlib.pyplot as plt
df_all_games['total_goals']=df_all_games['FTHG']+df_all_games['FTAG']
fig = plt.figure(figsize = (7,5))
ax = fig.gca()
plt.hist(df_all_games.total_goals,edgecolor='black')
plt.xticks(range(15))
print("\n----------------------------------------------------------------------\n")


# La loi de Poisson : P(x; μ) = (e-μ) (μx) / x!
#lam = 2.517894736842105
from scipy.special import factorial
#k est ici le nombre de buts dont on voudra calculer la probabilité

def poisson(k,exp_events):
    lam =(exp_events)
    p_k= np.exp(-lam)*np.power(lam,k)/factorial(k)
    #print(f'The probability of {k} goals in {minutes} minutes is {100*p_k:.2f}%.')
    return p_k

#proability of goals acc to poisson distribution
k=[]
p_k=[]
for i in range(10):
    p_k.append(poisson(i,2.517894736842105)*100)
    k.append(i)


fig = plt.figure()
plt.plot(k,p_k,'o-')
plt.xticks(range(10))
fig.suptitle('Probabilité de buts dans un match',fontsize=15)
plt.xlabel('Nombre de buts')
plt.ylabel('Probabilité en pourcentages')
for x,y in zip(k,p_k):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset pixels", # how to position the text
                 xytext=(0,3), # distance from text to points (x,y)
                 fontsize=12)
print("\n----------------------------------------------------------------------\n")




# In 10000 matches simulation goals scored distribution
N=10000
lam=2.517894736842105
counts=np.random.poisson(lam,size=N)
df=pd.DataFrame(counts)
df.hist(bins=11,edgecolor='black')
plt.xticks(range(15))
plt.grid(False)
print("\n----------------------------------------------------------------------\n")



def p_lessorequal(n_query,exp_events,quiet=True):
    p_n=poisson(np.arange(100),exp_events)
    p=p_n[:n_query+1].sum()
    if quiet:
        return p
    else:
        print(f'Probabilité de marquer {n_query} buts ou moins en un match: {100*p:.2f}%.')
    
def p_greaterorequal(n_query,exp_events,quiet=True):
    p = 1 - p_lessorequal(n_query,exp_events)
    if quiet:
        return p
    else:
        print(f'Probabilité de marquer plus de {n_query} en un match: {100*p:.2f}%.')



#probabilité de marquer plus ou moins de buts qu'une certaine valeur 

for i in range(1,10):
    print(p_lessorequal(i,lam,False))

for i in range(1,10):
    print(p_greaterorequal(i,lam,False))
    

print("\n----------------------------------------------------------------------\n")




prob=[]
for i in range(10):
    p = p_greaterorequal(i,2.517894736842105,True)
    prob.append(p*100)
fig = plt.figure()
plt.plot(k,prob,'o-y')
plt.xticks(range(10))
fig.suptitle("Probabilité qu'il y ait plus de n buts dans un match",fontsize=15)
plt.xlabel("n buts")
plt.ylabel('Probabilité en pourcentages')
for x,y in zip(k,prob):

    label = "{:.2f}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset pixels", # how to position the text
                 xytext=(0,4), # distance from text to points (x,y)
                 ha='left',fontsize=10)
plt.grid(False)
print("\n----------------------------------------------------------------------\n")




#prediction du score via la loi de poisson

import pandas as pd
import numpy as np
from scipy import stats 

def PredictScore():
    
    try:
        ht = input("Entrez une équipe de ligue 1 à domicile : ")
        at = input("Entrez une équipe de ligue 1 à l'extérieur : ")
            
        if len(df_all_games[(df_all_games.HomeTeam ==ht) & (df_all_games.AwayTeam ==at)]) > 5:
            
            avg_home_score = df_all_games[(df_all_games.HomeTeam ==ht) & (df_all_games.AwayTeam ==at)].FTHG.mean()
            avg_away_score = df_all_games[(df_all_games.HomeTeam ==ht) & (df_all_games.AwayTeam ==at)].FTAG.mean()
            
            home_goal = int(stats.mode(np.random.poisson(avg_home_score,100000))[0])                    
            away_goal = int(stats.mode(np.random.poisson(avg_away_score,100000))[0])

        else:
            avg_home_goal_conceded = df_all_games[(df_all_games.HomeTeam ==ht)].FTAG.mean()
            avg_away_goal_scored   = df_all_games[(df_all_games.AwayTeam ==at)].FTAG.mean()
            away_goal = int(stats.mode(np.random.poisson(1/2*(avg_home_goal_conceded+avg_away_goal_scored),100000))[0])

            avg_away_goal_conceded = df_all_games[(df_all_games.HomeTeam ==at)].FTHG.mean()
            avg_home_goal_scored   = df_all_games[(df_all_games.AwayTeam ==ht)].FTHG.mean()
            home_goal = int(stats.mode(np.random.poisson(1/2*(avg_away_goal_conceded+avg_home_goal_scored),100000))[0])
            
        avg_total_score = int(stats.mode(
            np.random.poisson((df_all_games[(df_all_games.HomeTeam ==ht) & (df_all_games.AwayTeam ==at)].total_goals.mean()),100000))[0])
        
        print(f'Nombre de buts attendus {avg_total_score}')
        print(f'Il y a eu {len(df_all_games[(df_all_games.HomeTeam ==ht) & (df_all_games.AwayTeam ==at)])} matchs entre 2009-2010 et 2018-2019')
        print(f'Le score est : {ht} {home_goal}:{away_goal} {at}')
    except:
        print("Equipe introuvable. Veuillez réessayer")
print("\n----------------------------------------------------------------------\n")


PredictScore()




















