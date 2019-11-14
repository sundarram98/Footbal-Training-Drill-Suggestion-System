
# coding: utf-8

# <h3> Football Drill Suggestion </h3>
# 
# 
# <p> 
# This program is used to analyze the data of 18,000+ soccer players from European Soccer leagues. The dataset used here is an open European Soccer Database downloaded from Kaggle. </p>
# 
# <b> <a href="https://www.kaggle.com/hugomathien/soccer/version/10"> Description of dataset </a> </b> 
# 
# <p>
# Here, Players are analysed and classified into different types first using ml algorithms. Next, depending on the players type and skill level, drills are suggested
# </p>

# In[1]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


# In[2]:


# Getting data from dataset and putting it in a dataframe, it is in the form of a database
conn = sqlite3.connect('database.sqlite')
df = pd.read_sql_query('select * from Player_Attributes',conn)


# In[3]:


df.shape


# In[4]:


# Cleaning dataframe by dropping empty rows
df = df.dropna()
df.shape


# In[5]:


# Get columns in database
print(50*"_")
print ('\nVarious attributes of a player:')
print(50*"_")
for i in df.columns:
    print (i)


# In[6]:


# Getting basic stat details about the database 
print()
print(90*"_")
print('\nStatistical values of the dataframe:')
print(90*"_")
print (df.describe().transpose())


# <h1> Correlation Analysis </h1>
# 
# <p> Here, <b> overall_rating </b> is defined as the most significant attribute tp identify a player. So the idea used is to perform correlation between all player attributes and player rating to identify which features correlate the most with player rating and use those for the purpose of clustering. The correlation coefficient used in pearson's coefficient </p>
# 
# <h3> Peasron's Correlation Coefficien </h3> 
# 
# <p> The Pearson's Correlation coefficient ranges from -1 to 1. A correlation of -1 indicates that the attributes are inversely related to each other. A correlation of 0 indicates that no relation exist between the attributes. If correlation is nearer to positive side of 0, then there exists a weak correlation between the attributes, and closer to 1 indicates the presence of strong correlation. A correlation coefficient of 1 indicates perfect correlation between the attributes.
# 
# For our analysis, we only consider the attributes which are either weakly or strongly correlated to the overall_rating of the player. Attributes with negative correlation coefficient or correlation coefficient closer to 0 are neglected to focus only on the prime factors which decide the overall_rating of the player. </p>

# In[7]:


# Randomizing order of data
df = df.reindex(np.random.permutation(df.index))


# In[8]:


# Following attributes of the players are considered to be either weakly or 
features = ['potential', 'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve', 
            'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions',
            'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression', 'interceptions', 
            'positioning', 'vision', 'penalties', 'standing_tackle', 'sliding_tackle',
            'gk_diving', 'gk_handling','gk_kicking', 'gk_positioning', 'gk_reflexes']


# In[9]:


# computing the correlation of each attribute with overall_rating
correlations = [ df['overall_rating'].corr(df[f]) for f in features ]


# In[10]:


# Drawing pyplot of overall_rating vs all features
def plot_correlation(df,x_label,y_label):
    fig = plt.gcf()
    fig.set_size_inches(20,15)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    ax = df.correlation.plot(linewidth=3.3,color = 'brown')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.attributes,rotation=70)
    plt.title('Correlation of features with Overall Rating')
    plt.show()
    
df2 = pd.DataFrame({'attributes' : features, 'correlation' : correlations })
plot_correlation(df2,'Features','Player\'s Overall rating')


# <h2> Analysis from graph </h2> 
# 
# <p> The peaks in the graph show that those attributes are strongly correlated with the overall_rating of the player. Hence, the following attributes should be considered while rating any player: </p> 
# 
# <ul>
#     <li> potential </li>
#     <li> reactions </li>
#     <li> shortpassing </li>
#     <li> ball control </li>
#     <li> vision </li>    
# </ul>

# <h4> These features can be used for the purpose of clustering </h4> 
# 
# <h3> Clustering Players </h3> 
# 
# <p> 
# In general, a football team typically consists of 11 players. Apart from the goalkeeper, there are 10 other players who are spread across the field according to the team's formation. The 10 players fall into either of the four categories:
# </p>
# 
# <ul>
#     <li>Defenders</li>
#     <li>Midfielders</li>
#     <li>Wingers</li>
#     <li>Centre Forwards</li>
# </ul>

# In[11]:


# Justification for using features


# <p>
#     
# Players falling into each of this four categories have unique attributes which makes them to standout in their particular category. For instance, the defenders are supposed to have strong interception capablities to intrude into opposition's attack, while midfielders cover the most part of ground which is squarely attributed to their potential. Basically the attackers are further classified into wing attackers and center forwards. As their name suggests wing attackers and center forwards should have good reaction and ball control capablilites to embrace their positions in the team respectively. So, we cluster players based on these five attributes:
# 
# </p>
# 
# <ol>
#     <li> Diving (goal keeper) </li>
#     <li> Interceptions </li>
#     <li> Potential </li>
#     <li> Reaction </li>
#     <li> Ball control </li>
# </ol>

# In[12]:


# Potential features of the players in each group
groupFeatures= ['gk_diving','interceptions', 'potential',  'reactions', 'ball_control']

print(50*"_")
print ("\nCorrelation Analysis of these grouping features:")
print(50*"_")
for f in groupFeatures:
    related = df['overall_rating'].corr(df[f])
    print ("%s : %.2f" % (f,related))


# In[13]:


#Generating a new dataframe from the features which are defined as group features
df_select = df[groupFeatures].copy(deep=True)

print(90*"_")
print ("\nNew DataFrame :")
print(90*"_")
print (df_select)

#Perform scaling on the dataframe containing the features
groups = scale(df_select)


# In[14]:


# Using k-means clustering to group into 5 sets of players


# In[15]:


# Define number of clusters#Define n 
clusters = 5


# In[16]:


# Train a model using KMeans() machine learning method
model = KMeans(init='k-means++',n_clusters=clusters,n_init=20).fit(groups)


# In[17]:


# Counting the number of players in each cluster
print(60*"_")
print("\nCount of players in each cluster : ")
print(60*"_")
pd.value_counts(model.labels_,sort=False)


# In[18]:


# Create a compostite dataframe for plotting the newly formed dataframe#Create a 
# Using custom function from customplot module which we import at the beginning of the program
df3 = model.cluster_centers_

print(90*"_")
print ("\nComposite DataFrame :")
print(90*"_")
print("\ngk_diving  Interceptions  potential  reactions  ball_control  prediction\n")
for i in range(5) :
    print(df3[i], "   ", i)


# In[19]:


# For plotting graph inside notebook we use matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


# Plotting the five groups of players in the same graph to analyse what features they share in common with each other
fig = plt.gcf()
fig.set_size_inches(15,10)
for i in range(5) :
    plt.plot(df3[i], label=str(i)) 
plt.legend(loc='upper right')
x=[0.0,1.0,2.0,3.0,4.0]
plt.xticks( x, groupFeatures)
plt.grid()
plt.show()


# <h2> Analysis from this graph </h2>
# 
# <p>
#     <ul> 
#         <li> Two groups (*red and black*) are similar to each other except for interceptions capablities- these groups can coach each other in interceptions and where they differ </li>
#         <li> Two groups (*green and yellow*) seem to be equally talented in potential and reactions. These groups can coach each other in ball control, interceptions and gk_diving. </li>
#     </ul>
# </p>
