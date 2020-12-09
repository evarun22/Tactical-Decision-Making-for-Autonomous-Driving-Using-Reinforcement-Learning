import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('colab.txt')
print(df.head(20))
df['mean_30'] = df.iloc[:,1].rolling(window=30).mean()
#plt.plot(df['episode'],df[''])
#df.to_csv('check.csv')
plt.plot(df['episode'],df['mean_30'])
plt.title('DDPG_KinematicsObservation_ContinuousActions')
plt.xlabel('episodes')
plt.ylabel('average score of last 30 episodes')
plt.savefig('image.png')