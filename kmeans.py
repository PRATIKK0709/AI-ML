import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r'./best-selling game consoles.csv')
df['Discontinuation Year'].replace(0, df['Discontinuation Year'].max(), inplace=True)
plt.scatter(df['Released Year'], df['Units sold (million)'])
plt.xlabel('Released Year')
plt.ylabel('Units Sold (millions)')
plt.show()

scaler = MinMaxScaler()
df[['Released Year', 'Units sold (million)']] = scaler.fit_transform(df[['Released Year', 'Units sold (million)']])
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Released Year', 'Units sold (million)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()

k = 3  
km = KMeans(n_clusters=k)
df['cluster'] = km.fit_predict(df[['Released Year', 'Units sold (million)']])


plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_df = df[df['cluster'] == i]
    plt.scatter(cluster_df['Released Year'], cluster_df['Units sold (million)'], label=f'Cluster {i}')


plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker='*', label='Centroids')
plt.xlabel('Released Year')
plt.ylabel('Units Sold (millions)')
plt.legend()
plt.show()
