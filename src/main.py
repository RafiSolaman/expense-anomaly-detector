import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import KMeans


# Generate Synthetic Data
def generate_expenses(num_days=180): 
    categories = ['Food', 'Transport', 'Shopping', 'Entertainment', 'Bills']
    start_date = datetime(2024,1,1)

    data = []
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        for _ in range(random.randint(1, 4)):  # 1-4 expenses per day
            category = random.choice(categories)
            # Normal spending patterns
            if category == 'Food':
                amount = round(np.random.normal(15, 5), 2)
            elif category == 'Transport':
                amount = round(np.random.normal(7, 2), 2)
            elif category == 'Shopping':
                amount = round(np.random.normal(50, 30), 2)
            elif category == 'Entertainment':
                amount = round(np.random.normal(25, 15), 2)
            else:  # Bills
                amount = round(np.random.normal(120, 20), 2)
            # Occasionally inject anomalies (big spending)
            if random.random() < 0.02:
                amount *= random.randint(5, 10)
            data.append([date, category, max(0.5, amount)])
    return pd.DataFrame(data, columns=['date','category','amount'])


# Create and save
df = generate_expenses(num_days=365)
csv_path = "data/synthetic_expenses.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Generated dataset with {len(df)} transactions saved at {csv_path}")


# Statistical Anomaly Detection
mean = df["amount"].mean()
std = df["amount"].std()
threshold = mean + 2 * std
df['stat_anomaly'] = df['amount'] > threshold


# K-Means Clustering
X = df[['amount']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)


# The cluster with highest average spending = anomalies
cluster_means = df.groupby('cluster')['amount'].mean()
anomaly_cluster = cluster_means.idxmax()
df['kmeans_anomaly'] = df['cluster'] == anomaly_cluster


# Print Sample Anomalies
print("\nStatistical anomalies (sample):")
print(df[df['stat_anomaly']].head())

print("\nK-Means anomalies (sample):")
print(df[df['kmeans_anomaly']].head())


# Plot Results
plt.figure(figsize=(14,7))
plt.plot(df['date'], df['amount'], marker='o', linestyle='-', label='Expenses')

# Highlight statistical anomalies
stat_anomalies = df[df['stat_anomaly']]
plt.scatter(stat_anomalies['date'], stat_anomalies['amount'],
            color='red', s=100, marker='x', label='Statistical Anomalies')

# Highlight K-Means anomalies
kmeans_anomalies = df[df['kmeans_anomaly']]
plt.scatter(kmeans_anomalies['date'], kmeans_anomalies['amount'],
            color='orange', s=100, marker='D', label='K-Means Anomalies')

# Format x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)

plt.title('Expense Anomaly Detection (Statistical vs. K-Means)')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.legend()
plt.tight_layout()
plt.show()
