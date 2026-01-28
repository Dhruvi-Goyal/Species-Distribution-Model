import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Read the CSV file
df = pd.read_csv("india_similarities.csv")

# Add ranks for columns used in rank correlation
df['Rank_Euclidean_Similarity'] = df['Euclidean_Similarity'].rank()
df['Rank_Average_Probability_Euclidean'] = df['Average_Probability'].rank()
df['Rank_Cosine_Similarity'] = df['Cosine_Similarity'].rank()
df['Rank_Average_Probability_Cosine'] = df['Average_Probability'].rank()

# Compute Spearman rank correlation for each pair
rank_corr_euclidean, _ = spearmanr(df['Euclidean_Similarity'], df['Average_Probability'])
rank_corr_cosine, _ = spearmanr(df['Cosine_Similarity'], df['Average_Probability'])

# Add the correlation values as new columns
df['Rank_Correlation_Euclidean'] = rank_corr_euclidean
df['Rank_Correlation_Cosine'] = rank_corr_cosine

# Save the updated DataFrame to a new CSV file
df.to_csv("india_similarities_rank_correlation.csv", index=False)

# Scatter Plot: Euclidean_Similarity vs Average_Probability
plt.figure(figsize=(10, 6))
plt.scatter(df['Euclidean_Similarity'], df['Average_Probability'], color='blue', label='Euclidean Similarity')
plt.title('Euclidean Similarity vs Average Probability')
plt.xlabel('Euclidean Similarity')
plt.ylabel('Average Probability')
plt.grid(True)
plt.legend()
plt.savefig("euclidean_vs_probability.png")  # Save plot as a file
plt.show()

# Scatter Plot: Cosine_Similarity vs Average_Probability
plt.figure(figsize=(10, 6))
plt.scatter(df['Cosine_Similarity'], df['Average_Probability'], color='green', label='Cosine Similarity')
plt.title('Cosine Similarity vs Average Probability')
plt.xlabel('Cosine Similarity')
plt.ylabel('Average Probability')
plt.grid(True)
plt.legend()
plt.savefig("cosine_vs_probability.png")  # Save plot as a file
plt.show()

print("Scatter plots saved as 'euclidean_vs_probability.png' and 'cosine_vs_probability.png'")
