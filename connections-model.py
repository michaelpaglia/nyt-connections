from scipy.cluster.hierarchy import linkage, fcluster
import gensim.downloader
import numpy as np

google_news = gensim.downloader.load('word2vec-google-news-300')
file_path = 'word-list.txt'

with open(file_path, 'r') as file:
    word_dict = [word.strip() for word in file]


def find_similarity(word_1, word_2):
    semantic_similarity = google_news.similarity(word_1, word_2)
    connection_similarity = max(
        google_news.similarity(word_1, f"{word_1} {word_2}") if f"{word_1} {word_2}" in google_news.key_to_index else 0,
        google_news.similarity(word_2, f"{word_1} {word_2}") if f"{word_1} {word_2}" in google_news.key_to_index else 0,
        google_news.similarity(word_1, f"{word_2} {word_1}") if f"{word_2} {word_1}" in google_news.key_to_index else 0,
        google_news.similarity(word_2, f"{word_2} {word_1}") if f"{word_2} {word_1}" in google_news.key_to_index else 0
    )
    return max(semantic_similarity, connection_similarity)


similarity_matrix = np.array([[find_similarity(w1, w2) for w2 in word_dict] for w1 in word_dict])

# Perform hierarchical clustering
linkage_matrix = linkage(similarity_matrix, method='complete')      # Farthest Point Algorithm
cluster_labels = fcluster(linkage_matrix,4, criterion='maxclust')
clusters = {i: [] for i in range(1, 5)}                             # 4 clusters

for i, label in enumerate(cluster_labels):
    clusters[label].append((word_dict[i], similarity_matrix[i]))

# Sort words in each cluster by their average similarity score
for _, words_in_cluster in clusters.items():
    words_in_cluster.sort(key=lambda x: np.mean([score for score in x[1]]), reverse=True)

for _, words_in_cluster in clusters.items():
    if len(words_in_cluster) > 4:
        over_4 = words_in_cluster.pop(0)  # Remove the word with the highest score
        under_4 = {k: v for k, v in clusters.items() if len(v) < 4}
        if under_4:
            best_match = max(
                under_4.items(),
                key=lambda x: np.mean([score for _, score in x[1]])
            )[0]
            clusters[best_match].append(over_4)

# Display the result
for CLUSTER_ID, words_in_cluster in clusters.items():
    words_to_display = [word[0] for word in words_in_cluster]
    print(f"Group {CLUSTER_ID}: {words_to_display}")