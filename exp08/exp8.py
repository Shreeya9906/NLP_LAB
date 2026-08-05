import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

reviews = []
n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input("Enter review: "))

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\nTopics:")
for i, topic in enumerate(lda.components_):
    print("\nTopic", i + 1)
    top_indices = topic.argsort()[-5:]
    for j in top_indices:
        print(words[j])

    # Bar Chart for Topic
    top_words = [words[j] for j in top_indices]
    top_weights = topic[top_indices]

    plt.figure(figsize=(6,4))
    plt.bar(top_words, top_weights)
    plt.title(f"Top Words in Topic {i+1}")
    plt.xlabel("Words")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# t-SNE Visualization
X_dense = X.toarray()

tsne = TSNE(n_components=2, random_state=42, perplexity=2)
X_tsne = tsne.fit_transform(X_dense)

print("\nt-SNE Coordinates:")
for i, point in enumerate(X_tsne):
    print("Review", i + 1, ":", point)

plt.figure(figsize=(6,5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i in range(len(reviews)):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], "R" + str(i + 1))

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
