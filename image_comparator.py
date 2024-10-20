from sklearn.metrics.pairwise import cosine_similarity

class ImageComparator:
    def __init__(self, product_features):
        self.product_features = product_features  # Dictionary of product IDs and their features

    def find_best_match(self, feature_vector):
        similarities = {}
        for product_id, features in self.product_features.items():
            similarity = cosine_similarity(feature_vector, features)
            similarities[product_id] = similarity

        # Return the product ID with the highest similarity
        best_match = max(similarities, key=similarities.get)
        return best_match
