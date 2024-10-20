import pandas as pd

class ProductDatabase:
    def __init__(self, database_path='data/products.csv'):
        self.database = pd.read_csv(database_path)
        self.product_features = self.load_product_features()

    def load_product_features(self):
        product_features = {}
        for index, row in self.database.iterrows():
            product_id = row['product_id']
            # Assuming features are stored in a numpy array file
            features = np.load(f'features/{product_id}.npy')  
            product_features[product_id] = features
        return product_features

    def get_product_info(self, product_id):
        return self.database[self.database['product_id'] == product_id].iloc[0]
