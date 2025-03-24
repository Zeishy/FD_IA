import tensorflow as tf
from getcat import *

def create_cat_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),  # Couche d'entrée avec 3 caractéristiques
        tf.keras.layers.Dense(64, activation='relu'),  # Couche cachée
        tf.keras.layers.Dense(32, activation='relu'),  # Couche cachée supplémentaire
        tf.keras.layers.Dense(1)  # Couche de sortie pour prédire le prix
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def cat_AI(cats: list[Cat]) -> None:
    # Préparer les données pour l'entraînement
    X = [[cat.race, cat.length, cat.price] for cat in cats]
    y = [cat.price for cat in cats]

    # Créer et entraîner le modèle
    model = create_cat_model()
    model.fit(X, y, epochs=100, verbose=1)

    # Prédire les prix avec le modèle entraîné
    predictions = model.predict(X)
    for cat, predicted_price in zip(cats, predictions):
        print(f"Cat: {cat.race}, Predicted Price: {predicted_price[0]:.2f}")

cats = get_cats()  # Assurez-vous que cette fonction génère 1000 chats
cat_AI(cats)