import tensorflow as tf
from gethouse import *
from getcat import *

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),  # Couche d'entrée avec 3 caractéristiques
        tf.keras.layers.Dense(32, activation='relu'),  # Couche cachée
        tf.keras.layers.Dense(1)  # Couche de sortie pour prédire le prix
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def housing_AI(houses: list[House]) -> None:
    # Préparer les données pour l'entraînement
    X = [[house.size, house.num_rooms, house.location_quality] for house in houses]
    y = [house.price for house in houses]

    # Créer et entraîner le modèle
    model = create_model()
    model.fit(X, y, epochs=500, verbose=1)

    # Prédire les prix avec le modèle entraîné
    predictions = model.predict(X)
    for house, predicted_price in zip(houses, predictions):
        print(f"Predicted price for house with size {house.size}, {house.num_rooms} rooms, and location quality {house.location_quality}: {predicted_price[0]:.2f}")

def main(av, ac) -> int:
    if ac != 2:
        print("Usage: main.py <house_file>.json")
        return 84
    houses: list[House] = get_houses(av[1])
    housing_AI(houses)
    return 0

if __name__ == '__main__':
    exit(main(argv, argc))