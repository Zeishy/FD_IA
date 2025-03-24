import tensorflow as tf
from getcat import *

def create_cat_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def cat_AI(cats: list[Cat]) -> None:
    X = [[cat.race, cat.length, cat.price] for cat in cats]
    y = [cat.price for cat in cats]

    model = create_cat_model()
    model.fit(X, y, epochs=100, verbose=1)

    predictions = model.predict(X)
    for cat, predicted_price in zip(cats, predictions):
        print(f"Cat: {cat.race}, Predicted Price: {predicted_price[0]:.2f}")

cats = get_cats()
cat_AI(cats)