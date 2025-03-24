# Prediction Model

A machine learning model built with TensorFlow that predicts  based on key features metrics.

## Overview

This model uses a neural network to estimate residential property values by analyzing:
- Total square footage
- Number of rooms
- Quality of life indicators

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- random

## Installation
```bash
pip install -r requirements.txt
```

## Ressource
```bash
.
├── getcat.py
├── gethouse.py
├── house_generator.py
├── houses_1000.json
├── main.py
└── README.md
```
### Important Class
HOUSE : size | nb_rooms | location_quality | price
```python
class House:
    def __init__(self, location_quality: int, price: float, num_rooms: int,  size: float):
        self.location_quality = location_quality
        self.price = price
        self.num_rooms = num_rooms
        self.size = size
```

CAT : breed ( race ) | size ( length ) | value ( price )
```python
class Cat:
    def __init__(self, race: int, price: float, length: float):
        self.race = race
        self.price = price
        self.length = length
```

## Optional / usefull
### House Generator :
```python
def get_houses() -> list[House]:
    houses: list[House] = []
    
    # Generate 1000 random houses with correlated properties
    for _ in range(1000):
        # Location quality from 1 (best) to 5 (worst)
        location_quality = random.randint(1, 5)
        
        # Base price depends on location quality
        base_price = 500000 - (location_quality - 1) * 75000  # Better locations cost more
        
        # Number of rooms varies by location quality (better locations tend to have more rooms)
        min_rooms = 2 + (5 - location_quality)
        max_rooms = 5 + (5 - location_quality)
        num_rooms = random.randint(min_rooms, max_rooms)
        
        # Size varies by number of rooms
        min_size = num_rooms * 30  # minimum 30 sq meters per room
        max_size = num_rooms * 50  # maximum 50 sq meters per room
        size = round(random.uniform(min_size, max_size), 1)
        
        # Final price affected by location, rooms, and size
        room_bonus = num_rooms * 25000  # Each room adds value
        size_bonus = size * 1000  # Each sq meter adds value
        final_price = round(base_price + room_bonus + size_bonus, 2)
        
        houses.append(House(location_quality, final_price, num_rooms, size))
    
    return houses
```

### CAT Generator :
```python
def get_cats() -> list[Cat]:
    cats: list[Cat] = []

    # Generate 1000 cats with correlated properties
    for _ in range(1000):
        # Race from 1 (highest quality) to 5 (common)
        race = random.randint(1, 5)

        # Base price depends on race (higher price for lower race number)
        base_price = 1000 - (race - 1) * 150  # race 1: 1000, race 2: 850, race 3: 700, etc.

        min_length = 30 + (5 - race) * 2
        max_length = 45 + (5 - race) * 2
        length = round(random.uniform(min_length, max_length), 1)

        # Final price affected by both race and length
        length_bonus = (length - min_length) * 10  # Larger cats are more valuable
        final_price = round(base_price + length_bonus, 2)

        cats.append(Cat(race, final_price, length))

    return cats
```

## WORKSHOP
### Step1
simple model, 1 layer, 1 inputs, 1 output
- Use only the size of the house to predict its price
- Learn about basic neural network structure
- Understand single input/output relationships

### Step2
simple model, 1 layer, 3 inputs, 1 output
- Use size, number of rooms, and location quality to predict house price
- Learn how to handle multiple inputs
- Understand feature normalization

### Step3
custom model, multiple layers, 3 inputs, 1 output
- Create a deeper network with multiple layers
- Experiment with different layer sizes
- Learn about model complexity and overfitting

### Bonus Challenge
create your own prediction model
- Design a custom architecture
- Choose your own features
- Implement data preprocessing
- Evaluate and optimize performance
