# Kessler Game Controller (2nd Place Winner)

This project implements a fuzzy logic controller for the Kessler Game, a space-themed game where a ship must navigate and destroy asteroids.

## Files

- `bestcontroller.py`: Contains the main controller logic using fuzzy logic systems.
- `graphics_both.py`: Handles graphical representations (details to be added).
- `scenario_test.py`: Provides test scenarios for the controller (details to be added).
- `test_controller.py`: Contains unit tests for the controller (details to be added).

## Controller Overview

The `BestController` class in `bestcontroller.py` is the core of this project. It implements a fuzzy logic system to control the ship's actions in the Kessler Game.

### Key Features

1. **Targeting System**: Uses a fuzzy logic control system to determine ship turning and firing actions based on:
   - Bullet travel time
   - Angular difference to target

2. **Collision Avoidance**: Implements a second fuzzy system to avoid collisions based on:
   - Distance to nearest asteroid
   - Relative speed

3. **Intercept Calculation**: Calculates the best intercept point for shooting asteroids, taking into account their movement.

## Dependencies

- `kesslergame`: The main game library
- `skfuzzy`: For implementing fuzzy logic systems
- `numpy`: For numerical computations
- `matplotlib`: For any plotting needs (if used in `graphics_both.py`)

## Usage

(Add instructions on how to run the controller with the Kessler Game)

## Testing

Use `scenario_test.py` to run specific game scenarios and `test_controller.py` for unit testing the controller's functions.