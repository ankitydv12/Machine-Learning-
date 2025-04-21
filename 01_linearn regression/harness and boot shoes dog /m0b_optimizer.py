import numpy as np

class MyOptimizer:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, model, inputs, targets):
        '''
        Trains the model using gradient descent

        model: An object with slope and intercept
        inputs: X values (e.g., dates)
        targets: y values (temperatures)
        '''

        for epoch in range(self.epochs):
            # Predict values with current model
            predictions = model.predict(inputs)

            # Calculate errors
            errors = predictions - targets

            # Compute gradients (derivatives)
            d_slope = np.mean(errors * inputs) * 2
            d_intercept = np.mean(errors) * 2

            # Update parameters
            model.slope -= self.learning_rate * d_slope
            model.intercept -= self.learning_rate * d_intercept

            # Optional: print cost every 100 epochs
            if epoch % 100 == 0:
                cost = np.sum(errors ** 2)
                print(f"Epoch {epoch}: Cost = {cost:.4f}, Slope = {model.slope:.4f}, Intercept = {model.intercept:.4f}")
