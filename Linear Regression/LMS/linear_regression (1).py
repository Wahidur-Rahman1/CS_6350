import numpy as np

class LMSRegression:
    def __init__(self):
        self.w = np.array([])

    def train_batch_gradient_descent(self, X, y, r=0.01, norm_w_diff_thresh=1e-6):
        cost_vals = []
        num_examples, num_features = X.shape
        self.w = np.zeros(num_features)
        while True:
            grad_J = np.dot(X.T, -(y-np.dot(X, self.w)))
            new_w = self.w - r * grad_J

            # Update weights
            norm_w_diff = np.linalg.norm((new_w-self.w))
            self.w = new_w

            # Calculate current cost
            cost_vals.append(LMSRegression.cost(X, y, self.w))

            # Return if the convergence critera is met
            if norm_w_diff < norm_w_diff_thresh:
                return cost_vals 

    def train_stochastic_gradient_descent(self, X, y, r=0.01, abs_cost_diff_thresh=1e-6):
        cost_vals = []
        num_examples, num_features = X.shape
        self.w = np.zeros(num_features)
        example_idxs = np.arange(num_examples)
        current_cost = 0
        while True:
            for i in range(num_examples):
                random_example_idx = np.random.choice(example_idxs)
                self.w += r * ((y[random_example_idx] - np.dot(self.w, X[random_example_idx,:])) * X[random_example_idx,:])

                # Calculate and update current cost
                new_cost = LMSRegression.cost(X, y, self.w)
                abs_cost_diff = abs(new_cost - current_cost)
                current_cost = new_cost
                cost_vals.append(current_cost)

                # Break if the convergence critera is met
                if abs_cost_diff < abs_cost_diff_thresh:
                    return cost_vals
    
    def train_analytical(self, X, y):
        self.w = np.dot((np.linalg.inv(X.T @ X) @ X.T), y)

    def predict(self, X):
        return np.dot(X, self.w)

    @classmethod
    def cost(cls, X, y, w):
        J = 0.5 * np.sum((y - np.dot(X, w))**2)
        return J

        
