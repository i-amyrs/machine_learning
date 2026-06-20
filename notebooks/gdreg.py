import numpy as np
import matplotlib.pyplot as plt
import os

class GDRegressor:

    def __init__(self,learning_rate=0.01,epochs=1000):
        '''
        Parameters:
        ------------------
        learning rate : float 
        step size for gd
        epochs: int
        number of iterations
        
        '''
        self.lr=learning_rate
        self.epochs=epochs
        self.b=0.0 #intercept / bias
        self.m=0.0 #slope / weight
        self.cost_history=[]

    def fit(self,X,y):
        """
        train the model using the gradient descent
         -----------
        X : numpy.ndarray
            Feature values (shape: m x 1)
        y : numpy.ndarray
            Target values (shape: m x 1)"""
        n=len(y) #no. of samples
        #if reshape needed
        if X.ndim==1:
            X=X.reshape(-1,1)
        if y.ndim ==1:
            y=y.reshape(-1,1)
        
        for i in range(self.epochs):
            ypred=self.m*X + self.b #forward pass

        
             #---------computing the cost--------------
            cost=np.mean((ypred-y)**2)
            self.cost_history.append(cost)
        #--------------
        #b backward pass gradient descent calculation
        # loss with respect to bias (b)
            loss_slope_b=-2 * np.sum(y-self.m*X -self.b)/n

            '''
            -------------------------
            
            gd loss w.r.t. slope (m)
            '''
            loss_slope_m=-2*np.sum((y-self.m*X-self.b)*X)/n
            '''-----
            update parameters------
            '''
            self.b = self.b - self.lr * loss_slope_b
            self.m = self.m - self.lr * loss_slope_m

            #--------progress log------------
            
            if i % 100==0:
                print(f"Epochs{i+1:4d} | cost: {cost:.6f} | m: {self.m:.4f} | b: {self.b:.4f}")
        print(f"\n training completed!")
        print(f"final m:{self.m:.4f}")
        print(f"final b:{self.b:.4f}")

    def predict(self,X):
        # male predictions
        return self.m*X + self.b
    
    def score(self,X,y):
        # calculate r2 score
        y_pred = self.predict(X)
        ss_residual= np.sum((y-y_pred)**2)
        ss_total= np.sum((y-np.mean(y))**2)
        
        r2_score = 1 -(ss_residual / ss_total)
        return r2_score
    
    def plot_cost(self, save_path='outputs/figures/cost_convergence.png',show=True):
        plt.figure(figsize=(10,6))
        plt.plot(self.cost_history, linewidth=2,color='blue')
        plt.xlabel('epochs',fontsize=12)
        plt.ylabel('cost(MSE)',fontsize=12)
        plt.title('cost function convergence', fontsize=14)
        plt.grid(True,alpha=0.3)
        plt.tight_layout()

        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure saved to: {save_path}")
    
    # Show or close
        if show:
            plt.show()
        else:
            plt.close()       


# =============================================
# TEST THE CODE (PUT THIS IN THE SAME FILE)
# =============================================



import numpy as np

# 1. Create data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(-1, 1)

# 2. Train model
model = GDRegressor(learning_rate=0.01, epochs=100)  # More epochs for better convergence
model.fit(X, y)

# 3. Plot and save
model.plot_cost()

# 4. Check if file exists
import os
print("File exists?", os.path.exists('outputs/figures/cost_convergence.png'))
