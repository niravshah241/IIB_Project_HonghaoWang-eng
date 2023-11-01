import numpy as np

class NewtonSolver():
    def __init__(self, tol, max_iter):
        self.tol = tol
        self.max_iter = max_iter

    def solve_1D_equation(self, guess, func, derivFunc, tol = None, max_iter = None):
        if max_iter == None:
            max_iter = self.max_iter
        if tol == None:
            tol = self.tol
        x = guess
        h = func(x) / derivFunc(x)
        iteration = 0
        while abs(h) >= tol and iteration < max_iter:
            iteration += 1
            h = func(x)/derivFunc(x)            
            # x(i+1) = x(i) - f(x) / f'(x)
            x = x - h
        
        print(f"The value of the root is: {x:.3f}")
        return x
    
    def solve_multidim_equation(self, guess, func, jacobian, tol = None, max_iter = None):
        if max_iter == None:
            max_iter = self.max_iter
        if tol == None:
            tol = self.tol

        x = guess
        h = np.linalg.solve(jacobian(x), func(x))
        iteration = 0
        while np.linalg.norm(h) > tol and iteration < max_iter:
            iteration += 1
            h = np.linalg.solve(jacobian(x), func(x))
            x = x - h
        
        print(f"The value of the root is:", x)
        return x

    def solve_1D_PDE(self, func, jacob, L, nx, left_boundary, 
                     right_boundary, tol = None, max_iter = None):
        
        """
            Newton's method for solving a 1D nonlinear PDE.
            -u''(x) = f(x, u)
            u''(x) ≈ (u[i-1] - 2u[i] + u[i+1]) / dx**2

            Parameters:
                func (function): A function that returns an equation to be solved.
                jocabian (function): A function that returns the jacobian of the equations.
                L (float): Length of the domain
                nx (int): Number of spatial grid points
                tol (float): Tolerance for stopping criteria.
                max_iter (int): Maximum number of iterations.
            Returns:
                x (numpy array): The approximate solution to the system of equations.
        """
        if max_iter == None:
            max_iter = self.max_iter
        if tol == None:
            tol = self.tol

        # Discretiaonise the whole length scale
        dx = L / nx
        x = np.linspace(0, L , nx)  

        u = np.linspace(left_boundary, right_boundary, nx)
        u_new = np.copy(u)

        iteration = 0
        h = np.ones(nx)
        while iteration < max_iter and np.max(abs(h)) >= tol:
            iteration += 1
            for i in range(1, nx - 1):
                # Calculate the residual based on the nonlinear PDE
                # Here I utilised the finite difference approximation of second derivative
                residual = u_new[i - 1] - 2 * u_new[i] + u_new[i + 1] + dx**2 * func(x[i], u_new[i])
                jacobian = -2 + dx**2 * jacob(u_new[i])

                # Newton's method: update residuals, jacobians and the results
                h[i] = -residual / jacobian
                u_new[i] = u_new[i] + h[i]

            # make sure Boundary conditions remain after each iteration
            h[0] = left_boundary - u_new[0]
            h[-1] = right_boundary - u_new[-1]
            u_new[0] = left_boundary
            u_new[-1] = right_boundary

        print(f"The result is: {u_new} at iteration {iteration}")
        return u_new
    
def func(x):
    return x * x * x - x * x + 2

def derivFunc(x):
    return 3 * x * x - 2 * x

def func_2d(x):
    f1 = x[0]**2 + x[1]**2 - 3
    f2 = -x[0]**2 + 2*x[1]
    return np.array([f1, f2])

def jacobian(x):
    df1_dx1 = 2 * x[0]
    df1_dx2 = 2 * x[1]
    df2_dx1 = -2*x[0]
    df2_dx2 = 2
    return np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])

def f_pde(x, u):
    return -1 + u**2 + np.sin(2 * np.pi * x)

def jacob_pde(u):
    return 2*u

# Driver program to test above: 1D and 2D
x0 = -20 # Initial values assumed
solver = NewtonSolver(0.001, 100)
solver.solve_1D_equation(x0, func, derivFunc)

x0 = np.array([1.0, 1.0])
solver.solve_multidim_equation(x0, func_2d, jacobian)

# Driver program to test PDE
solver.solve_1D_PDE(f_pde, jacob_pde, L = 1, nx = 20, left_boundary = 0, right_boundary = 1)