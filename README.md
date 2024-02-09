# OptimisationD-terministe
Directions de descente

import numpy as np


x0=[-2,-7]
epsilon = 10**-4
A = np.array([[10 ,0],#[2*x1**2, x1x2]
             [0,1]])#[x2x1,2x2**2]
b = np.array([[-3],#x1
             [-3]])#x2
tau = 0.7
omega = 10**-4

def f(X):
    y = (5*X[0]**2 + 0.5*(X[1]**2) - 3*(X[0]+X[1]))
    return y
    
def gradientf(X):
    f = np.array([10*X[0] - 3, X[1]-3])
    
    #Ecrire la dérivé de la fonction sous forme matricielle
    return f

def Hessiennef(X):
    y = np.array([[10, 0],
                  [0, 1]])
    #Ecrire la dérivé seconde de la fonction sous forme matricielle
    return y


def PasArmijo(x, d, omega, tau, rho0):
    rho = rho0
    grad = gradientf(x)
    while np.all(f(x + rho * d) > f(x) + omega * rho * np.dot(grad, d)):
        rho = tau * rho
    return rho


def PasCauchyQuadra(A, b, x, d):
    rho = -np.dot((np.dot(A, x) + b), d) / np.dot(np.dot(d, A), d)
    return rho

#Première itération

d0= -gradientf(x0)
rho0=PasCauchyQuadra(A, b, x0, d0)
x1 = x0 + rho0 * d0

#Deuxième itération

d1 = -gradientf(x1)
rho1 = PasCauchyQuadra(A, b, x1, d1)
x2 = x1 + rho1 * d1

#Troisème itération

d2 = -gradientf(x2)
rho2 = PasCauchyQuadra(A, b, x2, d2)
x3 = x2 + rho2 * d2

#Quatrième itération

d3 = -gradientf(x3)
rho3 = PasCauchyQuadra(A, b, x3, d3)
x4 = x3 + rho3 * d3


print("les différentes directions :")
print("0 : ")
print(d0)
print("1 : ")
print(d1)  # Assurez-vous que d1 est correctement défini
print("2 : ")
print(d2)  # Assurez-vous que d2 est correctement défini
print("3 : ")
print(d3)  # Assurez-vous que d3 est correctement défini
print("-------------")
print("Les différents pas :")
print("0 : ")
print(rho0)
print("1 : ")
print(rho1)  # Assurez-vous que rho1 est correctement défini
print("2 : ")
print(rho2)  # Assurez-vous que rho2 est correctement défini
print("3 : ")
print(rho3)  # Assurez-vous que rho3 est correctement défini



rho0 = PasArmijo(x0, d0, omega, tau, rho0)
rho1 = PasArmijo(x1, d1, omega, tau, rho0)
rho2 = PasArmijo(x2, d2, omega, tau, rho1)
rho3 = PasArmijo(x3, d3, omega, tau, rho2)


#Méthode du gradient à pas fice égal à 1 - Méthode de Newton


def gradientPasFixe1(x0, epsilon):
    x = x0
    k = 0
    while np.dot(np.dot(gradientf(x), Hessiennef(x)), gradientf(x)) >= epsilon:
        d = -np.dot(np.linalg.inv(Hessiennef(x)), gradientf(x))
        x = x + d
        k = k + 1
        
    print("x =", x)
    print("-----------")
    print("Iterations =", k)
    print("-----------")
    print("le résiduel =", np.linalg.norm(gradientf(x)))


#Fonction gradient cauchy pour les fonctions quadratiques



def gradientMethodeCauchy(x0, A, b, epsilon, itermax):
    x = x0
    d = -(np.dot(A, x) + b)
    k = 0
    
    while (np.linalg.norm(np.dot(A, x) + b) > epsilon) and (k < itermax):
        rho = -np.dot(np.dot(np.transpose(np.dot(A, x) + b), d), 1.0 / np.dot(np.dot(d, A), d))
        x = x + rho * d
        d = -(np.dot(A, x) + b)
        k = k + 1
    
    print("x =", x)
    print("-----------")
    print("Iterations =", k)
    print("-----------")
    print("le résiduel =", np.linalg.norm(gradientf(x)))


#Méthode gradient pas Armijo



def gradientMethodeArmijo(x0, rho0, omega, tau, epsilon, itermax):
    x = x0
    grad = gradientf(x)
    k = 0
    
    while (np.linalg.norm(grad) > epsilon) and (k < itermax):
        d = -grad
        rho = PasArmijo(x, d, omega, tau, rho0)
        x = x + rho * d
        grad = gradientf(x)
        k = k + 1
    
    print("x =", x)
    print("-----------")
    print("Iterations =", k)
    print("-----------")
    print("le résiduel =", np.linalg.norm(gradientf(x)))


#Méthode gradient Méthode conjuguée sur le modèle de Cauchy


def gradientMethodeConjugue(x0, A, b, epsilon, itermax):
    x = x0
    grad = gradientf(x)
    d = -grad
    k = 0
    
    while (np.linalg.norm(np.dot(A, x) + b) > epsilon) and (k < itermax):
        rho = -np.dot(np.dot(np.transpose(np.dot(A, x) + b), d), 1.0 / np.dot(np.dot(d, A), d))
        nextx = x + rho * d
        grad1 = gradientf(nextx)
        beta1 = np.dot(np.transpose(grad1), grad1) / np.dot(np.transpose(grad), grad)
        d = -(np.dot(A, nextx) + b) + beta1 * d
        x = nextx
        k = k + 1
    
    print("x =", x)
    print("-----------")
    print("Iterations =", k)
    print("-----------")
    print("le résiduel =", np.linalg.norm(gradientf(x)))



#Méthode d'un pas d'Armijo dans le cas de la variante Flechter-Reeves avec relance


def pasArmijoFletcherReevesRelance(x0, epsilon, max_iter, omega, tau):
    x = x0
    grad = gradientf(x)
    d = -grad
    k = 0
    while k < max_iter:
        rho = 1.0
        f_x = f(x)
        while f(x + rho * d) > f_x + omega * rho * np.dot(grad, d):
            rho *= tau  # Réduire le pas à chaque itération jusqu'à ce que la condition Armijo soit satisfaite
        
        x_next = x + rho * d
        grad_next = gradientf(x_next)
        beta = np.dot(grad_next, grad_next) / np.dot(grad, grad)
        d = -grad_next + beta * d
        x = x_next
        grad = grad_next
        
        if np.linalg.norm(grad) < epsilon:
            break
        
        k += 1
    
    return x, k

    
#Un pas d’Armijo dans le cas de la variante Polak Ribière avec relance.


def pasArmijoPolakRibiereRelance(x0, epsilon, max_iter, omega, tau):
    x = x0
    grad = gradientf(x)
    d = -grad
    k = 0
    while k < max_iter:
        rho = 1.0
        f_x = f(x)
        while f(x + rho * d) > f_x + omega * rho * np.dot(grad, d):
            rho *= tau  # Réduire le pas à chaque itération jusqu'à ce que la condition Armijo soit satisfaite
        
        x_next = x + rho * d
        grad_next = gradientf(x_next)
        
        # Calculer la valeur de beta en utilisant la formule de Polak-Ribière
        num = np.dot(grad_next, grad_next - grad)
        den = np.dot(grad, grad)
        beta = max(0, num / den)  # Assurez-vous que beta soit positif
        
        d = -grad_next + beta * d
        x = x_next
        grad = grad_next
        
        if np.linalg.norm(grad) < epsilon:
            break
        
        k += 1
    
    return x, k



def newton_method(f, df, d2f, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        gradient = df(x)
        hessian = d2f(x)
        step = np.linalg.solve(hessian, -gradient)
        x += step
        if np.linalg.norm(gradient) < tol:
            break
    return x




def bfgs_method(f, df, x0, tol=1e-6, max_iter=100):
    n = len(x0)
    H = np.identity(n)
    x = x0
    for i in range(max_iter):
        gradient = df(x)
        step = -np.dot(H, gradient)
        x_new = x + step
        s = x_new - x
        y = df(x_new) - gradient
        rho = 1.0 / np.dot(y, s)
        H = np.dot(np.dot((np.identity(n) - rho * np.outer(s, y)), H),
                   (np.identity(n) - rho * np.outer(y, s))) + rho * np.outer(s, s)
        x = x_new
        if np.linalg.norm(gradient) < tol:
            break
    return x
