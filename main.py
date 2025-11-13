from perceptron import Perceptron
import numpy as np

# Entradas e saídas esperadas (AND)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Cria e treina o Perceptron
p = Perceptron(n_inputs=2, learning_rate=0.1)
p.train(X, y, epochs=10)

# Testa depois do treino
print("\n🔍 Testando o modelo treinado:")
for i in X:
    print(f"{i} -> {p.predict(i)}")
