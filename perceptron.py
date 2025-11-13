import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        # Inicializa pesos e bias com valores aleatórios
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        print("🚀 Perceptron criado")
        print(f"Pesos iniciais: {self.weights}")
        print(f"Bias inicial: {self.bias}\n")

    def activation(self, x):
        # Função degrau: retorna 1 se x >= 0, senão 0
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Soma ponderada das entradas + bias
        total = np.dot(inputs, self.weights) + self.bias
        return self.activation(total)

    def train(self, X, y, epochs=10):
        print("🎯 Iniciando treinamento...\n")
        for epoch in range(epochs):
            print(f"Época {epoch + 1}/{epochs}")
            total_error = 0

            for inputs, target in zip(X, y):
                # Faz a previsão atual
                prediction = self.predict(inputs)
                error = target - prediction
                total_error += abs(error)

                print(f"Entrada: {inputs}")
                print(f" → Saída esperada: {target}")
                print(f" → Previsão: {prediction}")
                print(f" → Erro: {error}")

                # Atualiza pesos e bias
                self.weights += self.learning_rate * error * np.array(inputs)
                self.bias += self.learning_rate * error

                print(f" → Novos pesos: {self.weights}")
                print(f" → Novo bias: {self.bias}\n")

            print(f"Erro total na época {epoch + 1}: {total_error}\n")
            print("-" * 40)

        print("✅ Treinamento concluído!")
        print(f"Pesos finais: {self.weights}")
        print(f"Bias final: {self.bias}\n")


