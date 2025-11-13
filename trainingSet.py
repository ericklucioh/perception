import numpy as np

class TrainingSet:
    def __init__(self, name, X, y, X2=None, y2=None):
        self.name = name
        self.X = np.array(X)
        self.y = np.array(y)
        self.X2 = np.array(X2) if X2 is not None else None
        self.y2 = np.array(y2) if y2 is not None else None

    def print_data(self):
        print(f"\n📊 Conjunto de treinamento: {self.name}")
        for inputs, target in zip(self.X, self.y):
            print(f"  🧠 Treino → Entrada: {inputs} → Saída esperada: {target}")

        if self.X2 is not None and self.y2 is not None:
            print("\n🎯 Conjunto de teste:")
            for inputs, target in zip(self.X2, self.y2):
                print(f"  🔍 Teste → Entrada: {inputs} → Saída esperada: {target}")
