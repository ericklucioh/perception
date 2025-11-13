from perceptron import Perceptron
from trainingSet import TrainingSet

# 🧩 Cria vários conjuntos de treinamento

and_training = TrainingSet(
    "AND lógico",
    X=[
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ],
    y=[0, 0, 0, 1],
    X2=[
        [0, 0],
        [1, 1],
        [1, 0]
    ],
    y2=[0, 1, 0]
)

soma10_training = TrainingSet(
    "Soma >= 10",
    X=[
        [3, 7],
        [4, 5],
        [6, 4],
        [8, 2],
        [2, 9],
        [5, 5],
        [7, 1]
    ],
    y=[1, 0, 1, 1, 1, 0, 0],
    X2=[
        [9, 1],
        [2, 8],
        [4, 7],
        [5, 6]
    ],
    y2=[1, 1, 1, 1]
)

inteiros_training = TrainingSet(
    "Número inteiro > 5",
    X=[[x] for x in range(0, 11)],
    y=[1 if x > 5 else 0 for x in range(0, 11)],
    X2=[[12], [4], [7], [1]],
    y2=[1, 0, 1, 0]
)

# Lista com todos os conjuntos
trainings = [and_training, soma10_training, inteiros_training]

# 🚀 Testa todos os conjuntos de treinamento
for t in trainings:
    t.print_data()
    p = Perceptron(n_inputs=len(t.X[0]), learning_rate=0.1)
    p.train(t.X, t.y, epochs=10)

    print(f"\n🔍 Testando modelo para {t.name}:")
    for inputs in t.X:
        print(f"  {inputs} -> {p.predict(inputs)}")

    if t.X2 is not None and t.y2 is not None:
        print(f"\n🎯 Validação para {t.name}:")
        acertos = 0
        for inputs, esperado in zip(t.X2, t.y2):
            pred = p.predict(inputs)
            correto = "✅" if pred == esperado else "❌"
            print(f"  {inputs} → Esperado: {esperado} → Predito: {pred} {correto}")
            acertos += 1 if pred == esperado else 0
        total = len(t.y2)
        print(f"🎓 Acurácia: {acertos}/{total} ({(acertos/total)*100:.1f}%)")

    print("\n" + "-"*60)
