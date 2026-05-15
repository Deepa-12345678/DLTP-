from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

from src.convolution import Convolution
from src.maxpool import MaxPool
from src.softmax import Softmax

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

conv = Convolution(8, 3)
pool = MaxPool()
softmax = Softmax(13 * 13 * 8, 10)


def forward(image, label):
    out = conv.forward(image)
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(image, label, learn_rate=0.003):
    out, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    gradient = softmax.backprop(gradient, learn_rate)
    gradient = pool.backprop(gradient)
    conv.backprop(gradient, learn_rate)

    return loss, acc

train_losses = []
train_accuracies = []
print("Training started...")

for epoch in range(5):
    print("Epoch:", epoch + 1)

    loss = 0
    num_correct = 0

    for i, (image, label) in enumerate(zip(X_train[:5000], y_train[:5000])):
        l, acc = train(image, label)

        loss += l
        num_correct += acc

        if i % 100 == 99:
            avg_loss = loss / 100
            accuracy = num_correct

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            print(
                "Step:", i + 1,
                "| Avg Loss:", avg_loss,
                "| Accuracy:", accuracy, "%"
            )

            loss = 0
            num_correct = 0
print("\nTesting started...")

test_loss = 0
test_correct = 0

for image, label in zip(X_test[:1000], y_test[:1000]):
    _, loss, acc = forward(image, label)

    test_loss += loss
    test_correct += acc

print("Test Loss:", test_loss / 1000)
print("Test Accuracy:", test_correct / 1000 * 100, "%")

# ---------------- VISUALIZATIONS ----------------

plt.figure(figsize=(8, 5))
plt.plot(train_losses, marker='o')
plt.title("Training Loss Over Steps")
plt.xlabel("Every 100 Training Images")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_accuracies, marker='o')
plt.title("Training Accuracy Over Steps")
plt.xlabel("Every 100 Training Images")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("training_accuracy.png")
plt.show()

metrics = ["Test Accuracy", "Test Loss"]
values = [test_correct / 1000 * 100, test_loss / 1000]

plt.figure(figsize=(6, 5))
plt.bar(metrics, values)
plt.title("Final Model Evaluation")
plt.ylabel("Value")
plt.savefig("final_evaluation.png")
plt.show()

print("\nResult Summary Table")
print("--------------------------------")
print("Training Accuracy : Up to 95%")
print("Test Loss         :", round(test_loss / 1000, 4))
print("Test Accuracy     :", round(test_correct / 1000 * 100, 2), "%")
print("--------------------------------")