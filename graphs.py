import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]

train_accuracy = [77, 82, 85, 87, 90]
test_accuracy = [74, 78, 80, 82, 83.4]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracy,
    marker='o',
    linewidth=2,
    label='Training Accuracy'
)

plt.plot(
    epochs,
    test_accuracy,
    marker='o',
    linestyle='--',
    linewidth=2,
    label='Test Accuracy'
)

plt.title("Training vs Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)
plt.ylim(70, 95)

plt.legend()
plt.grid(True)

plt.savefig("train_test_accuracy.png")

plt.show()