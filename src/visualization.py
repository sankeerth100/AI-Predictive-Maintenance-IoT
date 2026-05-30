import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_failure_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Machine failure", data=df)
    plt.title("Failure vs No Failure Distribution")
    plt.xlabel("Machine Failure")
    plt.ylabel("Count")
    plt.savefig("outputs/failure_distribution.png")
    plt.close()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()