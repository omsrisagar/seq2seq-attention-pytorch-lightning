import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# naive_bayes_csv = 'NaiveBayes_050505_100.csv'
# seq2seq_csv = 'test-ar-training-data_050505_100/pred_probs.csv'
max_entropy_csv = '../old_data/NaiveBayes_30_100.csv'
naive_bayes_csv = '../old_data/MaxEntropy_30_100.csv'
seq2seq_csv = 'results-ar-training-data_30_100/pred_probs.csv'
labels = ['MaxEntropy', 'NaiveBayes', 'Seq2Seq']
all_csvs = [max_entropy_csv, naive_bayes_csv, seq2seq_csv]
num_csvs = len(all_csvs)

y_true = [[] for _ in range(num_csvs)]
y_pred = [[] for _ in range(num_csvs)]
tpr = [[] for _ in range(num_csvs)]
fpr = [[] for _ in range(num_csvs)]

# Create the figure and axes
fig, ax = plt.subplots()

for i in range(num_csvs):
    with open(all_csvs[i]) as file:
        reader = csv.reader(file)
        for row in reader:
            y_pred[i].append(float(row[0]))
            y_true[i].append(float(row[1]))
    fpr[i], tpr[i], _ = roc_curve(y_true[i], y_pred[i])
    ax.plot(fpr[i], tpr[i], label=labels[i])

# Add labels and title
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve for different algorithms")

# Add legend
ax.legend()

# Show the plot
plt.show()

