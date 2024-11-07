import string
import re
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

class DataVisualizer:

    def plotConfusionMatrix(self, cm, labels, clf_name):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g')
        ax.set(xlabel='Predicted labels', ylabel='Actual labels', title=f'Confusion Matrix of {clf_name} Classifier')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels, rotation=0)
        plt.tight_layout()
        fig.savefig(f'plots/cm_{clf_name.lower().replace(" ", "_")}.png')
        plt.close()

    def plotClassificationReport(self, cr, labels, clf_name):
        cr_mat = []
        allowed_labels = ['negative', 'positive', 'weighted avg']
        lines = cr.split('\n')
        
        for line in lines[2:-1]:
            row = re.split(r'\s{2,}', line.strip())
            if row and row[0] not in allowed_labels:
                continue
            cr_mat.append([float(row[i]) for i in range(-4, 0)])

        xlabels = ['precision', 'recall', 'f1-score', 'support']
        ylabels = labels + ['weighted avg']
        
        fig, ax = plt.subplots()
        sns.heatmap(cr_mat, annot=True, ax=ax, fmt='g')
        ax.set(xlabel='Metrics', ylabel='Classes', title=f'Classification Report of {clf_name} Classifier')
        ax.xaxis.set_ticklabels(xlabels)
        ax.yaxis.set_ticklabels(ylabels, rotation=0)
        plt.tight_layout()
        fig.savefig(f'plots/cr_{clf_name.lower().replace(" ", "_")}.png')
        plt.close()

    def plotClassifierPerformanceComparison(self, metrics_df, clf_names, strategy):
        fig, ax = plt.subplots()
        sns.barplot(x='Metrics', y='value', data=metrics_df, ax=ax, hue='Classifier')
        ax.set(xlabel='Evaluation Metrics', ylabel="Classifier's performance", 
               title=f'Overall Comparison of Classifier\'s Performance ({strategy})')
        ax.legend(bbox_to_anchor=(1, 0.5), loc='best')
        plt.tight_layout()
        fig.savefig(f'plots/classifiers_vs_metrics_{strategy.lower()}.png')
        plt.close()

    def plotClassifiersVsFeatures(self, data, clf_names, colors):
        fig, ax = plt.subplots()
        lines = []
        
        for d, c, clf_name in zip(data, colors, clf_names):
            sns.pointplot(x='x', y='y', data=d, ax=ax, color=c)
            lines.append(mpatches.Patch(color=c, label=clf_name))

        ax.legend(handles=lines, bbox_to_anchor=(1, 0.5), loc='best')
        ax.set(xlabel='K-Best Features', ylabel='Classification Accuracy Scores', 
               title="Comparison of Classifier's Performance over Selected Features")
        fig.savefig('plots/classifiers_vs_features.png')
        plt.close()
