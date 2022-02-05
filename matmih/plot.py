"""plot.py: Utility builder class for ML plots.
Uses scikit-learn code samples and framework
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import randomcolor
import math

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from .image import Image


class PlotBuilder:
    def __init__(self):
        self.__figures = []
        self._subplot_idx = None
        self._subplot_size = [1, 1]
        self._current_subplot = None
        self._next_plot = True

    def show(self, close=True):
        plt.show()
        if close:
            self.close()

    def close(self):
        for fig in self.__figures:
            plt.close(fig)
        self.__figures = []
        self._subplot_idx = None
        self._subplot_size = [1, 1]

    def same(self):
        self._next_plot = False

        return self

    @staticmethod
    def get_plot():
        return plt

    def create_subplots(self, rows, cols, fig_size=(18, 18)):
        self.__figures.append(plt.figure(figsize=fig_size))
        self._subplot_idx = 0
        self._subplot_size = [rows, cols]

        return self

    def _get_next_plot(self, **kwargs):
        if not self._next_plot:
            self._next_plot = False
            return self._current_subplot

        if self._subplot_idx is not None and self._subplot_idx >= (self._subplot_size[0] * self._subplot_size[1]):
            self._subplot_idx = None
            self._subplot_size = [1, 1]

        if self._subplot_idx is None:
            self.__figures.append(plt.figure(**kwargs))
            self._current_subplot = self.__figures[-1].add_subplot(1, 1, 1)
            return self._current_subplot

        self._subplot_idx += 1
        self._current_subplot = self.__figures[-1].add_subplot(*self._subplot_size, self._subplot_idx, **kwargs)
        return self._current_subplot

    def create_plot(self, title, x_data, *args):
        """
        Plot a series of graphs on X axis points given by x_data
        and Y axis by tuples of (y_values, y_title) in args
        """
        sp = self._get_next_plot()

        rand_color = randomcolor.RandomColor()
        limits = [[100000, -100000], [100000, -100000], [100000, -100000]]

        x_values, x_label = x_data
        limits[0][0] = min(limits[0][0], min(x_values))
        limits[0][1] = max(limits[0][1], max(x_values))

        for data in args:
            color = rand_color.generate()[0]
            if isinstance(data, list):
                i = 0
                for y_values in data:
                    i += 1
                    limits[1][0] = min(limits[1][0], min(y_values))
                    limits[1][1] = max(limits[1][1], max(y_values))
                    sp.plot(x_values, y_values, color=color, linewidth=2,
                            linestyle='--' if i % 2 else '-')
            else:
                y_values, y_title = data
                limits[1][0] = min(limits[1][0], min(y_values))
                limits[1][1] = max(limits[1][1], max(y_values))
                sp.plot(x_values, y_values, label=y_title, color=color, linewidth=2, linestyle='-')

        sp.set_xlim(limits[0])
        sp.set_xlabel(x_label)
        sp.set_ylim(limits[1])
        sp.legend(loc='upper right')
        sp.set_title(title)

        return self

    def create_horizontal_line(self, *args, **kwargs):
        sp = self._get_next_plot()

        rand_color = randomcolor.RandomColor()
        y_limits = list(sp.get_ylim())
        for data in args:
            y_value, y_title = data
            y_limits[0] = min(y_limits[0], y_value)
            y_limits[1] = max(y_limits[1], y_value)
            sp.plot(sp.get_xlim(), [y_value, y_value], label=y_title, color=rand_color.generate()[0], **kwargs)
        sp.set_ylim([y_limits[0] * 0.9, y_limits[1] * 1.1])
        sp.legend(loc='upper right')

        return self

    def create_scatter_plot(self, title, axis_labels, *args):
        """
        Plot a series of graphs of scatter points given by the
        list of tuples (x, y, data_label)
        """
        markers = ['o', '*', '+', 'P', 'X', 'D']
        is_3d = len(axis_labels) == 3

        sp = self._get_next_plot(projection='3d') if is_3d else self._get_next_plot()

        rand_color = randomcolor.RandomColor()
        limits = [[100000, -100000], [100000, -100000], [100000, -100000]]
        marker_id = 0
        for data in args:
            data = list(data)
            values = data[:-1]
            label = data[-1]
            color = rand_color.generate()[0]
            for i, v in enumerate(values):
                limits[i][0] = min(limits[i][0], min(v))
                limits[i][1] = max(limits[i][1], max(v))

            sp.scatter(*values, label=label, color=color, marker=markers[marker_id % len(markers)])
            marker_id += 1

        if is_3d:
            x_label, y_label, z_label = axis_labels
        else:
            x_label, y_label = axis_labels

        sp.set_xlim(limits[0])
        sp.set_xlabel(x_label)
        sp.set_ylim(limits[1])
        sp.set_ylabel(y_label)
        if is_3d:
            sp.set_zlim(limits[2])
            sp.set_zlabel(z_label)
        sp.legend(loc='upper right')
        sp.set_title(title)

        return self

    def create_histograms(self, pd_categoricals, titles, bins='auto', fig_size=(13, 6)):
        """
        Creates a histogram based on x_data
        """
        fig, ax = plt.subplots(figsize=fig_size)
        self.__figures.append(fig)
        subplot_no = len(pd_categoricals)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        colors = {}
        for i in range(len(pd_categoricals)):
            data = pd_categoricals[i]
            if isinstance(data, pd.core.series.Series):
                data = data[data.isnull() == False].value_counts()
            else:
                data, labels = data
                class_id, data = np.unique(data, return_counts=True)
                sort_freq = np.argsort(-data)
                data = dict(zip([labels[c] for c in class_id[sort_freq]], data[sort_freq]))

            labels = [name for name in data.keys()]
            data = [data[name] for name in labels]
            if len(colors) != len(labels):
                colors = dict(zip(labels, randomcolor.RandomColor().generate(count=len(labels))))

            sp = fig.add_subplot(1, subplot_no, i + 1)
            plt.bar(labels, data, color=[colors[l] for l in labels])
            plt.xticks(labels, rotation=90)
            sp.set_title(titles[i])
            plt.tight_layout()

        return self

    def create_images(self, images, titles, **kwargs):
        """
        Creates a grid of images
        """
        subplot_cols = min(len(images), 6)
        subplot_rows = math.ceil(len(images) / subplot_cols)
        fig_size = kwargs.pop('figsize', 3)
        dpi = kwargs.pop('dpi', 80)

        fig = plt.figure(figsize=(subplot_cols * fig_size, subplot_rows * fig_size), dpi=dpi)
        self.__figures.append(fig)

        if len(images) != len(titles):
            raise(Exception("Image and title list must be the same"))

        for i in range(len(images)):
            sp = fig.add_subplot(subplot_rows, subplot_cols, i + 1)
            sp.set_title(titles[i])
            image = images[i]
            if isinstance(image, str):
                image = Image.load(image)
            plt.imshow(Image.to_image(image), **kwargs)
            sp.axes.get_xaxis().set_visible(False)
            sp.axes.get_yaxis().set_visible(False)
            sp.grid(None)
        plt.tight_layout()

        return self

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def create_confusion_matrix(self, y_true, y_pred, classes, title=None,
                                x_label='Predicted class', y_label='True class', normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        self.__figures.append(fig)

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.array(range(len(classes))))

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...and label them with the respective list entries
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel=y_label,
               xlabel=x_label)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', pad=10)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="green" if i == j else "white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        return self

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    def create_roc_curve_multiclass(self, y_true_labels, y_predicted_scores, classes, plot_mask=False):
        """
        Compute ROC curve and ROC area for each class
        classes contains a list of target label names in the multiclass clasification
        plot_mask can contain a list of True/False values for each of the above class to be predicted
        """

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if not plot_mask:
            plot_mask = classes != None

        for i in range(len(classes)):
            if plot_mask[i]:
                fpr[i], tpr[i], _ = roc_curve(y_true_labels, y_predicted_scores[:, i], pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])

        # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
        # Compute micro-average ROC curve and ROC area for all classes
        y_true_micro = np.array([], dtype=np.int32)
        y_scores_micro = np.array([], dtype=np.float64)
        for i in range(len(classes)):
            if plot_mask[i]:
                y_true_micro = np.append(y_true_micro, y_true_labels == i)
                y_scores_micro = np.append(y_scores_micro, y_predicted_scores[:, i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_micro, y_scores_micro)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        self.__plot_roc_curve(fpr, tpr, roc_auc, classes)

        return self

    def __plot_roc_curve(self, fpr, tpr, roc_auc, classes):
        self.__figures.append(plt.figure(figsize=(15, 15)))

        rand_color = randomcolor.RandomColor()
        for i, c in enumerate(classes):
            if i in fpr:
                plt.plot(fpr[i], tpr[i], label='ROC curve [class=%s] (area = %0.2f)' % (c, roc_auc[i]),
                         color=rand_color.generate()[0], linewidth=2)

        if 'micro' in fpr:
            plt.plot(fpr['micro'], tpr['micro'], label='ROC curve Micro Average (area = %0.2f)' % roc_auc['micro'],
                     color='deeppink', linewidth=4, linestyle=':')
        if 'macro' in fpr:
            plt.plot(fpr['macro'], tpr['macro'], label='ROC curve Macro Average (area = %0.2f)' % roc_auc['macro'],
                     color='darkorange', linewidth=4, linestyle=':')

        plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (False Positive Rate)')
        plt.ylabel('Precision (True Positive Rate)')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
