"""
Guo et al.: O Calibration of Modern Neural Networks, 2017, ICML
https://arxiv.org/abs/1706.04599
Code based on implementation of G. Pleiss: https://gist.github.com/gpleiss/0b17bc4bd118b49050056cfcd5446c71
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pathlib

class CalibrationMeter(object):
    def __init__(self,
                 device,
                 n_bins: int = 10,
                 num_images: int = 500,
                 num_classes: int = 19):

        # Initiate bins
        self.device = device
        self.num_classes = num_classes
        self.num_images = num_images
        self.num_bins = n_bins
        self.width = 1.0 / n_bins
        self.bins = torch.linspace(0, 1, n_bins + 1, device=self.device)
        self.bin_centers = np.linspace(0, 1.0 - self.width, n_bins) + self.width / 2
        self.bin_uppers = self.bins[1:]
        self.bin_lowers = self.bins[:-1]

        # Save bins per class
        self.scores_per_class = torch.zeros(size=(self.num_classes, self.num_bins), device=self.device)
        self.corrects_per_class = torch.zeros_like(self.scores_per_class, device=self.device)
        self.ece_per_class = torch.zeros(size=(self.num_classes, 1), device=self.device)
        self.class_pixels_total = torch.zeros(size=(self.num_classes, 1), device=self.device)

        # Save accuracy and confidence values per class per batch
        self.class_acc_per_batch = [torch.zeros(0, device=self.device) for _ in range(self.num_classes)]
        self.class_conf_per_batch = [torch.zeros(0, device=self.device) for _ in range(self.num_classes)]

        # For whole dataset
        self.overall_corrects = torch.from_numpy(np.zeros_like(self.bin_centers)).to(device)
        self.overall_scores = torch.from_numpy(np.zeros_like(self.bin_centers)).to(device)
        self.overall_ece = 0

    def calculate_bins(self,
                       output: torch.Tensor,
                       label: torch.Tensor,
                       mcd: bool = False):
        """
        Calculate accuracy and confidence values per class and per image. Then, partition confidences into bins.
        This results into accuracy/confidence bins for each class per image.
        """

        # Get rid of batch dimension
        label = label.squeeze(0)

        if mcd:
            softmaxes = output
        else:
            # Logits to predictions
            softmaxes = torch.nn.functional.softmax(output, dim=1)

        for cls in range(self.num_classes):
            # Filter predictions
            confidences, predictions = softmaxes.max(dim=1)
            predictions[predictions != cls] = 255

            # Compute accuracies
            class_accuracy = torch.eq(predictions[label == cls], label[label == cls])
            class_confidence = confidences[label == cls]
            class_pixels = predictions[label == cls].size()[0]

            # Partition bins
            bin_indices = [class_confidence.ge(bin_lower) * class_confidence.lt(bin_upper) for bin_lower, bin_upper in
                           zip(self.bins[:-1], self.bins[1:])]
            bin_corrects = class_pixels * torch.tensor([torch.mean(class_accuracy[bin_index].float())
                                                        for bin_index in bin_indices], device=self.device)
            bin_scores = class_pixels * torch.tensor([torch.mean(class_confidence[bin_index].float())
                                                      for bin_index in bin_indices], device=self.device)

            # Calculate ECE
            ece = class_pixels * self._calc_ece(class_accuracy, class_confidence,
                                                bin_lowers=self.bin_lowers, bin_uppers=self.bin_uppers)

            # Check nan
            bin_corrects[torch.isnan(bin_corrects) == True] = 0
            bin_scores[torch.isnan(bin_scores) == True] = 0

            self.corrects_per_class[cls] += bin_corrects
            self.scores_per_class[cls] += bin_scores
            self.ece_per_class[cls] += ece
            self.class_pixels_total[cls] += class_pixels

    def calculate_mean_over_dataset(self):
        for cls in range(self.num_classes):
            self.overall_corrects += \
                (self.corrects_per_class[cls] / (self.class_pixels_total[cls].item() + 1e-9)) / self.num_classes
            self.overall_scores += \
                (self.scores_per_class[cls] / (self.class_pixels_total[cls].item() + 1e-9)) / self.num_classes
            self.overall_ece += \
                (self.ece_per_class[cls].item() / (self.class_pixels_total[cls].item() + 1e-9)) / self.num_classes

    def save_data(self,
                  where: str,
                  what: str):
        """
        Save entire calibration meter object instance for later use.
        """
        # Create directory for storing results
        pathlib.Path(where).mkdir(parents=True, exist_ok=True)

        # Save results
        with open(os.path.join(where, what), "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _calc_ece(accuracies, confidence, bin_lowers, bin_uppers):
        # Calculate ECE
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidence.gt(bin_lower.item()) * confidence.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def plot_mean(self):
        """
        Plots reliability diagram meant over all classes.
        Returns:
            Figure
        """
        # Calculate gaps
        gap = self.overall_scores - self.overall_corrects

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 9))
        plt.grid()
        fontsize = 25

        # Create bars
        confs = plt.bar(self.bin_centers, self.overall_corrects, width=self.width, ec='black')
        gaps = plt.bar(self.bin_centers, gap, bottom=self.overall_corrects, color=[1, 0.7, 0.7],
                       alpha=0.5, width=self.width, hatch='//', edgecolor='r')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='xx-large')

        # Clean up
        bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
        plt.text(0.2, 0.75, f"ECE: {np.round_(self.overall_ece, decimals=3)}", ha="center",
                 va="center", size=fontsize-2, weight='bold', bbox=bbox_props)
        plt.title("Reliability Diagram", size=fontsize + 2)
        plt.ylabel("Accuracy", size=fontsize)
        plt.xlabel("Confidence", size=fontsize)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)

        return fig

    def plot_cls_diagrams(self):
        """
        Plots for each class a reliability diagram.
        Returns:
            List of Figures
        """
        list_figures = []

        for cls in range(self.num_classes):
            bin_corrects = self.corrects_per_class[cls].cpu().numpy() / (self.class_pixels_total[cls].cpu().item() + 1e-9)
            bin_scores = self.scores_per_class[cls].cpu().numpy() / (self.class_pixels_total[cls].cpu().item() +1e-9)
            ece = self.ece_per_class[cls].cpu().item() / (self.class_pixels_total[cls].cpu().item() + 1e-9)

            # Calculate gaps
            gap = bin_scores - bin_corrects

            # Create figure
            figure = plt.figure(0, figsize=(8, 8))
            plt.grid()
            # Create bars
            confs = plt.bar(self.bin_centers, bin_corrects, width=self.width, ec='black')
            gaps = plt.bar(self.bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5,
                           width=self.width, hatch='//', edgecolor='r')
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')

            # Clean up
            bbox_props = dict(boxstyle="round", fc="lightgrey", ec="brown", lw=2)
            plt.text(0.2, 0.85, f"ECE: {np.round_(ece, decimals=3)}", ha="center", va="center", size=20,
                     weight='bold',
                     bbox=bbox_props)

            plt.title("Reliability Diagram", size=20)
            plt.ylabel("Accuracy", size=18)
            plt.xlabel("Confidence", size=18)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            list_figures.append(figure)

            # Clear current figure
            plt.close(figure)
        return list_figures
