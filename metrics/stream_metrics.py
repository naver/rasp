import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import distributed


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, n_sup_classes):
        super().__init__()
        self.n_classes = n_classes
        self.n_sup_classes = n_sup_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def to_str(self, results, verbose=True):
        string = "\n"
        ignore = ["Class IoU", "Class Acc", "Class Prec", "Agg",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text", 
                  "Summary"]

        for k, v in results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        if verbose:
            string += 'Class IoU:\n'
            for k, v in results['Class IoU'].items():
                string += "\tclass %d: %s\n"%(k, str(v))

            for i, name in enumerate(['Class IoU', 'Class Acc', 'Class Prec']):
                string += f"{name}:'\t: {results['Agg'][i]}\n"
            
            i = 0
            string += 'Summary Metrics:\n'
            for name in ('Class IoU', 'Class Acc', 'Class Prec'):
                string += f"{name}\n"
                string += f"\t1-{self.n_sup_classes - 1}: {results['Summary'][i]}\n"
                string += f"\t{self.n_sup_classes}-{self.n_classes-1}: {results['Summary'][i+1]}\n"
                i += 2

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()                               # overall pixel accuracy. including the bg class (index = 0)
        acc_cls_c = diag / (gt_sum + EPS)                           # class-wise mean accuracy.
        acc_cls = np.mean(acc_cls_c[mask])                          # mean class-wise accuracy excluding the bg class (index = 0)
        precision_cls_c = diag / (hist.sum(axis=0) + EPS)           
        precision_cls = np.mean(precision_cls_c)
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)        # class-wise IoU 
        mean_iu = np.mean(iu[mask])                                 # mean IoU excluding the background class
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        cls_prec = dict(zip(range(self.n_classes), [precision_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        short_metrics = []
        for metric in (cls_iu, cls_acc, cls_prec):
            base_classes = 0.
            novel_classes = 0.
            for k, v in metric.items():
                if v != "X":
                    if k < self.n_sup_classes and k > 0: # exclude the bkg class
                        base_classes += v
                    
                    if k >=self.n_sup_classes:
                        novel_classes += v
            
            base = base_classes / (self.n_sup_classes - 1)
            if (self.n_classes - self.n_sup_classes) != 0:
                novel = novel_classes / (self.n_classes - self.n_sup_classes)
            else:
                novel = 0.
            
            short_metrics += [base, novel]

        return {
                "Total samples":  self.total_samples,
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean Prec": precision_cls,
                # "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Class Acc": cls_acc,
                "Class Prec": cls_prec,
                "Agg": [mean_iu, acc_cls, precision_cls],
                "Confusion Matrix": self.confusion_matrix_to_fig(),
                'Summary': short_metrics
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1)+0.000001)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig

