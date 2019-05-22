import mxnet as mx
import numpy as np

def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class HingeLoss(mx.metric.EvalMetric):
    """Computes Hinge loss for SVM.
    The hinge loss for one example:
    .. math::
        \\L_i={\sum_{j\neq y_i}max(0, w_j^Tx_i-w_{y_i}^Tx_i+margin)}
    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    use_linear : boolean, Optional
        Whether to use L1-regularized objective (the default is False).
    margin : float, Optional
        Margin for the SVM (the default is 1.0).
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    --------
    """
    def __init__(self, name='hinge-loss', use_linear=False, margin=1.0,
                 output_names=None, label_names=None):
        super(HingeLoss, self).__init__(
            name, output_names=output_names, label_names=label_names)
        self.use_linear = use_linear
        self.margin = margin

    def norm(self, x):
        return x if self.use_linear else x**2.0

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            n = label.shape[0]
            pred = pred.asnumpy()
            label = label.asnumpy().astype('int32')

            pred = pred - pred[np.arange(n), label].reshape(-1, 1) + self.margin
            pred[np.arange(n), label] = 0

            loss = np.maximum(0, pred)

            self.sum_metric += np.sum(self.norm(loss))
            self.num_inst += n
