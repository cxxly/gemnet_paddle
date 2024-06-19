import logging

import numpy as np
import paddle

from ..utils import read_value_json
from ..utils import update_json


class AutomaticFit:
    """
    All added variables are processed in the order of creation.
    """

    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file, name):
        self.variable = variable
        self.scale_file = scale_file
        self._name = name
        self._fitted = False
        self.load_maybe()
        if AutomaticFit.fitting_mode and not self._fitted:
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []
            else:
                self._add2queue()

    def reset():
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    def fitting_completed():
        return AutomaticFit.queue is None

    def set2fitmode():
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        logging.debug(f"Add {self._name} to queue.")
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(
                    f"Variable with the same name ({self._name}) was already added to queue!"
                )
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """
        Set the next variable in the queue that should be fitted.
        """
        queue = AutomaticFit.queue
        if len(queue) == 0:
            logging.debug("Processed all variables.")
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """
        Load variable from file or set to initial value of the variable.
        """
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            logging.info(
                f"Initialize variable {self._name}' to {self.variable.numpy():.3f}"
            )
        else:
            self._fitted = True
            logging.debug(f"Set scale factor {self._name} : {value}")
            with paddle.no_grad():
                paddle.assign(paddle.to_tensor(data=value), output=self.variable)


class AutoScaleFit(AutomaticFit):
    """
    Class to automatically fit the scaling factors depending on the observed variances.

    Parameters
    ----------
        variable: tf.Variable
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file, name):
        super().__init__(variable, scale_file, name)
        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    def observe(self, x, y):
        """
        Observe variances for inut x and output y.
        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return
        if AutomaticFit.activeVar == self:
            nSamples = tuple(y.shape)[0]
            self.variance_in += paddle.mean(x=paddle.var(x=x, axis=0)) * nSamples
            self.variance_out += paddle.mean(x=paddle.var(x=y, axis=0)) * nSamples
            self.nSamples += nSamples

    def fit(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(
                    f"Did not track the variable {self._name}. Add observe calls to track the variance before and after."
                )
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples
            ratio = self.variance_out / self.variance_in
            value = np.sqrt(1 / ratio, dtype="float32")
            logging.info(
                f"Variable: {self._name}, Var_in: {self.variance_in.numpy():.3f}, Var_out: {self.variance_out.numpy():.3f}, "
                + f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
            )
            with paddle.no_grad():
                paddle.assign(self.variable * value, output=self.variable)
            update_json(self.scale_file, {self._name: float(self.variable.numpy())})
            self.set_next_active()


class ScalingFactor(paddle.nn.Layer):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.

    Parameters
    ----------
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
        name: str
            Name of the scaling factor
    """

    def __init__(self, scale_file, name, device=None):
        super().__init__()
        out_1 = paddle.create_parameter(
            shape=paddle.to_tensor(data=1.0, place=device).shape,
            dtype=paddle.to_tensor(data=1.0, place=device).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(data=1.0, place=device)
            ),
        )
        out_1.stop_gradient = not False
        self.scale_factor = out_1
        self.autofit = AutoScaleFit(self.scale_factor, scale_file, name)

    def forward(self, x_ref, y):
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)
        return y
