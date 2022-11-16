from __future__ import annotations

import pickle

import numpy as np
import scipy as sp
from lightgbm import LGBMRegressor, early_stopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler


def get_model(method, config):
    """Initialize and return the underlying Behavior Model variant with the provided configuration parameters.

    Parameters
    ----------
    method : str
        Regression model variant.
    config : dict[str, Any]
        Configuration parameters for the model.

    Returns
    -------
    model : Any
       A regression model.

    Raises
    ------
    ValueError
        If the requested method is not supported.
    """

    regressor = None

    # Tree-based Models.
    if method == "rf":
        regressor = RandomForestRegressor(
            n_estimators=config["rf"]["n_estimators"],
            criterion=config["rf"]["criterion"],
            max_depth=config["rf"]["max_depth"],
            random_state=config["random_state"],
            n_jobs=config["num_jobs"],
        )
    elif "gbm" in method:
        splits = method.split("_")
        alpha = None
        if len(splits) == 2:
            objective = splits[-1]
        elif len(splits) == 3:
            objective = splits[-2]
            assert objective == "quantile"
            alpha = float(splits[-1]) / 100.0
        elif len(splits) == 4:
            objective = splits[-3]
            assert objective == "quantile"
            alpha = (float(splits[-2]) + float(splits[-1]) / 10.0) / 100.0

        regressor = LGBMRegressor(
            max_depth=config["gbm"]["max_depth"],
            num_leaves=config["gbm"]["num_leaves"],
            n_estimators=config["gbm"]["n_estimators"],
            min_child_samples=config["gbm"]["min_child_samples"],
            objective=objective,
            random_state=config["random_state"],
            alpha=alpha,
        )
        regressor = MultiOutputRegressor(regressor)
    elif "mlp" in method:
        layers = int(method.split("_")[-1])
        neurons = int(method.split("_")[-2])
        assert layers > 0
        assert neurons > 0
        # Multi-layer Perceptron.
        hls = tuple(neurons for i in range(layers))
        regressor = MLPRegressor(
            hidden_layer_sizes=hls,
            early_stopping=config["mlp"]["early_stopping"],
            max_iter=config["mlp"]["max_iter"],
            alpha=config["mlp"]["alpha"],
            random_state=config["random_state"],
        )
    # Generalized Linear Models.
    elif method == "lr":
        regressor = LinearRegression(n_jobs=config["num_jobs"])
    elif method == "huber":
        regressor = HuberRegressor(max_iter=config["huber"]["max_iter"])
        regressor = MultiOutputRegressor(regressor)
    elif method == "elastic":
        regressor = ElasticNet(
            alpha=config["elastic"]["alpha"],
            l1_ratio=config["elastic"]["l1_ratio"],
            random_state=config["random_state"],
        )
        regressor = MultiOutputRegressor(regressor)

    assert regressor is not None
    return regressor


class BehaviorModel:
    def __init__(self, method, ou_name, config, features, targets):
        """Create a Behavior Model for predicting the resource consumption cost of a single PostgreSQL operating-unit.

        Parameters
        ----------
        method : str
            The method to use. Valid methods are defined in modeling/__init__.py.
        ou_name : str
            The name of this operating unit.
        config : dict[str, Any]
            The dictionary of configuration parameters for this model.
        features : list[str]
            Metadata describing input features for this model.
        targets : list[str]
            Targets that the model is being used to predict.
        """
        self.method = method
        self.ou_name = ou_name
        self.model = get_model(method, config)
        self.features = features
        self.normalize = config["normalize"]
        self.log_transform = config["log_transform"]
        self.robust = config["robust"]
        self.eps = 1e-4
        self.xscaler = RobustScaler() if config["robust"] else StandardScaler()
        self.yscaler = RobustScaler() if config["robust"] else StandardScaler()
        self.targets = targets
        self.fitted_round = False

    def support_incremental(self):
        # Can't do incremental if normalizing inputs.
        return "gbm" in self.method and not self.normalize

    def train(self, x, y):
        """Train a model using the input features and targets.

        Parameters
        ----------
        x : NDArray[np.float32]
            Input features.
        y : NDArray[np.float32]
            Input targets.
        """
        if self.log_transform:
            x = np.log(x + self.eps)
            y = np.log(y + self.eps)
            assert np.sum(x.isna()) == 0
            assert np.sum(y.isna()) == 0

        if self.normalize:
            # TODO(wz2): This partial_fitting technique for scalers is a little broken.
            x = self.xscaler.fit_transform(x)
            y = self.yscaler.fit_transform(y)

        if self.support_incremental() and self.fitted_round:
            # Allow for continuous fit.
            self.model.fit(x, y, init_model=self.model)
        else:
            self.model.fit(x, y)

        self.fitted_round = True


    def predict(self, x):
        """Run inference using the provided input features.

        Parameters
        ----------
        x : NDArray[np.float32]
            Input features.

        Returns
        -------
        NDArray[np.float32]
            Predicted targets.
        """
        # Extract only the relevant features. Good thing is this will complain
        # if a feature is not found.
        x = x[self.features]

        # Transform the features.
        if self.log_transform:
            x = np.log(x + self.eps)
        if self.normalize:
            x = self.xscaler.transform(x)

        # Perform inference (in the transformed feature space).
        y = self.model.predict(x)

        # Map the result back to the original space.
        if self.normalize:
            if len(self.targets) == 1:
                y = y.reshape(-1, 1)
            y = self.yscaler.inverse_transform(y)

        if self.log_transform:
            y = np.exp(y) - self.eps
            y = np.clip(y, 0, None)

        return y

    def save(self, output_path):
        """Save the model to disk.

        Parameters
        ----------
        output_path : Path | str
            The directory to save the model to.
        """
        with open(output_path / f"{self.ou_name}.pkl", "wb") as f:
            pickle.dump(self, f)

        with open(output_path / f"{self.ou_name}_info.txt", "w") as f:
            feature_list = ",".join(self.features)
            target_list = ",".join(self.targets)
            f.write(f"features: {feature_list}\n")
            f.write(f"targets: {target_list}\n")
            f.write(f"normalize: {self.normalize}\n")
            f.write(f"log_transform: {self.log_transform}\n")
            f.write(f"robust: {self.robust}\n")
