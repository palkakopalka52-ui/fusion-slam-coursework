"""Learn the forward wheel residual [m] from a causal window of diagnostics."""
import numpy as np
import keras
from keras import layers
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor


def build_classical(X, y):
    flat = X.reshape(len(X), -1)
    models = {
        'ridge': make_pipeline(StandardScaler(), Ridge(alpha=3.)),
        'poly': make_pipeline(PolynomialFeatures(2, include_bias=False),
                              StandardScaler(), Ridge(alpha=15.)),
        'boost': HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                    l2_regularization=1., random_state=9),
    }
    for m in models.values():
        m.fit(flat, y)
    return models


def build_network(name, window=8, width=24, lr=.003):
    inp = keras.Input((window, 2))
    if name == 'mlp':
        x = layers.Flatten()(inp)
        x = layers.Dense(width, activation='tanh')(x)
    elif name == 'cnn':
        x = layers.Conv1D(16, 3, padding='causal', activation='tanh')(inp)
        x = layers.Flatten()(x)
        x = layers.Dense(12, activation='tanh')(x)
    elif name == 'lstm':
        x = layers.LSTM(16)(inp)
    else:
        raise ValueError('unknown architecture')
    model = keras.Model(inp, layers.Dense(1)(x), name=name)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    return model


def fit_network(name, X, y, Xv, yv, epochs=15, width=24, lr=.003, seed=42):
    keras.utils.set_random_seed(seed)
    model = build_network(name, X.shape[1], width, lr)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                              restore_best_weights=True)
    h = model.fit(X,y,validation_data=(Xv,yv),epochs=epochs,
                  batch_size=128,verbose=0,callbacks=[callback])
    return model, h.history


def predict_bias(model, X, constant=0.):
    """X.shape=(n_windows, window, 2), output=(n_windows,) in metres."""
    X = np.asarray(X,float)
    if X.ndim != 3 or X.shape[-1] != 2 or not np.isfinite(X).all():
        raise ValueError('invalid causal windows')
    if model is None:
        p = np.full(len(X), constant)
    elif isinstance(model, keras.Model):
        p = model.predict(X, verbose=0).ravel()
    else:
        p = model.predict(X.reshape(len(X), -1)).ravel()
    if not np.isfinite(p).all():
        raise ValueError('nonfinite predictor')
    return np.clip(p, -.55, .55)
