import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors.kde import KernelDensity

from deepensemble.combiner import *
from deepensemble.layers.dense import Dense
from deepensemble.layers.recurrent import RecurrentLayer
from deepensemble.metrics import *
from deepensemble.models.ensemblemodel import EnsembleModel
from deepensemble.models.sequential import Sequential
from deepensemble.utils import *
from deepensemble.utils.utils_functions import ITLFunctions

a = 5
b = 2
N = 2000
s_noise = 0.02

x0 = np.linspace(-3.0, 3.0, num=N)
x1 = np.sin(b * x0)
x2 = np.cos(a * x0)

fx = 0.1 * x2 + 0.1 * x1 + 0.4

fx_str = '$f(x)=cos({%g x}) + sin({%g x})$' % (a, b)

nu = np.random.randn(N, ) * s_noise
nu[np.random.rand(N, ) > 0.95] += 0.2

z = fx + nu

n_train = int(N * 0.5)
i_test = N - n_train

w = 1
Y = z[w:]
X = z[:-w]

# X = (X - np.mean(X, axis=0)) / np.var(X, axis=0)

y_train = Y[:n_train][:, np.newaxis]
y_test = Y[n_train:][:, np.newaxis]
X_train = X[:n_train][:, np.newaxis]
X_test = X[n_train:][:, np.newaxis]

fx_train = fx[:n_train][:, np.newaxis]
fx_test = fx[n_train:-w][:, np.newaxis]

SNR = float(np.mean(z) / np.std(z))

print('SNR: %g' % SNR)

fig = plt.figure(figsize=(12, 6), dpi=80)
ax = plt.subplot(211)
t = x0[:-w]
ax.plot(t[:n_train], y_train, 'b.', alpha=0.25, label='Observaciones')
ax.plot(t[:n_train], fx_train, '-r', lw=3, label=fx_str)
plt.title('Muestras Entrenamiento')
plt.xlabel('x')
plt.legend(loc='best', numpoints=3)
plt.tight_layout()

ax = plt.subplot(212)
ax.plot(t[n_train:], y_test, 'b.', alpha=0.25, label='Observaciones')
ax.plot(t[n_train:], fx_test, '-r', lw=3, label=fx_str)
plt.title('Muestras para Pruebas')
plt.xlabel('x')
plt.legend(loc='best', numpoints=3)
plt.tight_layout()

n_neurons = 5
n_models = 3
lr = 0.01
batch_size = 100
max_epoch = 300
valid_size = 0.5
no_update_best_parameters = True
fn_activation1 = ActivationFunctions.tanh
fn_activation2 = ActivationFunctions.sigmoid

# Create Ensemble
ensemble = EnsembleModel(name="Ensemble")

# Create models for ensemble
for i in range(n_models):
    net = Sequential("net%d_ens" % i)  # by default is a regressor
    net.add_layer(RecurrentLayer(n_input=X_train.shape[1], n_recurrent=n_neurons, activation=fn_activation1))
    net.add_layer(Dense(n_output=y_train.shape[1], activation=fn_activation2))
    net.append_cost(cauchy_schwarz_divergence, name='CSD')
    net.set_update(sgd, name='SGD', learning_rate=5 * lr)
    ensemble.append_model(net)

ensemble.set_combiner(GeometricCombiner())
ensemble.compile(fast=True)
ensemble.add_cost_ensemble(fun_cost=cip_redundancy, name="CIP2", beta=0.2)
metrics_ensemble = FactoryMetrics.get_metric(ensemble)

# training
metrics = ensemble.fit(X_train, y_train, max_epoch=max_epoch, batch_size=batch_size,
                       early_stop=False, valid_size=valid_size,
                       no_update_best_parameters=no_update_best_parameters)
print("FINISHED!")

metrics_ensemble.append_metric(metrics)

plt.figure(figsize=(12, 6), dpi=80)

plt.subplot(211)
plt.plot(x0, z, 'k.', alpha=0.15, label='Muestras')
plt.plot(t[:n_train], ensemble.predict(X_train), lw=2, label='Ensamble')
plt.title('Conjunto Entrenamiento - %s' % fx_str)
plt.xlabel('x')
plt.xlim([-3, 3])
plt.ylim([-0.5, 1.1])
plt.legend(loc='best', ncol=5)

plt.subplot(212)
plt.plot(x0, z, 'k.', alpha=0.15, label='Muestras')
plt.plot(t[n_train:], ensemble.predict(X_test), lw=2, label='Ensamble')
plt.title('Conjunto Prueba - %s' % fx_str)
plt.xlabel('x')
plt.xlim([-3, 3])
plt.ylim([-0.5, 1.1])
plt.legend(loc='best', ncol=5)

plt.tight_layout()


def plot_pdf_error(pred, target, label_plot, _ax, n_points=1000, xmin=-3, xmax=3):
    error = pred - target
    _N = len(error)
    s = ITLFunctions.silverman(error, _N, 1).eval()  # Silverman
    kde = KernelDensity(kernel='gaussian', bandwidth=s)
    kde.fit(error)
    x_plot = np.linspace(xmin, xmax, n_points)[:, np.newaxis]
    y_plot = np.exp(kde.score_samples(x_plot))
    _ax.plot(x_plot, y_plot / np.sum(y_plot), label=label_plot)


plt.figure(figsize=(10, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)

plot_pdf_error(ensemble.predict(X_train), y_train, 'Ensamble', ax)

plt.xlabel('Error')
plt.ylabel('PDF del error')
plt.title("Función de Probabilidad (pdf) del Error conjunto Entrenamiento")
plt.legend()

fig = plt.figure(figsize=(10, 5), dpi=80)
ax = fig.add_subplot(1, 1, 1)

plot_pdf_error(ensemble.predict(X_test), y_test, 'Ensamble', ax)

plt.xlabel('Error')
plt.ylabel('PDF del error')
plt.title("Función de Probabilidad (pdf) del Error conjunto Entrenamiento")
plt.legend()

score_test_ensemble = ensemble.score(X_test, y_test)
score_train_ensemble = ensemble.score(X_train, y_train)

print('Score RMS')
print('Ensamble: %f / %f' % (score_train_ensemble, score_test_ensemble))

plt.show()
