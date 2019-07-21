
# %%
# Download tide information files
from urllib.request import urlopen
import os

filenames = ['h536.dat', 'h178.dat', 'h822.dat']
url = 'ftp://ftp.soest.hawaii.edu/uhslc/fast'
for filename in filenames:
    if not os.path.exists(filename):
        print("Downloading data from '%s/%s', please wait..." % (url, filename))
        opener = urlopen(url)
        open(filename, 'wb').write(opener.read())

import pandas as pd
import numpy as np

# %%
# Load files
def load(filename):
    data = pd.read_csv(filename, skiprows=1, delim_whitespace=True,
                    nrows=10000,
                    names=['pos', 'date', ] + ['%i' % i for i in range(12)])
    # Drop the non numerical lines
    data = data[np.logical_not(data['0'].str.startswith('LAT'))]
    for c in data.columns[2:]:
        data[c] = data[c].astype(np.float)

    timeserie = np.ravel(data[data.columns[2:]])
    # Drop missing data
    timeserie = timeserie[timeserie != 9999.]
    name = data['pos'][0][3:]
    return name, timeserie

name, timeserie = load(filename)


# %%
# Simple prediction

# Choose a time window rather at the end
N_SAMPLES = 800
y = timeserie[-N_SAMPLES:]

from matplotlib import pyplot as plt

# Set up figures look and feel
import style_figs

# %%
# Polynomial regression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV

t = np.linspace(-2, 2, N_SAMPLES)
t_test = np.linspace(-2.01, 2.01, int(1.005 * N_SAMPLES))
t_train = t[int(.1 * N_SAMPLES):int(.9 * N_SAMPLES)]
y_train = y[int(.1 * N_SAMPLES):int(.9 * N_SAMPLES)]

plt.figure()
plt.plot(t, y, color='.7')
plt.plot(t_train, y_train, color='.4')
plt.subplots_adjust(top=.96)
plt.ylim(-1000, 8000)
plt.xlim(-2.03, 2.03)
style_figs.light_axis()
plt.savefig('tide.pdf', facecolor='none',
            edgecolor='none')

for d in (10, 100, 1000):
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(t.reshape(-1, 1), y)
    plt.plot(t_test, model.predict(t_test.reshape(-1, 1)),
             label='dim=%d' % d)

    plt.subplots_adjust(top=.96)
    plt.ylim(-1000, 8000)
    plt.xlim(-2.03, 2.03)
    plt.legend(loc='lower center', borderaxespad=0, borderpad=0, ncol=2)

    style_figs.light_axis()
    plt.savefig('tide_polynome_%d.pdf' % d, facecolor='none',
                edgecolor='none')

# %%
# Plot the corresponding basis

plt.figure(figsize=[5.12, 3])

for d in (10, 100, 1000):
    transformer = PolynomialFeatures(degree=d)
    transformer.fit(t.reshape(-1, 1), y)
    basis = transformer.transform(t_test.reshape(-1, 1))
    for i in range(2, 10):
        this_signal = basis[:, -i]
        this_signal /= this_signal.max()
        plt.plot(t_test, this_signal, linewidth=2, color='.75')

    this_signal = basis[:, -3]
    this_signal /= this_signal.max()

    this_signal = basis[:, -1]
    this_signal /= this_signal.max()
    plt.plot(t_test, this_signal,
             label='Degree %d' % d)

    #style_figs.no_axis()
    plt.subplots_adjust(top=.96)
    plt.xlim(-2.03, 2.03)

    style_figs.light_axis()
    plt.savefig('polynome_basis_%d.pdf' % d, facecolor='none',
                edgecolor='none')

# %%
# Cosine and sine basis transform

from sklearn.base import BaseEstimator, TransformerMixin

class PeriodicBasis(BaseEstimator, TransformerMixin):

    def __init__(self, degree=10):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples, n_features = X.shape
        assert(n_features == 1)
        out = np.empty((n_samples, self.degree))
        for i in range(int(self.degree / 2)):
            out[:, 2 * i] = np.cos(i * .5 * np.pi * X[:, 0])
            out[:, 2 * i + 1] = np.sin(i * .5 * np.pi * X[:, 0])
        return out

# %%
# Plot the corresponding basis

plt.figure(figsize=[5.12, 3])

for d in (10, 100, 1000):
    transformer = PeriodicBasis(degree=d)
    transformer.fit(t.reshape(-1, 1), y)
    basis = transformer.transform(t_test.reshape(-1, 1))
    for i in range(2, 10):
        this_signal = basis[:, -i]
        this_signal /= this_signal.max()
        plt.plot(t_test, this_signal, linewidth=2, color='.75')

    this_signal = basis[:, -3]
    this_signal /= this_signal.max()

    this_signal = basis[:, -1]
    this_signal /= this_signal.max()
    plt.plot(t_test, this_signal,
             label='Degree %d' % d)

    #style_figs.no_axis()
    plt.subplots_adjust(top=.96)
    plt.xlim(-2.03, 2.03)

    style_figs.light_axis()
    plt.savefig('cosine_basis_%d.pdf' % d, facecolor='none',
                edgecolor='none')

# %%
# Cosine regression


plt.figure()
plt.plot(t, y, color='.7')
plt.plot(t_train, y_train, color='.4')


for d in (10, 100, 1000):
    model = make_pipeline(PeriodicBasis(degree=d), RidgeCV())
    model.fit(t.reshape(-1, 1), y)
    plt.plot(t_test, model.predict(t_test.reshape(-1, 1)),
             label='dim=%d' % d)

    #style_figs.no_axis()
    plt.subplots_adjust(top=.96)
    plt.ylim(-1000, 8000)
    plt.xlim(-2.03, 2.03)
    plt.legend(loc='lower center', borderaxespad=0, borderpad=0, ncol=2)

    style_figs.light_axis()
    plt.savefig('tide_periodic_%d.pdf' % d, facecolor='none',
                edgecolor='none')

