import numpy as np
from matplotlib import pyplot as plt

# Set up figures look and feel
import style_figs

# %%
# Our data-generating process

def f(t):
    return 1.2 * t ** 2 + .1 * t ** 3 - .4 * t ** 5 - .5 * t ** 9

N_SAMPLES = 50

rng = np.random.RandomState(0)
x = 2 * rng.rand(N_SAMPLES) - 1

y = f(x) + .4 * rng.normal(size=N_SAMPLES)

plt.figure()
plt.scatter(x, y, s=20, color='k')

style_figs.no_axis()
plt.subplots_adjust(top=.96)
plt.xlim(-1.1, 1.1)
plt.ylim(-.74, 2.1)
plt.savefig('polynomial_overfit_0.pdf', facecolor='none', edgecolor='none')

# %%
# Our model (polynomial regression)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# %%
# Fit model with various complexity in the polynomial degree

plt.figure()
plt.scatter(x, y, s=20, color='k')

t = np.linspace(-1, 1, 100)

for d in (1, 2, 5, 9):
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    plt.plot(t, model.predict(t.reshape(-1, 1)), label='Degree %d' % d)

    style_figs.no_axis()
    plt.legend(loc='upper center', borderaxespad=0, borderpad=0)
    plt.subplots_adjust(top=.96)
    plt.ylim(-.74, 2.1)

    plt.savefig('polynomial_overfit_%d.pdf' % d, facecolor='none',
                edgecolor='none')

plt.plot(t, f(t), 'k--', label='Truth')

style_figs.no_axis()
plt.legend(loc='upper center', borderaxespad=0, borderpad=0)
plt.ylim(-.74, 2.1)
plt.subplots_adjust(top=.96)

plt.savefig('polynomial_overfit.pdf', facecolor='none', edgecolor='none')

# %%
# A figure with the true model and the estimated one

plt.figure(figsize=[.5 * 6.4, .5 * 4.8])
plt.scatter(x, y, s=20, color='k')
plt.plot(t, model.predict(t.reshape(-1, 1)), color='C3',
         label='$\hat{f}$')

plt.plot(t, f(t), 'k--', label='$f^{\star}$')
style_figs.no_axis()
plt.legend(loc='upper center', borderaxespad=0, borderpad=0, fontsize=26)
plt.subplots_adjust(top=.96)

plt.savefig('polynomial_overfit_simple.pdf', facecolor='none', edgecolor='none')

# %%
# Assymptotic settings

rng = np.random.RandomState(0)
x = 2 * rng.rand(10 * N_SAMPLES) - 1

y = f(x) + .4 * rng.normal(size=10 * N_SAMPLES)


model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
model.fit(x.reshape(-1, 1), y)

plt.figure(figsize=[.5 * 6.4, .5 * 4.8])
plt.scatter(x, y, s=20, color='k', alpha=.3)
plt.plot(t, model.predict(t.reshape(-1, 1)), color='C0',
         label='$\hat{f} \\approx f^{\star}$')

plt.plot(t, f(t), 'k--', label='$g$')
style_figs.no_axis()
plt.legend(loc='upper center', borderaxespad=0, borderpad=0, fontsize=26)
plt.subplots_adjust(top=.96)

plt.savefig('polynomial_overfit_assymptotic.pdf', facecolor='none', edgecolor='none')

