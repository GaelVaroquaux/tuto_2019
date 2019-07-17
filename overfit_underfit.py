import numpy as np
from matplotlib import pyplot as plt

# Set up figures look and feel
import style_figs

# %%
# Our data-generating process

def f(t):
    return 1.2 * t ** 2 + .1 * t ** 3 - .4 * t ** 5 - .5 * t ** 9

rng = np.random.RandomState(0)
x = 2 * rng.rand(100) - 1

y = f(x) + .4 * rng.normal(size=100)

plt.figure()
plt.scatter(x, y, s=4)

# %%
# Our model (polynomial regression)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# %%
# Fit model with various complexity in the polynomial degree

plt.figure()
plt.scatter(x, y, s=4)

t = np.linspace(-1, 1, 100)

for d in (1, 2, 5, 9):
    model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    plt.plot(t, model.predict(t.reshape(-1, 1)), label='Degree %d' % d)

style_figs.light_axis()

plt.legend(loc='best')

plt.savefig('polynomial_overfit.pdf')

