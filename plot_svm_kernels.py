import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Our dataset and targets
X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-2, -1.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (-1, 1),
          # --
          (2.1, 1),
          (1.3, .8),
          (1.2, .5),
          (1.1, 1.5),
          (1.5, -1.3),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = np.array([0] * 10 + [1] * 10)

# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2, C=.01)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(.65*4, .65*3))
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    if kernel == 'rbf':
        plt.contour(XX, YY, Z, colors=[.5 * np.array(plt.cm.Paired(0)), 'k',
                    plt.cm.Paired(255)],
                    linestyles=[':', '--', ':'],
                    levels=[-.01, 0, .01])
    else:
        plt.contour(XX, YY, Z, colors=[plt.cm.Paired(0), 'k',
                    plt.cm.Paired(255)],
                    linestyles=[':', '--', ':'],
                    levels=[-.25, 0, .25])
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
    plt.tight_layout(pad=.01)
    plt.savefig('plot_svm_%s.pdf' % kernel)
