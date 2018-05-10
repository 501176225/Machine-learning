from numpy import *

def makeLinearSeparableData(weights, numLines):
    ''' (list, int) -> array

    Return a linear Separable data set. 
    Randomly generate numLines points on both sides of 
    the hyperplane weights * x = 0.

    Notice: weights and x are vectors.

    >>> data = pla.makeLinearSeparableData([2,3],5)
    >>> data
    array([[ 0.54686091,  3.60017244,  1.        ],
           [ 2.0201362 ,  7.5046425 ,  1.        ],
           [-3.14522458, -7.19333582, -1.        ],
           [ 9.72172678, -7.99611918, -1.        ],
           [ 9.68903615,  2.10184495,  1.        ]])
>>> data = pla.makeLinearSeparableData([4,3,2],10)
>>> data
array([[ -4.74893955e+00,  -5.38593555e+00,   1.22988454e+00,   -1.00000000e+00],
       [  4.13768071e-01,  -2.64984892e+00,  -5.45073234e-03,   -1.00000000e+00],
       [ -2.17918583e+00,  -6.48560310e+00,  -3.96546373e+00,   -1.00000000e+00],
       [ -4.34244286e+00,   4.24327022e+00,  -5.32551053e+00,   -1.00000000e+00],
       [ -2.55826469e+00,   2.65490732e+00,  -6.38022703e+00,   -1.00000000e+00],
       [ -9.08136968e+00,   2.68875119e+00,  -9.09804786e+00,   -1.00000000e+00],
       [ -3.80332893e+00,   7.21070373e+00,  -8.70106682e+00,   -1.00000000e+00],
       [ -6.49790176e+00,  -2.34409845e+00,   4.69422613e+00,   -1.00000000e+00],
       [ -2.57471371e+00,  -4.64746879e+00,  -2.44909463e+00,   -1.00000000e+00],
       [ -5.80930468e+00,  -9.34624147e+00,   6.54159660e+00,   -1.00000000e+00]])
    '''
    w = array(weights)
    numFeatures = len(weights)
    dataSet = zeros((numLines, numFeatures + 1))

    for i in range(numLines):
        x = random.rand(1, numFeatures) * 20 - 10
        innerProduct = sum(w * x)
        if innerProduct <= 0:
            dataSet[i] = append(x, -1)
        else:
            dataSet[i] = append(x, 1)

    return dataSet


def plotData(dataSet):
    ''' (array) -> figure

    Plot a figure of dataSet

    '''

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = array(dataSet[:,2])
    idx_1 = where(dataSet[:,2]==1)
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], marker='o', color='g', label=1, s=20)
    idx_2 = where(dataSet[:,2]==-1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], marker='x', color='r', label=2, s=20)
    plt.legend(loc = 'upper right')
    plt.show()


def train(dataSet, plot = False):

    numLines = dataSet.shape[0]
    numFeatures = dataSet.shape[1]
    w = zeros((1, numFeatures - 1)) #an array which save weight

    i = 0
    while i < numLines:
        if dataSet[i][-1] * sum(w * dataSet[i,0:-1]) <= 0:
            w = w + dataSet[i][-1] * dataSet[i,0:-1]
            i = 0
        else:
            i += 1
    
    if plot == True:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Linear separable data set')
        plt.xlabel('X')
        plt.ylabel('Y')
        labels = array(dataSet[:,2])
        idx_1 = where(dataSet[:,2]==1)
        p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], 
            marker='o', color='g', label=1, s=20)
        idx_2 = where(dataSet[:,2]==-1)
        p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], 
            marker='x', color='r', label=2, s=20)
        x = w[0][0] / abs(w[0][0]) * 10
        y = w[0][1] / abs(w[0][0]) * 10
        ann = ax.annotate(u"",xy=(x,y), 
            xytext=(0,0),size=20, arrowprops=dict(arrowstyle="-|>"))
        ys = (-12 * (-w[0][0]) / w[0][1], 12 * (-w[0][0]) / w[0][1])
        ax.add_line(Line2D((-12, 12), ys, linewidth=1, color='blue'))
        plt.legend(loc = 'upper right')
        plt.show()

    return w

