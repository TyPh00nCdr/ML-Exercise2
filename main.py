import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

alpha = 2.37
iterations = 100


def main():
    data = np.loadtxt('data/data.txt')

    # indexing and slicing: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    # 1) bool index all rows where 3rd column is 1
    # 2) index all rows, and all columns until index 2 (exclusive)
    green_cloud = data[data[:, 2] == 1][:, :2]
    red_cloud = data[data[:, 2] == 0][:, :2]

    global theta
    # theta = np.random.uniform(-0.01, 0.01, 3)
    # np.savetxt('models/theta_init.txt', theta)
    theta = np.loadtxt('models/theta_init.txt')

    default_error = np.zeros(iterations)
    for i in range(iterations):
        for data_point in data:
            adjust_weight(data_point[0], data_point[1], data_point[2])
        default_error[i] = 0.5 * sum([(g_func(x[0], x[1]) - x[2]) ** 2 for x in data])

    rms_error = [np.sqrt((2 * i)/ len(data)) for i in default_error]

    np.savetxt('models/theta_model.txt', theta)
    np.savetxt('models/error.txt', default_error)
    np.savetxt('models/rms_error.txt', rms_error)

    diff_err = next(idx for idx, err
                    in enumerate([abs(t - s) for s, t in zip(default_error, default_error[1:])])
                    if err <= 0.001) + 1
    print('Error smaller than 1/1000 after {} iterations'.format(diff_err))

    x_axis1 = np.linspace(data[:, 0].min(), data[:, 0].max())
    x_axis2 = np.arange(0, len(default_error))
    # astrix syntax: list all rows individually (equal to x, y = green_cloud.T)
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, :])  # row 1, span all columns
    ax1.plot(*green_cloud.T, 'go')
    ax1.plot(*red_cloud.T, 'ro')
    ax1.plot(x_axis1, y_func(x_axis1), 'b-')

    ax2 = fig.add_subplot(gs[1, 0])  # row 0, col 0
    ax2.plot(x_axis2, default_error)

    ax3 = fig.add_subplot(gs[1, 1])  # row 0, col 1
    ax3.plot(x_axis2, rms_error)

    plt.show()


def g_func(x1, x2):
    return 1.0 / (1.0 + np.exp(-np.dot(theta, [1, x1, x2])))


def y_func(x):
    return (theta[0] + theta[1] * x) * (-1.0 / theta[2])


def adjust_weight(x1, x2, label):
    global theta
    diff = label - g_func(x1, x2)
    theta = [the + alpha * diff * x for the, x in zip(theta, [1, x1, x2])]


# --- Main-Function declaration ---
if __name__ == '__main__':
    main()
