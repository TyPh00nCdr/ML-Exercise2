import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
iterations = 2000


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

    error = np.zeros(iterations)
    for i in range(iterations):
        for data_point in data:
            adjust_weight(data_point[0], data_point[1], data_point[2])
        error[i] = 0.5 * sum([(y_func(x[0]) - x[1]) ** 2 for x in data])

    np.savetxt('models/theta_model.txt', theta)
    np.savetxt('models/error.txt', error)

    diff_err = next(idx for idx, err
                    in enumerate([abs(t - s) for s, t in zip(error, error[1:])])
                    if err <= 0.0001) + 1
    print('Error smaller than 1/10000 after {} iterations'.format(diff_err))

    x_axis1 = np.linspace(data[:, 0].min(), data[:, 0].max())
    x_axis2 = np.arange(0, len(error))
    # astrix syntax: list all rows individually (equal to x, y = green_cloud.T)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(*green_cloud.T, 'go')
    ax1.plot(*red_cloud.T, 'ro')
    ax1.plot(x_axis1, y_func(x_axis1), 'b-')
    ax2.plot(x_axis2, error)
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
