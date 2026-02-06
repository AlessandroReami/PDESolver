from matplotlib import pyplot as plt


def plot_pointwise_error(color: str = "blue", label: str = "Pointwise error"):
    title_fontsize = 21
    label_fontsize = 19
    figsize = (8.5, 7.5)
    plt.plot([0, 1], [0, 1], color=color, label=label)

    plt.legend(prop={'size': 18})
    plt.show()


def plot_decay():
    title_fontsize = 21
    label_fontsize = 19
    figsize = (8.5, 7.5)

    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], color="blue", label="Error over the different runs")
    # plt.plot([0, 1], [0, 1], color="red", label="Order of convergence 3")
    plt.legend(prop={'size': 18})
    plt.title('Error decay for the Embedded implicit Runge Kutta \n  Lobatto 3B on a European Option', fontsize=title_fontsize)

    # plt.xlabel('Log space subdivisions', fontsize=label_fontsize)
    plt.xlabel('Tolerance', fontsize=label_fontsize)
    plt.ylabel('Error', fontsize=label_fontsize)
    #plt.show()
    plt.savefig("legenda.png")


plot_decay()
