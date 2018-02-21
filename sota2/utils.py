import matplotlib.pyplot as plt

def losses_to_ewma(losses, alpha = 0.999):
    losses_ewma = []
    ewma = losses[0]
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma


def graph(cnn):
    plt.figure(1)
    plt.plot(cnn.x_indices, cnn.losses, "r", label="CNN Training Loss")
    plt.plot(cnn.val_x_indices, cnn.val_losses, "g--", label="CNN Validation Loss")
    plt.legend(loc=1)

    plt.figure(2)
    plt.plot(cnn.acc_indices, cnn.acc, "b", label="CNN Accuracy")
    plt.legend(loc=1)

    plt.show()
