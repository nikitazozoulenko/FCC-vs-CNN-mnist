import matplotlib.pyplot as plt

def losses_to_ewma(losses, alpha = 0.9):
    losses_ewma = []
    ewma = losses[0]
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma


def graph(fcc, cnn):
    plt.figure(1)

    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    
    plt.plot(cnn.x_indices, cnn.losses, "r", label="CNN Training Loss")
    plt.plot(cnn.val_x_indices, cnn.val_losses, "r--", label="CNN Validation Loss")
    plt.plot(fcc.x_indices, fcc.losses, "g", label="FCC Training Loss")
    plt.plot(fcc.val_x_indices, fcc.val_losses, "g--", label="FCC Validation Loss")
    plt.legend(loc=1)

    plt.figure(2)

    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    
    plt.plot(cnn.acc_indices, cnn.acc, "r", label="CNN Accuracy")
    plt.plot(fcc.acc_indices, fcc.acc, "g", label="FCC Accuracy")
    plt.legend(loc=1)

    plt.show()
