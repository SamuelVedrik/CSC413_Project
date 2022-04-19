import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (13, 7)
plt.style.use("seaborn")

def plot_losses(train_loss, val_loss, save_path):
    plt.plot(train_loss, label="training loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend()
    plt.savefig(save_path)
    plt.clf()
    
    
def plot_accuracies(train_acc, val_acc, save_path):
    plt.plot(train_acc, label="training acc")
    plt.plot(val_acc, label="validation acc")
    plt.legend()
    plt.savefig(save_path)
    plt.clf()
    
def plot_confusion(conf_matrix, labels, title, save_path):
    plt.imshow(conf_matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(10), labels, rotation=45)
    plt.yticks(np.arange(10), labels, rotation=0)
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    
    
    