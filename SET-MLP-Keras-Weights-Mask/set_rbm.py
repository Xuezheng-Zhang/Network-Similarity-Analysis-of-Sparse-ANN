import json
import numpy as np
import os
from scipy.sparse import lil_matrix, dok_matrix, csr_matrix, save_npz
# import sparseoperations
import datetime
import matplotlib.pyplot as plt

from tqdm import tqdm


try:
    from tensorflow.keras.datasets import mnist
except ImportError:
    from keras.datasets import mnist

def contrastive_divergence_updates_Numpy(wDecay, lr, DV, DH, MV, MH, rows, cols, out):
    for i in range (out.shape[0]):
        s1=0
        s2=0
        for j in range(DV.shape[0]):
            s1+=DV[j,rows[i]]*DH[j, cols[i]]
            s2+=MV[j,rows[i]]*MH[j, cols[i]]
        out[i]+=lr*(s1/DV.shape[0]-s2/DV.shape[0])-wDecay*out[i]
    #return out

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def createSparseWeights(epsilon,noRows,noCols):
    # generate an Erdos Renyi sparse weights mask
    weights=lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0,noRows),np.random.randint(0,noCols)]=np.float64(np.random.randn()/20)
    print ("Create sparse matrix with ",weights.getnnz()," connections and ",(weights.getnnz()/(noRows * noCols))*100,"% density level")
    weights=weights.tocsr()
    return weights

class Sigmoid:
    @staticmethod
    def activation(z):

        return 1 / (1 + np.exp(-z))

    def activationStochastic(z):
        z=Sigmoid.activation(z)
        za=z.copy()
        prob=np.random.uniform(0,1,(z.shape[0],z.shape[1]))
        za[za>prob]=1
        za[za<=prob]=0
        return za


GRAPH_SNAPSHOTS_RBM_BASE = "SET-MLP-Keras-Weights-Mask/results/graph_snapshots_rbm"
RESULTS_DIR = "SET-MLP-Keras-Weights-Mask/results"


class SET_RBM:
    def __init__(self, noVisible, noHiddens, epsilon=10, run_id=0):
        self.noVisible = noVisible  # number of visible neurons
        self.noHiddens = noHiddens  # number of hidden neurons
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.run_id = run_id

        self.learning_rate = None
        self.weight_decay = None
        self.zeta = None

        self.W = createSparseWeights(self.epsilon, self.noVisible, self.noHiddens)
        self.bV = np.zeros(self.noVisible)
        self.bH = np.zeros(self.noHiddens)

    def save_snapshot(self, snapshot_dir, epoch, stage, W=None):
        """
        Save weights and mask to graph_snapshots_rbm, compatible with analyze_networks format.
        RBM has only one weight layer (visible -> hidden).
        """
        W = self.W if W is None else W
        epoch_dir = os.path.join(snapshot_dir, f"epoch_{epoch:04d}", stage)
        os.makedirs(epoch_dir, exist_ok=True)

        # W is sparse (csr), convert to dense for consistency with set_mlp format, then save as sparse
        W_dense = W.toarray() if hasattr(W, 'toarray') else np.asarray(W)
        mask = (W_dense != 0).astype(np.float64)

        weight_file = os.path.join(epoch_dir, "weight_layer_1.npz")
        mask_file = os.path.join(epoch_dir, "mask_layer_1.npz")
        save_npz(weight_file, csr_matrix(W_dense))
        save_npz(mask_file, csr_matrix(mask))

        sparsity = 1.0 - np.count_nonzero(mask) / mask.size
        tqdm.write(f"  Saved snapshot to {epoch_dir} (sparsity: {sparsity*100:.2f}%)")
        return {"layer_1": sparsity}

    def fit(self, X_train, X_test, batch_size, epochs, lengthMarkovChain=2, weight_decay=0.0000002, learning_rate=0.1, zeta=0.3, testing=True, save_filename="", save_snapshots=True, y_train=None, y_test=None):

        self.lengthMarkovChain = lengthMarkovChain
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.zeta = zeta

        snapshot_dir = os.path.join(GRAPH_SNAPSHOTS_RBM_BASE, f"run_{self.run_id}") if save_snapshots else None
        if save_snapshots:
            os.makedirs(snapshot_dir, exist_ok=True)
            print(f"Run {self.run_id}: snapshot dir {snapshot_dir}")

        metadata_file = os.path.join(RESULTS_DIR, f"training_metadata_rbm_run_{self.run_id}.json") if save_snapshots else None
        metadata_records = []

        minimum_reconstructin_error = 100000
        metrics = np.zeros((epochs, 2))
        reconstruction_error_train = 0

        n_batches = X_train.shape[0] // batch_size
        for i in tqdm(range(epochs), desc=f"Run {self.run_id} Epochs", unit="epoch"):
            # Shuffle the data
            seed = np.arange(X_train.shape[0])
            np.random.shuffle(seed)
            x_ = X_train[seed]

            # training
            t1 = datetime.datetime.now()
            for j in tqdm(range(n_batches), desc=f"Epoch {i} batches", unit="batch", leave=False):
                k = j * batch_size
                l = (j + 1) * batch_size
                reconstruction_error_train += self.learn(x_[k:l])
            t2 = datetime.datetime.now()

            reconstruction_error_train = reconstruction_error_train / n_batches
            metrics[i, 0] = reconstruction_error_train

            reconstruction_error_test = None
            val_accuracy = None
            if testing:
                t3 = datetime.datetime.now()
                reconstruction_error_test = self.reconstruct(X_test)
                t4 = datetime.datetime.now()
                metrics[i, 1] = reconstruction_error_test
                minimum_reconstructin_error = min(minimum_reconstructin_error, reconstruction_error_test)
                if y_train is not None and y_test is not None:
                    val_accuracy = eval_accuracy_on_hidden(self, X_train, y_train, X_test, y_test)

            tqdm.write(f"\nRun {self.run_id} | Epoch {i+1}/{epochs}")
            parts = []
            if val_accuracy is not None:
                parts.append(f"Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            if reconstruction_error_test is not None:
                parts.append(f"Reconstruction Error (val): {reconstruction_error_test:.4f}")
            if parts:
                tqdm.write(", ".join(parts))
            tqdm.write(f"Reconstruction Error (train): {reconstruction_error_train:.4f}")

            # save snapshot after training
            sparsity_after_training = None
            if save_snapshots and snapshot_dir:
                sparsity_after_training = self.save_snapshot(snapshot_dir, i, "after_training")

            # change connectivity pattern
            t5 = datetime.datetime.now()
            if i < epochs - 1:
                self.weightsEvolution(addition=True)
            else:
                if i == epochs - 1:
                    self.weightsEvolution(addition=False)
            t6 = datetime.datetime.now()
            tqdm.write(f"Weights evolution time: {t6 - t5}")

            # save snapshot after pruning
            sparsity_after_pruning = None
            if save_snapshots and snapshot_dir:
                sparsity_after_pruning = self.save_snapshot(snapshot_dir, i, "after_pruning")
                if sparsity_after_pruning is None:
                    sparsity_after_pruning = sparsity_after_training

            # append training metadata
            if save_snapshots and metadata_file and sparsity_after_training:
                rec = {
                    "epoch": int(i),
                    "reconstruction_error_train": float(metrics[i, 0]),
                    "reconstruction_error_test": float(metrics[i, 1]) if reconstruction_error_test is not None else None,
                    "val_accuracy": float(val_accuracy) if val_accuracy is not None else None,
                    "val_loss": float(metrics[i, 1]) if reconstruction_error_test is not None else None,  # RBM: reconstruction error
                    "sparsity_after_training": {k: float(v) for k, v in sparsity_after_training.items()},
                    "sparsity_after_pruning": {k: float(v) for k, v in (sparsity_after_pruning or sparsity_after_training).items()},
                }
                metadata_records.append(rec)

            if save_filename != "":
                np.savetxt(save_filename, metrics)

        if save_snapshots and metadata_file and metadata_records:
            with open(metadata_file, 'w') as f:
                json.dump(metadata_records, f, indent=2)
            print(f"Metadata saved to {metadata_file}")

    def runMarkovChain(self,x):
        self.DV=x
        self.DH=self.DV@self.W  + self.bH
        self.DH=Sigmoid.activationStochastic(self.DH)

        for i in range(1,self.lengthMarkovChain):
            if (i==1):
                self.MV = self.DH @ self.W.transpose() + self.bV
            else:
                self.MV = self.MH @ self.W.transpose() + self.bV
            self.MV = Sigmoid.activation(self.MV)
            self.MH=self.MV@self.W  + self.bH
            self.MH = Sigmoid.activationStochastic(self.MH)

    def reconstruct(self,x):
        self.runMarkovChain(x)
        return (np.mean((self.DV-self.MV)*(self.DV-self.MV)))

    def learn(self,x):
        self.runMarkovChain(x)
        self.update()
        return (np.mean((self.DV - self.MV) * (self.DV - self.MV)))

    def getRecontructedVisibleNeurons(self,x):
        #return recontructions of the visible neurons
        self.reconstruct(x)
        return self.MV

    def getHiddenNeurons(self,x):
        # return hidden neuron values
        self.reconstruct(x)
        return self.MH


    def weightsEvolution(self,addition):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # TODO: this method could be seriously improved in terms of running time using Cython
        values=np.sort(self.W.data)
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)

        largestNegative = values[int((1-self.zeta) * firstZeroPos)]
        smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

        wlil = self.W.tolil()
        wdok = dok_matrix((self.noVisible,self.noHiddens),dtype="float64")

        # remove the weights closest to zero
        keepConnections=0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if (((val < largestNegative) or (val > smallestPositive))):
                    wdok[ik,jk]=val
                    keepConnections+=1

        # add new random connections
        if (addition):
            for kk in range(self.W.data.shape[0]-keepConnections):
                ik = np.random.randint(0, self.noVisible)
                jk = np.random.randint(0, self.noHiddens)
                while ((wdok[ik,jk]!=0)):
                    ik = np.random.randint(0, self.noVisible)
                    jk = np.random.randint(0, self.noHiddens)
                wdok[ik, jk]=np.random.randn() / 20

        self.W=wdok.tocsr()

    def update(self):
        #compute Contrastive Divergence updates
        self.W=self.W.tocoo()
        # sparseoperations.contrastive_divergence_updates_Cython(self.weight_decay, self.learning_rate, self.DV, self.DH, self.MV, self.MH, self.W.row, self.W.col, self.W.data)
        # If you have problems with Cython please use the contrastive_divergence_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        contrastive_divergence_updates_Numpy(self.weight_decay, self.learning_rate, self.DV, self.DH, self.MV, self.MH, self.W.row, self.W.col, self.W.data)

        # perform the weights update
        # TODO: adding momentum would make learning faster
        self.W=self.W.tocsr()
        self.bV=self.bV+self.learning_rate*(np.mean(self.DV,axis=0)-np.mean(self.MV,axis=0))-self.weight_decay*self.bV
        self.bH = self.bH + self.learning_rate * (np.mean(self.DH, axis=0) - np.mean(self.MH, axis=0)) - self.weight_decay * self.bH

def load_mnist(return_labels=False):
    """Load MNIST dataset, normalized to [0, 1] for RBM."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float64') / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float64') / 255.0
    if return_labels:
        return x_train, x_test, y_train, y_test
    return x_train, x_test


def eval_accuracy_on_hidden(rbm, X_train, y_train, X_test, y_test):
    """
    Evaluate classification accuracy using hidden representations + logistic regression.
    Returns val_accuracy in [0, 1].
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    h_train = rbm.getHiddenNeurons(X_train)
    h_test = rbm.getHiddenNeurons(X_test)
    clf = LogisticRegression(max_iter=200, solver='lbfgs')
    clf.fit(h_train, y_train)
    return clf.score(h_test, y_test)


NUM_RUNS = 1
NO_HIDDENS = 200
EPSILON = 10
EPOCHS = 10
BATCH_SIZE = 256

if __name__ == "__main__":
    np.random.seed(0)

    x_train, x_test, y_train, y_test = load_mnist(return_labels=True)
    no_visible = x_train.shape[1]  # 784 for MNIST

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPH_SNAPSHOTS_RBM_BASE, exist_ok=True)

    for run_id in range(NUM_RUNS):
        print(f"\nTraining SET-RBM run {run_id + 1}/{NUM_RUNS} ...")
        setrbm = SET_RBM(no_visible, noHiddens=NO_HIDDENS, epsilon=EPSILON, run_id=run_id)
        setrbm.fit(
            x_train, x_test,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lengthMarkovChain=2,
            weight_decay=0.0000002,
            learning_rate=0.1,
            zeta=0.3,
            testing=True,
            save_filename=os.path.join(RESULTS_DIR, f"set_rbm_run_{run_id}.txt"),
            save_snapshots=True,
            y_train=y_train, y_test=y_test,
        )

        reconstructions = setrbm.getRecontructedVisibleNeurons(x_test)
        print("\nReconstruction error on test (last epoch):", np.mean((reconstructions - x_test) ** 2))

    print(f"\nDone. {NUM_RUNS} models saved to {GRAPH_SNAPSHOTS_RBM_BASE}/run_0..run_{NUM_RUNS-1}")

