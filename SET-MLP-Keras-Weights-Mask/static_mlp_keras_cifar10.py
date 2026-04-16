"""
Static Sparse MLP (no rewiring)
The mask stays fixed throughout training.
"""
from __future__ import division
from __future__ import print_function
import json
import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz, load_npz

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, ReLU
    from tensorflow.keras import optimizers, backend as K
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.constraints import Constraint
    from tensorflow.keras.regularizers import l2
except ImportError:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, ReLU
    from keras import optimizers, backend as K
    from keras.datasets import mnist
    from keras.utils import to_categorical
    from keras.regularizers import l2
    try:
        from keras.constraints import Constraint
    except ImportError:
        class Constraint(object):
            def __call__(self, w):
                return w
            def get_config(self):
                return {}


class MaskWeights(Constraint):
    def __init__(self, mask):
        self.mask = K.cast(mask, K.floatx())

    def __call__(self, w):
        return w * self.mask

    def get_config(self):
        return {'mask': K.get_value(self.mask)}

def createWeightsMask(epsilon, noRows, noCols):
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print(f"Create Sparse Matrix: No parameters: {noParameters}, Shape: ({noRows}, {noCols})")
    return [noParameters, mask_weights]

class STATIC_MLP_CIFAR10:
    def __init__(self, run_id=0):
        self.run_id = run_id
        self.epsilon = 20
        self.batch_size = 100
        self.zeta = 0.3
        self.maxepoches = 200
        self.learning_rate = 3e-5
        self.weight_decay = 1e-4
        self.num_classes = 10
        self.momentum = 0.9

        # 784 -> 512 -> 256 -> 128 -> 10 (same architecture as SET-MLP)
        [self.noPar1, self.wm1] = createWeightsMask(self.epsilon, 784, 512)
        [self.noPar2, self.wm2] = createWeightsMask(self.epsilon, 512, 256)
        [self.noPar3, self.wm3] = createWeightsMask(self.epsilon, 256, 128)

        self.create_model()
        self.train()

    def create_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28, 1)))

        reg = l2(self.weight_decay)
        self.model.add(Dense(512, name="sparse_1", kernel_constraint=MaskWeights(self.wm1), kernel_regularizer=reg, use_bias=True))
        self.model.add(ReLU(name="srelu1"))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(256, name="sparse_2", kernel_constraint=MaskWeights(self.wm2), kernel_regularizer=reg, use_bias=True))
        self.model.add(ReLU(name="srelu2"))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(128, name="sparse_3", kernel_constraint=MaskWeights(self.wm3), kernel_regularizer=reg, use_bias=True))
        self.model.add(ReLU(name="srelu3"))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(self.num_classes, name="dense_4", kernel_regularizer=reg, use_bias=True))
        self.model.add(Activation('softmax'))

    def get_weights_and_masks(self):
        weights = {
            'layer_1': np.array(self.model.get_layer("sparse_1").get_weights()[0]),
            'layer_2': np.array(self.model.get_layer("sparse_2").get_weights()[0]),
            'layer_3': np.array(self.model.get_layer("sparse_3").get_weights()[0]),
            'layer_4': np.array(self.model.get_layer("dense_4").get_weights()[0])
        }
        masks = {
            'layer_1': np.array(self.wm1),
            'layer_2': np.array(self.wm2),
            'layer_3': np.array(self.wm3)
        }
        return weights, masks

    def calculate_sparsity(self, masks):
        sparsity = {}
        for layer_name, mask_matrix in masks.items():
            sparsity[layer_name] = 1.0 - np.count_nonzero(mask_matrix) / mask_matrix.size
        return sparsity

    def save_weights_and_masks(self, output_dir, epoch, stage='', weights=None, masks=None):
        if weights is None or masks is None:
            model_weights, model_masks = self.get_weights_and_masks()
            if weights is None:
                weights = model_weights
            if masks is None:
                masks = model_masks

        epoch_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
        if stage:
            epoch_dir = os.path.join(epoch_dir, stage)
        os.makedirs(epoch_dir, exist_ok=True)

        for layer_name, weight_matrix in weights.items():
            filename = os.path.join(epoch_dir, f"weight_{layer_name}.npz")
            save_npz(filename, csr_matrix(weight_matrix))

        for layer_name, mask_matrix in masks.items():
            filename = os.path.join(epoch_dir, f"mask_{layer_name}.npz")
            save_npz(filename, csr_matrix(mask_matrix))

        sparsity = self.calculate_sparsity(masks)
        print(f"  Saved weights and masks to {epoch_dir}")
        return sparsity

    def append_training_metadata(self, metadata_file, epoch, sparsity, val_accuracy, val_loss):
        def to_float(d):
            return {k: float(v) for k, v in d.items()}

        record = {
            "epoch": int(epoch),
            "val_accuracy": float(val_accuracy),
            "val_loss": float(val_loss),
            "sparsity": to_float(sparsity),
        }

        if not os.path.exists(metadata_file):
            records = []
        else:
            with open(metadata_file, 'r') as f:
                records = json.load(f)
        records.append(record)
        with open(metadata_file, 'w') as f:
            json.dump(records, f, indent=2)

    def train(self):
        [x_train, x_test, y_train, y_test] = self.read_data()
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)

        snapshot_dir = os.path.join("./SET-MLP-Keras-Weights-Mask/results/graph_snapshots_static", f"run_{self.run_id}")
        os.makedirs(snapshot_dir, exist_ok=True)
        print(f"Run {self.run_id}: snapshot dir {snapshot_dir}")

        metadata_file = os.path.join("SET-MLP-Keras-Weights-Mask/results", f"training_metadata_static_run_{self.run_id}.json")
        with open(metadata_file, 'w') as f:
            json.dump([], f)

        try:
            sgd = optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        except TypeError:
            sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.accuracies_per_epoch = []
        for epoch in range(self.maxepoches):
            print(f"\nRun {self.run_id} | Epoch {epoch+1}/{self.maxepoches}")

            history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=self.batch_size),
                steps_per_epoch=len(x_train) // self.batch_size,
                epochs=1,
                validation_data=(x_test, y_test),
                verbose=1
            )

            acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
            loss_key = 'val_loss' if 'val_loss' in history.history else 'loss'
            val_accuracy = history.history[acc_key][0]
            val_loss = history.history[loss_key][0]

            self.accuracies_per_epoch.append(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

            # save weights and masks (mask never changes)
            sparsity = self.save_weights_and_masks(snapshot_dir, epoch, stage='after_training')

            # save training metadata
            self.append_training_metadata(metadata_file, epoch, sparsity, val_accuracy, val_loss)

    def read_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        xTrainMean = np.mean(x_train, axis=0)
        xTrainStd = np.std(x_train, axis=0)
        xTrainStd = np.where(xTrainStd < 1e-7, 1.0, xTrainStd)
        x_train = (x_train - xTrainMean) / xTrainStd
        x_test = (x_test - xTrainMean) / xTrainStd

        return [x_train, x_test, y_train, y_test]

NUM_RUNS = 1

if __name__ == '__main__':
    os.makedirs('SET-MLP-Keras-Weights-Mask/results', exist_ok=True)
    for run_id in range(NUM_RUNS):
        print(f"Training Static MLP run {run_id + 1}/{NUM_RUNS} ...\n")
        model = STATIC_MLP_CIFAR10(run_id=run_id)
    print(f"\nDone. {NUM_RUNS} models saved to graph_snapshots_static/run_0..run_{NUM_RUNS-1}")
