"""
USE THIS MODEL AS A BASELINE
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz, load_npz

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, ReLU
    from tensorflow.keras import optimizers, backend as K
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.constraints import Constraint
except ImportError:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, ReLU
    from keras import optimizers, backend as K
    from keras.datasets import cifar10
    from keras.utils import to_categorical
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

def find_first_pos(array, value):
    return (np.abs(array - value)).argmin()

def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def createWeightsMask(epsilon, noRows, noCols):
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    print(f"Create Sparse Matrix: No parameters: {noParameters}, Shape: ({noRows}, {noCols})")
    return [noParameters, mask_weights]

class SET_MLP_CIFAR10:
    def __init__(self):
        self.epsilon = 20 
        self.zeta = 0.3 
        self.batch_size = 100 
        self.maxepoches = 10 
        self.learning_rate = 0.01 
        self.num_classes = 10 
        self.momentum = 0.9 

        # initialize masks
        [self.noPar1, self.wm1] = createWeightsMask(self.epsilon, 3072, 4000)
        [self.noPar2, self.wm2] = createWeightsMask(self.epsilon, 4000, 1000)
        [self.noPar3, self.wm3] = createWeightsMask(self.epsilon, 1000, 4000)

        self.w1, self.w2, self.w3, self.w4 = None, None, None, None
        
        self.create_model()
        self.train()

    def create_model(self):
        # if there is any weights data, model weights will be set when training

        # input layer
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(32, 32, 3)))
        
        # first hidden layer
        self.model.add(Dense(4000, name="sparse_1", kernel_constraint=MaskWeights(self.wm1), use_bias=True))
        self.model.add(ReLU(name="srelu1"))
        self.model.add(Dropout(0.3))
        
        # second hidden layer
        self.model.add(Dense(1000, name="sparse_2", kernel_constraint=MaskWeights(self.wm2), use_bias=True))
        self.model.add(ReLU(name="srelu2"))
        self.model.add(Dropout(0.3))
        
        # third hidden layer
        self.model.add(Dense(4000, name="sparse_3", kernel_constraint=MaskWeights(self.wm3), use_bias=True))
        self.model.add(ReLU(name="srelu3"))
        self.model.add(Dropout(0.3))
        
        # output layer
        self.model.add(Dense(self.num_classes, name="dense_4", use_bias=True))
        self.model.add(Activation('softmax'))
        
    def rewireMask(self, weights, noWeights):
        values = np.sort(weights.ravel())
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)
        
        largestNegative = values[int((1-self.zeta) * firstZeroPos)]
        pos_idx = int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))
        smallestPositive = values[pos_idx]
        
        # filter weights
        rewiredWeights = np.zeros_like(weights)
        rewiredWeights[weights > smallestPositive] = 1
        rewiredWeights[weights < largestNegative] = 1
        weightMaskCore = rewiredWeights.copy()

        # add new weights
        nrAdd = 0
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if rewiredWeights[i, j] == 0:
                rewiredWeights[i, j] = 1
                nrAdd += 1
        return [rewiredWeights, weightMaskCore]

    def get_weights_and_masks(self):
        """
        extract the weights and masks from the current model
        """
        weights = {
            'layer_1': np.array(self.model.get_layer("sparse_1").get_weights()[0]),  # (3072, 4000)
            'layer_2': np.array(self.model.get_layer("sparse_2").get_weights()[0]),  # (4000, 1000)
            'layer_3': np.array(self.model.get_layer("sparse_3").get_weights()[0]),  # (1000, 4000)
            'layer_4': np.array(self.model.get_layer("dense_4").get_weights()[0])    # (4000, 10)
        }
        
        masks = {
            'layer_1': np.array(self.wm1),  # (3072, 4000)
            'layer_2': np.array(self.wm2),  # (4000, 1000)
            'layer_3': np.array(self.wm3)   # (1000, 4000)
        }
        
        return weights, masks

    def calculate_sparsity(self, masks):
        """
        calculate the sparsity of the mask matrix
        """
        sparsity = {}
        for layer_name, mask_matrix in masks.items():
            sparsity[layer_name] = 1.0 - np.count_nonzero(mask_matrix) / mask_matrix.size
        return sparsity

    def save_weights_and_masks(self, output_dir, epoch, stage='', weights=None, masks=None):
        """
        save the weights and masks of the current model to the output directory
        """
        # if weights and masks are not provided, get them from the model
        if weights is None or masks is None:
            model_weights, model_masks = self.get_weights_and_masks()
            if weights is None:
                weights = model_weights
            if masks is None:
                masks = model_masks
        
        # create output directory
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
        if stage:
            epoch_dir = os.path.join(epoch_dir, stage)
        os.makedirs(epoch_dir, exist_ok=True)
        
        # save weights
        for layer_name, weight_matrix in weights.items():
            filename = os.path.join(epoch_dir, f"weight_{layer_name}.npz")
            sparse_weight = csr_matrix(weight_matrix)
            save_npz(filename, sparse_weight)
        
        # save masks
        for layer_name, mask_matrix in masks.items():
            filename = os.path.join(epoch_dir, f"mask_{layer_name}.npz")
            sparse_mask = csr_matrix(mask_matrix)
            save_npz(filename, sparse_mask)
        
        sparsity = self.calculate_sparsity(masks)
        print(f"  Saved weights and masks to {epoch_dir}")

        return sparsity

    def weightsEvolution(self):
        """
        evolve the weights
        """
        
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        # update the masks
        [self.wm1, core1] = self.rewireMask(self.w1[0], self.noPar1)
        [self.wm2, core2] = self.rewireMask(self.w2[0], self.noPar2)
        [self.wm3, core3] = self.rewireMask(self.w3[0], self.noPar3)

        # clear the original weights of the pruned weights
        self.w1[0] *= core1
        self.w2[0] *= core2
        self.w3[0] *= core3

    def append_training_metadata(self, metadata_file, epoch, sparsity_after_training, 
                                 sparsity_after_pruning, val_accuracy, val_loss):
        """
        output training metadata to a file
        """
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w') as f:
                f.write("Epoch\tVal_Accuracy\tVal_Loss\t")
                f.write("Sparsity_After_Training_L1\tSparsity_After_Training_L2\tSparsity_After_Training_L3\t")
                f.write("Sparsity_After_Pruning_L1\tSparsity_After_Pruning_L2\tSparsity_After_Pruning_L3\n")
        
        # write training metadata
        with open(metadata_file, 'a') as f:
            f.write(f"{epoch}\t{val_accuracy:.6f}\t{val_loss:.6f}\t")
            f.write(f"{sparsity_after_training['layer_1']:.6f}\t")
            f.write(f"{sparsity_after_training['layer_2']:.6f}\t")
            f.write(f"{sparsity_after_training['layer_3']:.6f}\t")
            f.write(f"{sparsity_after_pruning['layer_1']:.6f}\t")
            f.write(f"{sparsity_after_pruning['layer_2']:.6f}\t")
            f.write(f"{sparsity_after_pruning['layer_3']:.6f}\n")

    def train(self):
        [x_train, x_test, y_train, y_test] = self.read_data()
        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        
        snapshot_dir = "./SET-MLP-Keras-Weights-Mask/results/graph_snapshots"
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
            print(f"Created snapshot directory: {snapshot_dir}")
        
        metadata_file = "SET-MLP-Keras-Weights-Mask/results/training_metadata.txt"
        
        self.accuracies_per_epoch = []
        for epoch in range(self.maxepoches):
            print(f"\nEpoch {epoch+1}/{self.maxepoches}")
            
            # compile the model
            try:
                sgd = optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
            except TypeError:
                sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            
            # if there are saved weights, set the weights before training
            if self.w1 is not None:
                self.model.get_layer("sparse_1").set_weights(self.w1)
                self.model.get_layer("sparse_2").set_weights(self.w2)
                self.model.get_layer("sparse_3").set_weights(self.w3)
                if self.w4 is not None:
                    self.model.get_layer("dense_4").set_weights(self.w4)

            history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=self.batch_size),
                steps_per_epoch=len(x_train)//self.batch_size,
                epochs=1,
                validation_data=(x_test, y_test),
                verbose=1
            )
            
            # get validation accuracy and loss
            acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
            loss_key = 'val_loss' if 'val_loss' in history.history else 'loss'
            val_accuracy = history.history[acc_key][0]
            val_loss = history.history[loss_key][0]
            
            self.accuracies_per_epoch.append(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

            # save weights and masks after training & pruning
            sparsity_after_training = self.save_weights_and_masks(snapshot_dir, epoch, stage='after_training')
            
            sparsity_after_pruning = None
            if epoch < self.maxepoches - 1:
                self.weightsEvolution()
                
                weights_after_pruning = {
                    'layer_1': np.array(self.w1[0]),  
                    'layer_2': np.array(self.w2[0]),  
                    'layer_3': np.array(self.w3[0]),  
                    'layer_4': np.array(self.w4[0])   
                }
                masks_after_pruning = {
                    'layer_1': np.array(self.wm1),  
                    'layer_2': np.array(self.wm2),  
                    'layer_3': np.array(self.wm3)  
                }
                sparsity_after_pruning = self.save_weights_and_masks(snapshot_dir, epoch, stage='after_pruning', 
                                                                     weights=weights_after_pruning, masks=masks_after_pruning)
                
                K.clear_session()
                self.create_model()
            else:
                # because the last epoch doesn't have pruning, use the sparsity after training
                sparsity_after_pruning = sparsity_after_training
            
            # save training metadata
            self.append_training_metadata(metadata_file, epoch, sparsity_after_training, 
                                         sparsity_after_pruning, val_accuracy, val_loss)

    def read_data(self):
        #read CIFAR10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #normalize data
        xTrainMean = np.mean(x_train, axis=0)
        xTtrainStd = np.std(x_train, axis=0)
        x_train = (x_train - xTrainMean) / xTtrainStd
        x_test = (x_test - xTrainMean) / xTtrainStd

        return [x_train, x_test, y_train, y_test]

if __name__ == '__main__':
    if not os.path.exists('SET-MLP-Keras-Weights-Mask/results'): os.makedirs('SET-MLP-Keras-Weights-Mask/results')
    model = SET_MLP_CIFAR10()