# -*- coding: utf-8 -*-

# Imports 

import pandas as pd

try:
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers
    from keras import backend as K
except:
    print('keras not installed.')

from sklearn.model_selection import train_test_split
from .feature_selection import standardize, drop_single_value_columns

import matplotlib.pyplot as plt

# Autoencoder class

class autoencoder:
    """Class represents an implementation of a simple autoencoder.
    
    Class provides a framework for deminsionality reduction in a manner 
    compatible with the edbo.objective class.
    """
    
    def __init__(self, layers=[50, 25], activity_l1=[1e-7, 1e-7],
                 epochs=200, batch_size=100):
        """        
        Parameters
        ----------
        layers : list
            List of hidden layer sizes.
        activity_l1 : list
            List of activity regularizer l1 values.
        epochs : int
            Training epochs.
        batch_size : int
            Training batch sizes.
        
        Returns
        ----------
        None
        """
            
        self.layers = layers
        self.activity_l1 = activity_l1
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, obj, random_state=0, test_size=0.1):
        """Train the autoencoder to reconstruct the reaction space.
        
        Parameters
        ----------
        obj : edbo.objective:
            Initialized edbo.objective object.
        random_state : int
            Random seed for training/validation split.
        test_size : float
            Portion of data used as validation set.
        
        Returns
        ----------
        None
        """
        
        # Get data
        X = obj.domain
        y = [0 for y in range(len(X))]
        x_train, x_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=test_size, 
                                                            random_state=random_state)
        
        # Set regularizers
        regs_l1 = []
        for l1 in self.activity_l1:
            if l1 == None:
                regs_l1.append(None)
            else:
                regs_l1.append(regularizers.l1(l1))
    
        # Input layer
        x = Input(shape=(X.shape[1],))
    
        # Encode
        h = Dense(self.layers[0], activation='relu',
                  activity_regularizer=regs_l1[0],
                  )(x)
    
        for layer, reg in zip(self.layers[1:], regs_l1[1:]):
            h = Dense(layer, activation='relu',
                      activity_regularizer=reg,
                      )(h)
        
        # Decode
        reverse_layers = [self.layers[len(self.layers) - x - 1] for x in range(len(self.layers))]
        reverse_regs = [regs_l1[len(regs_l1) - x - 1] for x in range(len(regs_l1))]
    
        for layer, reg in zip(reverse_layers[1:], reverse_regs[1:]):
            h = Dense(layer, activation='relu',
                      activity_regularizer=reg,
                      )(h)

        # Output layer
        r = Dense(X.shape[1], activation='sigmoid')(h)

        autoencoder = Model(input=x, output=r)
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(x_train, x_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        verbose=0)
        
        self.model = autoencoder
        
    def plot_loss(self):
        """Plot the loss in reconstructing the validation set on each epoch.
        
        Returns
        ----------
        matplotlib.pyplot
            Plot of validation loss.
        """
        # Plot training & validation loss values
        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        return plt.show()
    
    def transform(self, obj):
        """Transform the encoded domain in edbo.objective object
        
        Parameters
        ----------
        obj : edbo.objective:
            Initialized edbo.objective object.
        
        Returns
        ----------
        None
        """
        
        get_layer_output = K.function([self.model.layers[0].input],
                                      [self.model.layers[len(self.layers)].output])
        
        columns = ['e' + str(i) for i in range(self.layers[-1])]
        embedding = pd.DataFrame(get_layer_output([obj.domain])[0], columns=columns)
        domain = drop_single_value_columns(standardize(embedding, scaler='minmax'))
        
        obj.domain = domain
        
        if len(obj.exindex) > 0:
            target = obj.exindex[obj.target].values
            obj.exindex = domain.copy()
            obj.exindex[obj.target] = target
