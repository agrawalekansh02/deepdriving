from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Reshape, ReLU


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_duim = latent_dim
        self.encoder = Sequential([
            Flatten(),
             
            Dense(units=latent_dim)
        ])
        self.decoder = Sequential([
            Input(shape=(latent_dim,)),
            # Reshape()
            Conv2DTranspose(filters=128, kernel_size=3, strides=2, 
                    padding='same', activation='relu'),
            Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
                    padding='same', activation='relu'),
            Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
                    padding='same', activation='relu')
        #     Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')
        ])

        # self.encoder = Sequential([
        #     Input(shape=(img_shape), name='input_image'),
        #     Conv2D(filters=32, kernel_size=3, strides=2,
    
    def call(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return encoded
