# example of defining the generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

# define the size of the latent space
latent_dim = 100
# define the generator model
model = define_generator(latent_dim)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

# Program:
#    1. Zdefiniowanie modelu generatora
#    2. Wyprintowanie informacji o modelu
#    3. Zapisanie modelu do pliku graficznego (pokazanie struktur modelu)

#     Struktura modelu (warstwy):
#     1. Dense (n_nodes, input_dim=latent_dim) - wykonuje przetworzenie danych wejściowych
#	  2. LeakyReLU (alpha=0.2) - wprowadza nieliniowość do modelu (pomaga w uczeniu się)
#	  3. Reshape (7, 7, 128) - zmienia kształt danych na 3D (7x7x128)
#	  4. Conv2DTranspose (128, (4,4), strides=(2,2), padding='same') - rozciąga obraz
#	  5. LeakyReLU (alpha=0.2) 
#	  6. Conv2D (1, (7,7), activation='sigmoid', padding='same') - generuje obraz 28x28x1 (czarno-biały)
