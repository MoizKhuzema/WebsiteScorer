from keras import layers
from keras import models
from keras import regularizers
from keras import initializers
from keras.layers import Layer
from keras import backend as K
import h5py
import cv2
import warnings
warnings.filterwarnings("ignore")


# Function to traverse weight file
def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path


# Function to predict web score
def predict(img):
    # Resize the image to the desired shape
    img = cv2.resize(img, (256, 192))
    # Ensure the image has 3 channels (RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape(1, 192, 256, 3)
    pred = model.predict(img)
    return (pred[0][0] / 6) * 9


# LRN (Local Response Normalization) layer.
class LRN(Layer):
    
    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LRN, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_data_format() == "th":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        half_n = self.n // 2
        squared = K.square(x)
        pooled = K.pool2d(squared, (half_n, half_n), strides=(1, 1),
                         padding="same", pool_mode="avg")
        if K.image_data_format() == "th":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom
    
    def get_output_shape_for(self, input_shape):
        return input_shape
    

# The architecture that will be used for the CNN is CaffeNet.
l = 0.001 
input_shape = (192, 256, 3)
im_data = layers.Input(shape=input_shape, dtype='float32', name='im_data')

conv1 = layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), name='conv1', 
                        activation='relu', input_shape=input_shape, 
                        kernel_regularizer=regularizers.l2(l))(im_data)

pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
norm1 = LRN(name="norm1")(pool1)
drop1 = layers.Dropout(0.1)(norm1)

layer1_1 = layers.Lambda(lambda x: x[:, :, :, :48])(drop1)
layer1_2 = layers.Lambda(lambda x: x[:, :, :, 48:])(drop1)

conv2_1 = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                        activation='relu',
                        padding='same', 
                        name='conv2_1', 
                        kernel_regularizer=regularizers.l2(l))(layer1_1)

conv2_2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                        activation='relu',
                        padding='same', 
                        name='conv2_2',
                        kernel_regularizer=regularizers.l2(l))(layer1_2)

conv2 = layers.Concatenate(name='conv_2')([conv2_1, conv2_2])

pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
norm2 = LRN(name="norm2")(pool2)
drop2 = layers.Dropout(0.1)(norm2)

conv3 = layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
                        name='conv3',
                        padding='same',
                        kernel_regularizer=regularizers.l2(l))(drop2)
drop3 = layers.Dropout(0.1)(conv3)

layer3_1 = layers.Lambda(lambda x: x[:, :, :, :192])(drop3)
layer3_2 = layers.Lambda(lambda x: x[:, :, :, 192:])(drop3)

conv4_1 = layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu', 
                        padding='same',
                        name='conv4_1',
                        kernel_regularizer=regularizers.l2(l))(layer3_1)

conv4_2 = layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu', 
                        padding='same',
                        name='conv4_2',
                        kernel_regularizer=regularizers.l2(l))(layer3_2)

conv4 = layers.Concatenate(name='conv_4')([conv4_1, conv4_2])

layer4_1 = layers.Lambda(lambda x: x[:, :, :, :192])(conv4)
layer4_2 = layers.Lambda(lambda x: x[:, :, :, 192:])(conv4)

conv5_1 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu',
                        padding='same', 
                        name='conv5_1',
                        kernel_regularizer=regularizers.l2(l))(layer4_1)

conv5_2 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu',
                        padding='same', 
                        name='conv5_2',
                        kernel_regularizer=regularizers.l2(l))(layer4_2)

conv5 = layers.Concatenate(name='conv_5')([conv5_1, conv5_2])

pool5 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv5)

flat = layers.Flatten()(pool5)
fc6 = layers.Dense(1024, activation='relu', name='fc6',
                        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                        bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(l))(flat)
drop6 = layers.Dropout(0.5)(fc6)

fc7 = layers.Dense(512, activation='relu', name='fc7', 
                        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                        bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(l))(drop6)
drop7 = layers.Dropout(0.5)(fc7)

fc8 = layers.Dense(1, name='fc8',
                        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                        bias_initializer='zeros')(drop7)

model = models.Model(inputs=im_data, outputs=fc8)


# Extract pretrained model weights
# Enter path of pretrained model weight
filename = r'C:\Users\moizk\Desktop\Upwork\Marcel-M\calista_rating_based.h5'

weights = {}
with h5py.File(filename, 'r') as f:
    for dset in traverse_datasets(filename):
        # print('Path:', dset)
        # print('Shape:', f[dset].shape)
        # print('Data type:', f[dset].dtype)
        if dset[:6] == '/model':
            weights[dset] = f[dset][:]

conv1_bias = weights['/model_weights/conv1/conv1/bias:0']
conv1_kernel = weights['/model_weights/conv1/conv1/kernel:0']
conv2_1_bias = weights['/model_weights/conv2_1/conv2_1/bias:0']
conv2_1_kernel = weights['/model_weights/conv2_1/conv2_1/kernel:0']
conv2_2_bias = weights['/model_weights/conv2_2/conv2_2/bias:0']
conv2_2_kernel = weights['/model_weights/conv2_2/conv2_2/kernel:0']
conv3_bias = weights['/model_weights/conv3/conv3/bias:0']
conv3_kernel = weights['/model_weights/conv3/conv3/kernel:0']
conv4_1_bias = weights['/model_weights/conv4_1/conv4_1/bias:0']
conv4_1_kernel = weights['/model_weights/conv4_1/conv4_1/kernel:0']
conv4_2_bias = weights['/model_weights/conv4_2/conv4_2/bias:0']
conv4_2_kernel = weights['/model_weights/conv4_2/conv4_2/kernel:0']
conv5_1_bias = weights['/model_weights/conv5_1/conv5_1/bias:0']
conv5_1_kernel = weights['/model_weights/conv5_1/conv5_1/kernel:0']
conv5_2_bias = weights['/model_weights/conv5_2/conv5_2/bias:0']
conv5_2_kernel = weights['/model_weights/conv5_2/conv5_2/kernel:0']
fc6_bias = weights['/model_weights/dense_1/dense_1/bias:0']
fc6_kernel = weights['/model_weights/dense_1/dense_1/kernel:0']
fc7_bias = weights['/model_weights/dense_2/dense_2/bias:0']
fc7_kernel = weights['/model_weights/dense_2/dense_2/kernel:0']
fc8_bias = weights['/model_weights/dense_3/dense_3/bias:0']
fc8_kernel = weights['/model_weights/dense_3/dense_3/kernel:0']

# Set pretrained model weights
model.get_layer('conv1').set_weights([conv1_kernel[:, :, :, :], conv1_bias[:]])
model.get_layer('conv2_1').set_weights([conv2_1_kernel[:, :, :, :], conv2_1_bias[:]])
model.get_layer('conv2_2').set_weights([conv2_2_kernel[:, :, :, :], conv2_2_bias[:]])
model.get_layer('conv3').set_weights([conv3_kernel[:, :, :, :], conv3_bias[:]])
model.get_layer('conv4_1').set_weights([conv4_1_kernel[:, :, :, :], conv4_1_bias[:]])
model.get_layer('conv4_2').set_weights([conv4_2_kernel[:, :, :, :], conv4_2_bias[:]])
model.get_layer('conv5_1').set_weights([conv5_1_kernel[:, :, :, :], conv5_1_bias[:]])
model.get_layer('conv5_2').set_weights([conv5_2_kernel[:, :, :, :], conv5_2_bias[:]])
model.get_layer('fc6').set_weights([fc6_kernel[:], fc6_bias[:]])
model.get_layer('fc7').set_weights([fc7_kernel[:], fc7_bias[:]])
model.get_layer('fc8').set_weights([fc8_kernel[:], fc8_bias[:]])


# image_path = r'C:\Users\moizk\Documents\Capture.png'
# img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# print(f"result: {predict(img)}")