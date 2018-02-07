import keras.applications.xception as xcep
from keras.utils import plot_model
# from keras.models import Model






if __name__ == '__main__':
    xception_model = xcep.Xception(include_top=False, input_shape = (299, 299, 3), weights='imagenet')
    print(xception_model.get_layer(name='add_11').output)
    # print(xception_model.summary())
    # plot_model(xception_model, to_file='model.png')