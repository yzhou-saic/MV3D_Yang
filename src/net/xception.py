import keras.applications.xception as xcep
from keras.utils import plot_model
from keras.layers import Conv2D
# from keras.models import Model






if __name__ == '__main__':
    # xception_model = xcep.Xception(include_top=False, input_shape = (299, 299, 3), weights='imagenet')
    # print(xception_model.get_layer(name='add_11').output)
    # print(xception_model.summary())
    # plot_model(xception_model, to_file='model.png')
    xception_model = xcep.Xception(include_top=False,  weights='imagenet')
    print(xception_model.layers[1].name)
    xception_model.layers[1] = Conv2D(32, [3, 3])
    print(xception_model.get_layer(name='add_11').name)
    plot_model(xception_model, to_file='model.png')
