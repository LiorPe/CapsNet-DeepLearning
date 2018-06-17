import numpy as np
from data.load_data import load_mnist
from structure.caps_net import CapsNet
from keras import callbacks
import numpy as np
from keras import  optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image

# default_routing = 3
# learning_rate = 0.001
# learning_rate_decay = 0.9
# loss_decoder_coeff = 0.392
# batch_size = 100
# epochs = 1

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data ):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=save_dir + '/tensorboard-logs',
                               batch_size=batch_size, histogram_freq=int(debug))
    checkpoint = callbacks.ModelCheckpoint(save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay_scheduler = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * ( lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, batch_size, shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay_scheduler])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    print('1')
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('2')

    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, digit):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(save_dir + '/manipulate-%d.png' % digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (save_dir, digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    # parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    # parser.add_argument('--epochs', default=1, type=int)
    epochs = 1
    # parser.add_argument('--batch_size', default=100, type=int)
    batch_size = 100
    # parser.add_argument('--lr', default=0.001, type=float,
    #                     help="Initial learning rate")
    lr = 0.001
    # parser.add_argument('--lr_decay', default=0.9, type=float,
    #                     help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    lr_decay = 0.9
    # parser.add_argument('--lam_recon', default=0.392, type=float,
    #                     help="The coefficient for the loss of decoder")
    lam_recon = 0.392
    # parser.add_argument('-r', '--routings', default=3, type=int,
    #                     help="Number of iterations used in routing algorithm. should > 0")
    routings = 3
    # parser.add_argument('--shift_fraction', default=0.1, type=float,
    #                     help="Fraction of pixels to shift at most in each direction.")
    shift_fraction = 0.1
    # parser.add_argument('--debug', action='store_true',
    #                     help="Save weights by TensorBoard")
    debug = True
    # parser.add_argument('--save_dir', default='./result')
    save_dir = './result'
    # parser.add_argument('-t', '--testing', action='store_true',
    #                     help="Test the trained model on testing dataset")
    t = True
    # parser.add_argument('--digit', default=5, type=int,
    #                     help="Digit to manipulate")
    digit = 5
    # parser.add_argument('-w', '--weights', default=None,
    #                     help="The path of the saved weights. Should be specified when testing")
    weights = None
    # args = parser.parse_args()
    # print(args)
    testing = False

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=routings)
    model.summary()

    # train or test
    if weights is not None:  # init the model weights with provided one
        model.load_weights(weights)
    if not testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)))
    else:  # as long as weights are given, will run testing
        if weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), digit)
        test(model=eval_model, data=(x_test, y_test))


