__author__ = 'alex'

from matplotlib import pyplot


def get_value_coord(diction, value_idx=0):

    coord = []
    value = []
    for k in sorted(diction):
        if value_idx >= len(diction[k]):
            continue
        coord.append(k)
        value.append(diction[k][value_idx])

    return coord,value

def draw_loss(numbers, axe=None, training_loss_id=3, testing_loss_id=2):
    """
    The function for drawing loss curve
    :param numbers:
    :return:
    """

    training_loss = numbers['Training']['loss']
    testing_loss = numbers['Testing']['loss']

    training_loss_iter, training_loss_total_value = get_value_coord(training_loss, training_loss_id)
    testing_loss_iter, testing_loss_total_value = get_value_coord(testing_loss, testing_loss_id)

    if axe is None:
        axe = pyplot

    axe.cla()
    axe.grid(True)
    axe.plot(training_loss_iter, training_loss_total_value), axe.plot(testing_loss_iter, testing_loss_total_value)
    axe.text(0.5,.7, 'Latest iteration: {:}'.format(training_loss_iter[-1]), transform=axe.transAxes)
    axe.text(0.5,.9, 'Latest Testing Loss: {:}'.format(testing_loss_total_value[-1]), transform=axe.transAxes)
    axe.text(0.5,0.85, 'Latest 5 Training Loss: {:}, {:}, {:}, {:}, {:}'.format(*training_loss_total_value[-5:]), transform=axe.transAxes)
    return axe, axe, training_loss_total_value[-1], testing_loss_total_value[-1]

def draw_acc(numbers, axe=None, acc_id=0, acc_5_id=None):
    """
    Draw testing accuracy curve
    :param numbers:
    :return:
    """

    testing_acc = numbers['Testing']['accuracy']

    testing_acc_iter, testing_acc_value = get_value_coord(testing_acc,acc_id)


    if axe is None:
        axe = pyplot

    axe.cla()
    axe.grid(True)
    axe.text(0.3,.9, 'Latest Testing Accuracy: {:}'.format(testing_acc_value[-1]), transform=axe.transAxes)
    axe.text(0.3,.7, 'Latest Testing Iteration: {:}'.format(testing_acc_iter[-1]), transform=axe.transAxes)
    
    if acc_5_id is not None:

        testing_acc_5_iter, testing_acc_5_value = get_value_coord(testing_acc, int(acc_5_id))

        axe.plot(testing_acc_5_iter, testing_acc_5_value)
        axe.text(0.3,.8, 'Latest Testing Top-5 Accuracy: {:}'.format(testing_acc_5_value[-1]), transform=axe.transAxes)

    return axe.plot(testing_acc_iter, testing_acc_value), testing_acc_value[-1]

def draw_both(numbers):
    fig = pyplot.figure(num=1, figsize=(15,9))

    ax = fig.add_subplot(2, 1, 1)
    draw_loss(numbers, ax)
    ax = fig.add_subplot(2, 1, 2)
    draw_acc(numbers, ax)
    pyplot.show()


if __name__ == '__main__':

    from log_loader import *

    log_files = [
        '../models/googlenet/log/oct10_0_20000.22550',
        '../models/googlenet/log/oct10_20000_35000.28025',
        '../models/googlenet/log/oct11_35000_70000.5614',
        '../models/googlenet/log/oct11_70000_85000.26819',
        # '../models/googlenet/log/oct11_85000_100000.2021',
        '../models/googlenet/log/caffe.mmlab-107.alex.log.INFO.20141012-195207.1957'
    ]

    file_contents = load_log(log_files)

    numbers = select_log_part(file_contents)


    draw_both(numbers)
    # draw_loss(numbers)
    # draw_acc(numbers)
