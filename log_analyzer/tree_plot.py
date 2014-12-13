__author__ = 'alex'

from matplotlib import pyplot


def get_value_coord(diction, value_idx=0):

    coord = []
    value = []
    for k in sorted(diction):
        if len(diction[k])<= value_idx:
            continue
        coord.append(k)
        value.append(diction[k][value_idx])

    return coord,value

def draw_loss(numbers, axe=None):
    """
    The function for drawing loss curve
    :param numbers:
    :return:
    """

    training_loss = numbers['Training']['loss']
    testing_loss = numbers['Testing']['loss']

    training_loss_iter, training_loss_total_value = get_value_coord(training_loss, 0)
    training_loss_iter, training_loss_value1 = get_value_coord(training_loss, 1)
    training_loss_iter, training_loss_value2 = get_value_coord(training_loss, 2)
    testing_loss_iter, testing_loss_value1 = get_value_coord(testing_loss, 0)
    testing_loss_iter, testing_loss_value2 = get_value_coord(testing_loss, 1)
    testing_loss_total_value = [x+3*y for x,y in zip(testing_loss_value1, testing_loss_value2)]

    if axe is None:
        axe = pyplot
    return axe.plot(training_loss_iter, training_loss_value1), axe.plot(testing_loss_iter, testing_loss_value1)

def draw_acc(numbers, axe=None):
    """
    Draw testing accuracy curve
    :param numbers:
    :return:
    """

    testing_acc = numbers['Testing']['accuracy']

    testing_acc_iter, testing_acc_value1 = get_value_coord(testing_acc,0)
    testing_acc_iter, testing_acc_value2 = get_value_coord(testing_acc,1)
    testing_acc_second = [float(y)/float(x) for x,y in zip(testing_acc_value1, testing_acc_value2)]

    if axe is None:
        axe = pyplot
    return axe.plot(testing_acc_iter, testing_acc_value1), axe.plot(testing_acc_iter, testing_acc_value2), axe.plot(testing_acc_iter, testing_acc_second)

    # pyplot()


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
        '/home/zhirongw/logs/tree2/tree2_138_1026'
    ]

    file_contents = load_log(log_files)

    numbers = select_log_part(file_contents, part_identifiers=[('Testing','Testing',3,2),('Training','loss',3,0)])


    draw_both(numbers)
    # draw_loss(numbers)
    # draw_acc(numbers)
