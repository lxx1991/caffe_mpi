__author__ = 'alex'

from log_loader import *
from progress_visulization import *
from matplotlib import animation

class animate_monitor():

    def __init__(self, log_files, fig, training_loss_id=3, testing_loss_id=2):
        """
        Initialize and record log list
        :param log_files:
        :return:
        """
        self.log_files = log_files
        self.fig = fig
        self.axes = []
        self.training_loss_id=training_loss_id
        self.testing_loss_id = testing_loss_id

    def start(self):
        for i in xrange(1,3):
            self.axes.append(self.fig.add_subplot(2,1,i, sharex=self.axes[0] if i>1 else None))

        return self.show_value()

    def show_value(self, fd=None):
        """
        Internal function to read updated progress
        :return:
        """
        numbers = select_log_part(load_log(self.log_files),[('Testing','Testing', 8, 2), ('Training','loss', 4, 0)])
        y0, y1 = draw_loss(numbers, self.axes[0], self.training_loss_id, self.testing_loss_id)
        y2 = draw_acc(numbers, self.axes[1])
        # return self.fig

if __name__ == "__main__":
    log_files = [
        # '../models/googlenet/log/oct10_0_20000.22550',
        # '../models/googlenet/log/oct10_20000_35000.28025',
        # '../models/googlenet/log/oct11_35000_70000.5614',
        # '../models/googlenet/log/oct11_70000_85000.26819',
        # '../models/googlenet/log/oct12_85000_105000.24475',
        '../models/googlenet/log/caffe.mmlab-107.alex.log.INFO.20141014-110026.14019'
        # '/media/ssd/backup/caffe.pvg-gpu-desktop.zhirongw.log.INFO.20141012-131959.4493'
        # '/media/ssd/backup/googlenet_2510_slowlr.log'
    ]

    fig = pyplot.figure(num=1, figsize=(15,9))

    # am = animate_monitor(log_files, fig, testing_loss_id=0, training_loss_id=1)
    am = animate_monitor(log_files, fig)

    anime = animation.FuncAnimation(fig, am.show_value, blit=False, interval=5000, init_func=am.start)

    pyplot.show()



