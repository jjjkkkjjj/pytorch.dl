import matplotlib.pyplot as plt


class LiveGraph(object):
    def __init__(self, yrange=(0, 14)):
        self.yrange = yrange

        # initialise the graph and settings
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()

        self.fig.show()
        self.fig.canvas.draw()

        self.x = []
        self.losses = {}
        self.names = []
        self.main_name = ''

    def initialize(self, names):
        self.names = names
        for name in names:
            self.losses[name] = []
        self.main_name = names[0]

    @property
    def isInitialized(self):
        return len(self.names) > 0

    def update(self, epoch, iteration, x, names, losses_dict):
        if not self.isInitialized:
            self.initialize(names)

        self.x = x

        self.ax.clear()
        # plot
        for name in names:
            if not name in self.losses.keys():
                raise KeyError('must pass {} in initialize method'.format(name))
            self.losses[name] = losses_dict[name]
            self.ax.plot(self.x, self.losses[name], label=name)

        self.ax.legend()

        # self.ax.axis(xmin=0, xmax=iterations) # too small to see!!
        if self.yrange:
            self.ax.axis(ymin=self.yrange[0], ymax=self.yrange[1])

        self.ax.set_title('Learning curve\nEpoch: {}, Iteration: {}, Loss: {}'.format(epoch, iteration,
                                                                                     self.losses[self.main_name][-1]))

        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('loss')
        # update
        self.fig.canvas.draw()

    def update_info(self, info):
        if self.fig is None or self.ax is None:
            raise NotImplementedError('Call initialize first!')

        self.ax.clear()
        # plot
        for name, losses in self.losses.items():
            self.ax.plot(self.x, losses, label=name)

        self.ax.legend()

        # self.ax.axis(xmin=0, xmax=iterations) # too small to see!!
        if self.yrange:
            self.ax.axis(ymin=self.yrange[0], ymax=self.yrange[1])

        self.ax.set_title('Learning curve\n{}'.format(info))
                          #fontsize=self.fontsize)

        self.ax.set_xlabel('iteration')
        self.ax.set_ylabel('loss')
        # update
        self.fig.canvas.draw()