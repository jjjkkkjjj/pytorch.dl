from glob import glob
from datetime import date, datetime
import os, logging, re, torch, shutil
import matplotlib.pyplot as plt

class SaveManager(object):
    def __init__(self, modelname, interval, max_checkpoints, plot_yrange=(0, 14), plot_interval=10,
                 weightsdir=os.path.join(os.path.expanduser("~"), 'weights')):
        """
        :param modelname: str, saved model name.
        :param interval: int, save for each designated iteration
        :param max_checkpoints: (Optional) int, how many dl will be saved during training.
        """
        logging.getLogger().setLevel(level=logging.INFO)
        if max_checkpoints > 15:
            logging.warning('One model size will be about 0.1 GB. Please take care your storage.')
        save_checkpoints_dir = os.path.join(weightsdir, modelname, 'checkpoints')
        today = '{:%Y%m%d}'.format(date.today())

        """
        # check existing checkpoints file
        filepaths = sorted(
            glob(os.path.join(save_checkpoints_dir, modelname + '_i[-]*_checkpoints{}.pth'.format(today))))
        if len(filepaths) > 0:
            logging.warning('Today\'s checkpoints is remaining. Remove them?\nInput any key. [n]/y')
            i = input()
            if re.match(r'y|yes', i, flags=re.IGNORECASE):
                for file in filepaths:
                    os.remove(file)
                logging.warning('Removed {}'.format(filepaths))
            else:
                logging.warning('Please rename them.')
                exit()
        """
        savedir = os.path.join(weightsdir, modelname)
        if os.path.exists(savedir):
            dirname = modelname + datetime.now().strftime('-%Y%m%d-%H:%M:%S')

            prev_savedir = savedir
            savedir = os.path.join(weightsdir, dirname)
            save_checkpoints_dir = os.path.join(weightsdir, dirname, 'checkpoints')
            logging.warning('{} has already existed. Create {} instead? [y]/n'.format(prev_savedir, savedir))
            i = input()
            if re.match(r'n|no', i, flags=re.IGNORECASE):
                logging.warning('Please rename them.')
                exit()


        os.makedirs(savedir)
        logging.info('Created directory: {}'.format(savedir))
        os.makedirs(save_checkpoints_dir)
        logging.info('Created directory: {}'.format(save_checkpoints_dir))

        self.savedir = savedir
        self.save_checkpoints_dir = save_checkpoints_dir
        self.modelname = modelname
        self.today = today
        self.interval = interval
        self.plot_yrange = plot_yrange
        self.plot_interval = plot_interval

        self.max_checkpoints = max_checkpoints



    def update_iteration(self, model, now_iteration, max_iterations):
        return self._update(model, now_iteration, max_iterations, 'iteration')

    def update_epoch(self, model, now_epoch, max_epochs):
        return self._update(model, now_epoch, max_epochs, 'epoch')

    def _update(self, model, now, maximum_number, mode):
        saved_path = ''
        removed_path = ''

        if mode == 'epoch':
            removed_checkpoints_regex_filename = 'e[-]*_checkpoints{}.pth'.format(self.today)
            created_checkpoints_filename = 'e-{:07d}_checkpoints{}.pth'.format(now, self.today)
        elif mode == 'iteration':
            removed_checkpoints_regex_filename = 'i[-]*_checkpoints{}.pth'.format(self.today)
            created_checkpoints_filename = 'i-{:07d}_checkpoints{}.pth'.format(now, self.today)
        else:
            raise ValueError()

        if now % self.interval == 0 and now != maximum_number:
            filepaths = sorted(
                glob(os.path.join(self.save_checkpoints_dir, removed_checkpoints_regex_filename)))

            # remove oldest checkpoints
            if len(filepaths) > self.max_checkpoints - 1:
                removed_path += os.path.basename(filepaths[0])
                os.remove(filepaths[0])

            # save model
            saved_path = os.path.join(self.save_checkpoints_dir, created_checkpoints_filename)
            torch.save(model.state_dict(), saved_path)


        return saved_path, removed_path

    def finish(self, model, optimizer, scheduler, mode, x, names, losses_dict):
        if mode == 'epoch':
            model_filename = self.modelname + '_model_e-{}.pth'.format(x[-1])
            optimizer_filename = self.modelname + '_optimizer_e-{}.pth'.format(x[-1])
            scheduler_filename = self.modelname + '_scheduler_e-{}.pth'.format(x[-1])
            graph_filename = self.modelname + '_learning-curve_e-{}.png'.format(x[-1])

        elif mode == 'iteration':
            model_filename = self.modelname + '_model_i-{}.pth'.format(x[-1])
            optimizer_filename = self.modelname + '_optimizer_i-{}.pth'.format(x[-1])
            scheduler_filename = self.modelname + '_scheduler_i-{}.pth'.format(x[-1])
            graph_filename = self.modelname + '_learning-curve_i-{}.png'.format(x[-1])

        else:
            raise ValueError()

        print()

        # model
        savepath = os.path.join(self.savedir, model_filename)
        torch.save(model.state_dict(), savepath)
        logging.info('Saved model to {}'.format(savepath))

        # optimizer
        savepath = os.path.join(self.savedir, optimizer_filename)
        torch.save(optimizer.state_dict(), savepath)
        logging.info('Saved optimizer to {}'.format(savepath))

        # scheduler
        if scheduler:
            savepath = os.path.join(self.savedir, scheduler_filename)
            torch.save(model.state_dict(), savepath)
            logging.info('Saved scheduler to {}'.format(savepath))

        # graph
        savepath = os.path.join(self.savedir, graph_filename)
        # initialise the graph and settings
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        ax.clear()
        # plot
        for label in names:
            ax.plot(x, losses_dict[label], label=label)
        ax.legend()

        if self.plot_yrange:
            ax.axis(ymin=self.plot_yrange[0], ymax=self.plot_yrange[1])

        ax.set_title('Learning curve')
        ax.set_xlabel(mode)
        ax.set_ylabel('loss')
        # ax.axis(xmin=1, xmax=iterations)
        # save
        fig.savefig(savepath)

        print('Saved graph to {}'.format(savepath))