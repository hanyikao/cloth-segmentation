import argparse
import os

class parser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training Options")
        self.initialize_arguments()
        self.args = self.parser.parse_args()

        if self.args.logs_dir == "":
            self.args.logs_dir = os.path.join('logs', self.args.name)

        if self.args.save_dir == "":
            self.args.save_dir = os.path.join('results', self.args.name)
        

    def initialize_arguments(self):
        self.parser.add_argument('--name', type=str, default="training_cloth_segm_u2net_exp6", 
                                 help='Experiment name')
        self.parser.add_argument('--image_folder', type=str, default="../imaterialist/taelor/", 
                                 help='Image folder path')
        self.parser.add_argument('--df_path', type=str, default="./taelor.csv", 
                                 help='Label CSV path')
        self.parser.add_argument('--distributed', action='store_true', 
                                 help='Use multi-GPU training')
        
        self.parser.add_argument('--fine_width', type=int, default=192*4,
                                 help='Fine width')
        self.parser.add_argument('--fine_height', type=int, default=192*4,
                                 help='Fine height')

        self.parser.add_argument('--mean', type=float, default=0.5,
                                 help='Mean for normalization')
        self.parser.add_argument('--std', type=float, default=0.5,
                                 help='Standard deviation for normalization')

        self.parser.add_argument('--batchSize', type=int, default=4, 
                                 help='Batch size')
        self.parser.add_argument('--nThreads', type=int, default=4, 
                                 help='Number of threads for data loading')
        self.parser.add_argument('--max_dataset_size', type=float, default=float("inf"),
                                 help='Maximum dataset size')

        self.parser.add_argument('--serial_batches', action='store_true', default=False, 
                                 help='Use serial batches instead of random')
        self.parser.add_argument('--continue_train', action='store_true', default=True, 
                                 help='Continue training from the last checkpoint')
        self.parser.add_argument('--unet_checkpoint', type=str, 
                                 default="prev_checkpoints/cloth_segm_unet_surgery.pth",
                                 help='Checkpoint file for U-Net')

        self.parser.add_argument('--save_freq', type=int, default=1000,
                                 help='Frequency of saving the model')
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='Frequency of printing training progress')
        self.parser.add_argument('--image_log_freq', type=int, default=100,
                                 help='Frequency of logging images')

        self.parser.add_argument('--iter', type=int, default=100000,
                                 help='Number of iterations for training')
        self.parser.add_argument('--lr', type=float, default=0.0001,
                                 help='Learning rate')
        self.parser.add_argument('--clip_grad', type=float, default=5,
                                 help='Gradient clipping threshold')
        
        self.parser.add_argument('--logs_dir', type=str, default="", 
                                 help='Directory for logs')
        self.parser.add_argument('--save_dir', type=str, default="", 
                                 help='Directory for results')
        self.parser.add_argument('--test', action='store_true', default=False, 
                                help='')


    def get_options(self):
        return self.args
