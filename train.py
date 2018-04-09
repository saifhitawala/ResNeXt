import argparse
import os
import json
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from time import time, gmtime, strftime

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR dataset')
    # Dataset arguments
	parser.add_argument('--data_path', '-dp', type=str, default='cifar10', help='Root for the dataset')
	parser.add_argument('--dataset', '-ds', type=str, default='cifar10', help='Dataset type')
    # Optimization options
	parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of epochs to train')
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Learning rate')
	parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay for L2 loss')
	parser.add_argument('--momemtum', '-m', type=float, default=0.9, help='Momentum')
	parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Schedule to decrease learning rate at given epochs')
	parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate is multiplied by gamma on schedule')
	parser.add_argument('--batch_size', '-b', type=int, default=128, help='Training batch size')
	parser.add_argument('--test_bs', type=float, default=10, help='Test batch size')
    # Architecture options
	parser.add_argument('--depth', type=int, default=29, help='Depth of the model')
	parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality or group')
	parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group')
	parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
	# Checkpoint options
	parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save model checkpoints')
	parser.add_argument('--log', type=str, default='./', help='Log folder')

	args = parser.parse_args()
	print(args)

	timestamp = strftime('%d-%m-%Y_%H-%M-%S', gmtime())
	print(timestamp)

	log_loc = 'log-' + timestamp + '.txt'
	model_loc = 'model-' + timestamp + '.pytorch'
	train_data_dir = args.data_path + '/train'
	validation_data_dir = args.data_path + '/validation'

	print(log_loc)
	print(model_loc)
	print(train_data_dir)
	print(validation_data_dir)
 
	if not os.path.isdir(args.log):
		os.makedirs(args.log)
	log = open(os.path.join(args.log, log_loc), 'w')
	state = {k : v for k, v in args._get_kwargs()}
	print(state)
	log.write(json.dumps(state) + '\n')

	if not os.path.isdir(args.data_path):
		os.makedirs(args.data_path)

	train_data = dset.ImageFolder(train_data_dir, transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()]))
	test_data = dset.ImageFolder(validation_data_dir, transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor()]))
	print(train_data.__getitem__(0))
	print(test_data.__getitem__(0))
	nlabels = 10

	train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
	test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=args.test_bs)

	if not os.path.isdir(args.save):
		os.makedirs(args.save)

	