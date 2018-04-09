import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from time import time, gmtime, strftime
from model import ResNeXt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR dataset')
    # Dataset arguments
	parser.add_argument('--data_path', '-dp', type=str, default='cifar10', help='Root for the dataset')
	parser.add_argument('--dataset', '-ds', type=str, default='cifar10', help='Dataset type')
    # Optimization options
	parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of epochs to train')
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Learning rate')
	parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay for L2 loss')
	parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum')
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
	parser.add_argument('--save', '-s', type=str, default='./snapshot', help='Folder to save model checkpoints')
	parser.add_argument('--log', type=str, default='./log', help='Log folder')

	args = parser.parse_args()

	timestamp = strftime('%d-%m-%Y_%H-%M-%S', gmtime())

	log_loc = 'log-' + timestamp + '.txt'
	model_loc = 'model-' + timestamp + '.pytorch'
	train_data_dir = args.data_path + '/train'
	validation_data_dir = args.data_path + '/validation'

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
	nlabels = 10

	train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
	test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=args.test_bs)

	if not os.path.isdir(args.save):
		os.makedirs(args.save)

	net = ResNeXt(args.cardinality, args.depth, args.base_width, nlabels, args.widen_factor)
	print(net)

	optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

	# Training
	def train():
		net.train()
		loss_avg = 0.0
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

			output = net(data)

			optimizer.zero_grad()
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()

			loss_avg = loss_avg * 0.2 + loss.data[0] * 0.8
		state['train_loss'] = loss_avg

	# Testing
	def test():
		net.eval()
		loss_avg = 0.0
		correct = 0
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)

			output = net(data)
			loss = F.cross_entropy(output, target)

			pred = output.data.max(1)[1]
			correct += pred.eq(target.data).sum()

			loss_avg += loss.data[0]

		state['test_loss'] = loss_avg / len(test_loader)
		state['test_accuracy'] = correct / len(test_loader.dataset)

	best_accuracy = 0.0
	for epoch in range(args.epoch):
		if epoch in args.schedule:
			state['learning_rate'] *= args.gamma
			for param_group in optimizer.param_groups:
				param_group['lr'] = state['learning_rate']

		state['epoch'] = epoch
		train()
		test()
		if state['test_accuracy'] > best_accuracy:
			best_accuracy = state['test_accuracy']
			torch.save(net.state_dict(), os.path.join(args.save, 'model.pytorch'))
		log.write('%s\n' % json.dumps(state))
		log.flush()
		print(state)
		print('Best Accuracy: %f' % best_accuracy)

	log.close()
