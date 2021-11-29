import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type = int, default = '0')
args = parser.parse_args()

dataset_list = ["CIFAR10", "CIFAR100", "DIGITS"]

prev_ds = None
for ds in dataset_list:
	if not prev_ds is None:
		if not os.path.isfile(os.path.join(prev_ds, 'status_{}.txt'.format(ds))):
			print("\n\n{} did not complete successfully".format(prev_ds))
			print("Breaking Operation", flush = True)
			break
	os.system("rm status_{}.txt 2> /dev/null".format(ds))
	print("Now Running {}".format(ds), flush = True)
	os.system('python3 -Wignore main.py --dataset {} --cuda {} && python3 -Wignore main.py --dataset {} --cuda {} --resume --eval > results.txt'.format(ds, args.cuda, ds, args.cuda))
	prev_ds = ds