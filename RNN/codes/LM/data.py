from __future__ import unicode_literals, print_function, division
from glob import glob
from tqdm import tqdm
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import RNN
from sklearn.model_selection import train_test_split

# path 의 패턴과 일치하는 모든 파일과 디렉터리의 리스트 반환( ex *.txt)
def findFiles(path): return glob(path)


import unicodedata
import string

# ascii 코드로 나타낼 수 있는 모든 소/대문자 글자들 (특스문자 제외)
# string은 이처럼 whitespace, 구두점(puctuation), hexdigits 등의 문자들을 문자열로 제공한다.
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
	return ''.join(
			c for c in unicodedata.normalize('NFD', s)
			if unicodedata.category(c) != '\n'
			and c in all_letters
			)

category_lines = {}
all_categories = []
category_types = []

data_names = []
data_categories =[]
n_categories = 0

def readLines(filename):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]

for filename in findFiles('../data/names/*.txt'):
	# os.paht.splittext() : 경로명과 확장자를 나누어서 리스트로 제공
	category = os.path.splitext(os.path.basename(filename))[0]
	lines = readLines(filename)
	for line in lines:
		data_names.append(line)
		data_categories.append(category)
	category_types.append(category)
	n_categories += 1





def convert2vector(name):
	data = torch.zeros((len(name), len(all_letters)))
	for t, word in zip(data, name):
		t[all_letters.find(name)] = 1
	return data
	
def letter2Index(letter):
	return all_letters.find(letter)

def letter2Tensor(letter):
	data = torch.zeros(1, n_letters)
	data[0][letter2Index(letter)] = 1
	return data

def line2Tensor(line):
	data = torch.zeros(len(line), 1, n_letters)
	for t, word in zip(data, line):
		t[0][letter2Index(word)] = 1
	return data

def convert2Tensor(data):
	ret = list()
	for name in data:
		ret.append(convert2vector(name))
	return ret

def convert2Label(data):
	ret = list()

	for country in data:
		ret.append(category_types.index(country))
	return torch.tensor(ret)

data_X = convert2Tensor(data_names)
data_y = convert2Label(data_categories)
	

n_hidden = 128
print(n_letters)
print(n_hidden)
print(n_categories)
rnn = RNN(n_letters, n_hidden, n_categories)


def categoryFromOutput(output):
	# 텐서에서 가장 큰 값과 주소 반환(index)
	top_n, top_i = output.topk(1)
	category_i = top_i[0].item()
	return all_categories[category_i], category_i

def randomChoice(l):
	return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
	line_tensor = line2Tensor(line)
	return category, line, category_tensor, line_tensor



X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, stratify=data_y)


criterion = F.cross_entropy


epochs = 10
lr = 1e-3
optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)

use_optim = True
#use_optim = False

def train(epoch):
	rnn.train()

	loss_mean = 0
	losses = 0
	correct = 0
	acc = 0

	pbar = tqdm(enumerate(zip(X_train, y_train)), total=len(X_train), desc=f'{epoch} epoch...')
	
	for i, (name, y) in pbar:
		hidden = rnn.initHidden()
		for word in name:
			output, hidden = rnn(word, hidden)
		y = y.expand(1,)

		loss = criterion(output, y)
		if use_optim:
			optimizer.zero_grad()
		else:
			rnn.zero_grad()
		losses += loss.item()
		loss.backward()
		loss_mean = losses / (i+1)
		if output.argmax(axis=1) == y.item():
			correct += 1
		acc = correct / (i+1)
	

		if use_optim:
			optimizer.step()
		else:
			with torch.no_grad():
				for p in rnn.parameters():
					new_val = p - lr * p.grad
					p.copy_(new_val)


		'''
		for p in rnn.parameters():
			p.data.add_(p.grad.data, alpha=-lr)
		'''
	
		pbar.set_postfix_str(f'Loss : {loss_mean:.2f}, acc : {acc:.2f}')



	'''
	for p in rnn.parameters():
		p.data.add_(p.grad.data, alpha=-lr)
	'''

def eval(epoch):
	hidden = rnn.initHidden()
	rnn.eval()
	matches = []
	losses = 0
	with torch.no_grad():
		pbar = tqdm(enumerate(zip(X_test, y_test)), total=len(X_test), desc=f'{epoch} epoch...')

		for i, (name, y) in pbar:
			hidden = rnn.initHidden()
			for word in name:
				output, hidden = rnn(word, hidden)
			y = y.expand(1,)
			loss = criterion(output, y)
			prob, idx = output.max(dim=1)
			loss_val = loss.item()
			losses += loss_val
			loss_mean = losses / (i+1)

			match = torch.eq(y, idx).detach()
			matches.extend(match.cpu())
			accuracy = np.sum(matches) / len(matches) if 0 < len(matches) else 0
			pbar.set_postfix_str(f'Loss : {loss_mean:.2f}, acc : {accuracy:.2f}')

			



for epoch in range(epochs):
	train(epoch)
	eval(epoch)
