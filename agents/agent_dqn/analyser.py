#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, getopt
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import os


IDX_EPISODE_NB = 1
IDX_EPISODE_SCORE = 3
IDX_STEPS_NB = 7
IDX_GSTEPS_NB = 10
IDX_LOSS = 13
IDX_MINX = 16
IDX_X = 17
IDX_MAXX = 18
IDX_SCORE_TOTAL = 20


IDX_NAME = {'score': IDX_EPISODE_SCORE, 'loss': IDX_LOSS, 'steps': IDX_STEPS_NB, 'varx': IDX_X, 'minx': IDX_MINX, 'maxx':IDX_MAXX, 'score_total':IDX_SCORE_TOTAL}

def file_name_comparator(x):
	x = os.path.basename(x)
	return int(x[3:x.index('.')])

def get_sources(path):
	return [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f.startswith('log') and (f.endswith('.txt') or f.ends('.log'))]

def plot_data(files, metric, samples, xlabel='x', ylabel='y', legend='legend', i=-1, f=-1):
	x = []
	y = []
	idx = 0
	avg = 0.0
	count = 0
	files.sort(key=file_name_comparator)
	counted_samples = 0
	for ff in files:
		arq = open(ff, 'r')
		
		line = arq.readline()
		while line:
			if 'REWARDS:' in line:
				line = line.strip().split(' ')
				gs = int( line[IDX_EPISODE_NB][0:-1] )
				score = float(line[metric][0:-1])
				avg += score 
				counted_samples += 1
				if gs % samples == 0 and gs > 0 and (i < 0 or gs >= i) and (f < 0 or gs <= f):
					y.append(avg/counted_samples)
					counted_samples = 0
					avg = 0.0
					x.append(idx)
					idx += 1
				count += 1
			line = arq.readline()
		arq.close()
	plt.plot(x, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(legend)
	plt.show()

def print_help():
	print("analyser -i <inputfile> -m <metric> -s <samples_number>")
	print("<metric> pode ser 8 ou 12, que significam score e perda (loss), respectivamente.")

path = "."

try:
	opts, args = getopt.getopt(sys.argv[1:], "hi:m:s:s:e:a:b:x:y:")
	path = None
	metric = -1
	samples = 200
	xlabel = 'time step'
	ylabel = 'score'
	i = -1
	f = -1
	for opt, arg in opts:
		if opt=='-h':
			print_help()
		elif opt=='-i':
			path = arg
		elif opt=='-m':
			metric = IDX_NAME[arg]
		elif opt=='-s':
			samples = int(arg)
		elif opt=='-a':
			i = int(arg)
		elif opt=='-b':
			f = int(arg)
		elif opt=='-x':
			xlabel = arg
		elif opt=='-y':
			ylabel = arg
	if path and metric > 0:
		plot_data(get_sources(path), metric, samples, xlabel, ylabel, "", i, f)
	else:
		print("ERRO: tente seguir as seguintes instruções")
		print_help()
except getopt.GetoptError:
	print_help()