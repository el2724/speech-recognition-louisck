
# for indexing files
from os import listdir
from os.path import isfile, join, splitext
from random import shuffle
# for analyzing data
import sys
sys.path.append('./python_speech_features-master')
from features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import tree
import sklearn.metrics as metrics


def getFiles(dirpath):
	'''gets all filenames in the directory'''
	return [f for f in listdir(dirpath) if isfile(join(dirpath, f))]

def labelFiles(dirpaths):
	'''puts files in either punchline or setup list (ignores laughter files)'''
	labeled_files = { 'punchline': [], 'setup': [] }
	for d in dirpaths:
		files = getFiles(d)
		for f in files:
			(name, ext) = splitext(f)
			if 'a' in name:
				labeled_files['setup'].append(join(d,f))
			elif 'p' in name:
				labeled_files['punchline'].append(join(d,f))
			else: #skip 'l' - laughter
				continue
	return labeled_files

def partitionData(labeled_files, chunks):
	'''partitions the data equally into chunks'''
	# randomize order of files
	for k in labeled_files:
		shuffle(labeled_files[k])

	# get proportion of each filetype (either punchline or setup)
	proportion = {}
	for k in labeled_files:
		proportion[k] = len(labeled_files[k])/chunks

	# grab the chunks
	partitions = [{'punchline': [], 'setup': []} for i in range(chunks)]
	for k in labeled_files:
		num_files = proportion[k]
		for i, partition in enumerate(partitions):
			partition[k] = labeled_files[k][i*num_files:(i+1)*num_files]

	return partitions

def testTrainSplit(labeled_files):
	'''partitions the data into test and train data pairs'''
	# partition data into test/train for each chunk
	x = partitionData(labeled_files, 2)
	train = x[0]
	test = x[1]
	# combine punchline and setup files
	traindata = []
	for k in train:
		traindata += [(n,k) for n in train[k]]
	testdata = []
	for k in test:
		testdata += [(n,k) for n in test[k]]
	return testdata, traindata

def wavToMFCC(path, winlen=0.025, winstep=0.01, numcep=13):
	'''returns mfcc of a given wav file'''
	(rate,sig) = wav.read(path)
	mfcc_feat = mfcc(sig,samplerate=rate,winlen=winlen,winstep=winstep,numcep=numcep)
	return mfcc_feat

def taggedFilesToMFCC(files, winlen=0.025, winstep=0.01, numcep=13):
	'''returns pairs of mfcc feats and its tag'''
	pairs = []
	for (path,tag) in files:
		mfcc = wavToMFCC(path, winlen=winlen, winstep=winstep, numcep=numcep)
		for i in range(len(mfcc)):
			pairs.append((mfcc[i],tag))
	return pairs

def taggedFeatsToNumpy(tagged_feats, filterind=None, filterthresh=None):
	'''converts tagged feature pairs to numpy'''
	points = []
	labels = []
	for (feats, label) in tagged_feats:
		if filterind and filterthresh and feats[filterind] < filterthresh:
			continue
		points.append(feats)
		labels.append(label)
	return np.array(points), np.array(labels)

def addCPfeats(tagged_feats, winsize=20):
	'''
	adds CP features to the end of the existing features,
	edge points are removed due to windowing
	'''
	num_feats = len(tagged_feats)
	if num_feats < 2*winsize:
		raise Exception('Not enough points given, window size too large')

	scores = []
	for i in range(2*winsize,num_feats+1):
		# grab windows
		prev = tagged_feats[i-2*winsize:i-winsize]
		curr = tagged_feats[i-winsize:i]
		# convert to numpy
		prev_points = taggedFeatsToNumpy(prev)[0]
		curr_points = taggedFeatsToNumpy(curr)[0]
		total_points = np.concatenate((prev_points, curr_points))
		# compute sufficient statistics
		prev_mean = np.mean(prev_points, axis=0)
		curr_mean = np.mean(curr_points, axis=0)
		total_mean = np.mean(total_points, axis=0)
		prev_cov = np.cov(prev_points, rowvar=0)
		curr_cov = np.cov(curr_points, rowvar=0)
		total_cov = np.cov(total_points, rowvar=0)
		# compute log likelihoods
		prev_lls = multivariate_normal.logpdf(prev_points, mean=prev_mean, cov=prev_cov)
		curr_lls = multivariate_normal.logpdf(curr_points, mean=curr_mean, cov=curr_cov)
		total_lls = multivariate_normal.logpdf(total_points, mean=total_mean, cov=total_cov)
		# compute overall changepoint score
		score = np.sum(prev_lls) + np.sum(curr_lls) - np.sum(total_lls)
		scores.append(score)

	# remove edge points
	for i in range(winsize):
		tagged_feats.pop(0)
	for i in range(winsize-1):
		tagged_feats.pop()
	# add changepoint scores to features
	for i, ((feats, label), score) in enumerate(zip(tagged_feats, scores)):
		tagged_feats[i] = (np.append(feats, score), label)

def taggedFilesToFeats(files, cpwinsize=30, mfccwinlen=0.025, mfccwinstep=0.01, mfccnumcep=13):
	# Extract MFCC features
	tagged_feats = taggedFilesToMFCC(files, winlen=mfccwinlen, winstep=mfccwinstep, numcep=mfccnumcep)
	# add CP features
	addCPfeats(tagged_feats, winsize=cpwinsize)
	return tagged_feats

def trainDT(tagged_feats, cpthresh=0.0, cpfeatind=13):
	'''returns a decision tree trained on the labelled features'''
	# convert features to numpy; filter by cpthresh
	points, labels = taggedFeatsToNumpy(tagged_feats, filterind=cpfeatind, filterthresh=cpthresh)
	# convert 'setup' and 'punchline' to -1 and 1, respectively
	labels = [-1 if k=='setup' else 1 for k in labels]
	# fit data
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(points, labels)
	return clf

def testDT(dt, tagged_feats, cpthresh=0.0, cpfeatind=13):
	'''returns results of testing decision tree dt on the labelled features'''
	# convert features to numpy; filter by bythresh
	points, labels = taggedFeatsToNumpy(tagged_feats, filterind=cpfeatind, filterthresh=cpthresh)
	labels = [-1 if k=='setup' else 1 for k in labels]
	results = dt.predict(points)
	return labels, results

def displayMetrics(labels, results):
	'''prints metrics for the result'''
	print 'num points', len(labels)
	print 'num setup', labels.count(-1)
	print 'num punchline', labels.count(1)
	outputs = {
		'f1 score': metrics.f1_score,
		'accuracy': metrics.accuracy_score,
		'precision': metrics.precision_score,
		'recall': metrics.recall_score
	}
	for metric in outputs:
		print metric, outputs[metric](labels,results)

def punchlineDetection(num_chunks=16, cp_thresh=-float('inf'), cp_win=30, mfccwin=0.025, mfccwinstep=0.01, mfccnumcep=13):
	relpath = 'data'
	dirs = ['bacontheatre','chewedup','comedystore','ohmygod']

	dirpaths = [join(relpath, d) for d in dirs]
	# pool data from all dirs together
	labeled_files = labelFiles(dirpaths)

	NUM_CHUNKS = num_chunks
	# partition data into chunks so it fits in memory
	partitions = partitionData(labeled_files, NUM_CHUNKS)

	partitions = partitions[0:1]
	print 'results for:'
	print 'num chunks: ', num_chunks
	print 'changepoint min threshold: ', cp_thresh
	print 'changepoint window size: ', cp_win
	print 'mfcc window size: ', mfccwin
	print 'mfcc window step: ', mfccwinstep
	print 'mfcc num ceps: ', mfccnumcep
	# train and test on each chunk
	for i, partition in enumerate(partitions):
		testdata, traindata = testTrainSplit(partition)

		# TRAIN
		tagged_feats = taggedFilesToFeats(traindata, cpwinsize=cp_win,
			mfccwinlen=mfccwin, mfccwinstep=mfccwinstep, mfccnumcep=mfccnumcep)
		# train DT
		dt = trainDT(tagged_feats, cpthresh=cp_thresh)
		# TEST
		tagged_feats = taggedFilesToFeats(testdata, cpwinsize=cp_win,
			mfccwinlen=mfccwin, mfccwinstep=mfccwinstep, mfccnumcep=mfccnumcep)
		# test DT
		labels, results = testDT(dt, tagged_feats, cpthresh=cp_thresh)

		# display results
		print 'chunk ' + str(i)
		displayMetrics(labels, results)
		print ''


if __name__ == '__main__':
	settings = [
		# increase ceps
		(1, 30.0, 30, 0.025, 0.01, 13),
		(1, 30.0, 30, 0.025, 0.01, 15),
		(1, 30.0, 30, 0.025, 0.01, 17),
		(1, 30.0, 30, 0.025, 0.01, 19),
		# increase cp_thresh
		(1, 25.0, 30, 0.025, 0.01, 13),
		(1, 35.0, 30, 0.025, 0.01, 13),
		(1, 40.0, 30, 0.025, 0.01, 13),
		(1, 45.0, 30, 0.025, 0.01, 13),
		# increase cp_win
		(1, 30.0, 20, 0.025, 0.01, 13),
		(1, 30.0, 25, 0.025, 0.01, 13),
		(1, 30.0, 30, 0.025, 0.01, 13),
		(1, 30.0, 35, 0.025, 0.01, 13)
	]
	settings = [{'num_chunks': f, 'cp_thresh': a, 'cp_win': b, 'mfccwin': c,
		'mfccwinstep': d, 'mfccnumcep': e }
		for (f,a,b,c,d,e) in settings]
	print ''
	for setting in settings:
		punchlineDetection(**setting)
		print '===================================='
		print ''
