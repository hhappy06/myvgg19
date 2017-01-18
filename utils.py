import skimage
import skimage.io
import skimage.transform
import numpy as np

def load_image(image_path):
	ori_image = skimage.io.imread(image_path)
	ori_image = ori_image.astype(float)
	short_edge = min(ori_image.shape[:2])
	xx = int((ori_image.shape[0] - short_edge) / 2)
	yy = int((ori_image.shape[1] - short_edge) / 2)

	crop_image = ori_image[xx : xx+short_edge, yy : yy + short_edge]
	# resize image
	resize_image = skimage.transform.resize(crop_image, (224, 224))
	return ori_image, resize_image

def print_prob(prob, class_file):
	with open(class_file, 'r') as infile:
		synset = [l.strip() for l in infile.readlines()]

	pred = np.argsort(prob)[::-1]
	top1 = synset[pred[0]]
	print("Top1: ", top1, prob[pred[0]])
	# Get top5 label
	top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
	print("Top5: ", top5)
	return top1
