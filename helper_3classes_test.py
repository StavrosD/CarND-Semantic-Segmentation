'''
You should not edit helper.py as part of your submission.

This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
'''

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
	"""
	Report download progress to the terminal.
	:param tqdm: Information fed to the tqdm library to estimate progress.
	"""
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		Store necessary information for tracking progress.
		:param block_num: current block of the download
		:param block_size: size of current block
		:param total_size: total download size, if known
		"""
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)  # Updates progress
		self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
	"""
	Download and extract pretrained vgg model if it doesn't exist
	:param data_dir: Directory to download the model to
	"""
	vgg_filename = 'vgg.zip'
	vgg_path = os.path.join(data_dir, 'vgg')
	vgg_files = [
		os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
		os.path.join(vgg_path, 'variables/variables.index'),
		os.path.join(vgg_path, 'saved_model.pb')]

	missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
	if missing_vgg_files:
		# Clean vgg dir
		if os.path.exists(vgg_path):
			shutil.rmtree(vgg_path)
		os.makedirs(vgg_path)

		# Download vgg
		print('Downloading pre-trained vgg model...')
		with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
			urlretrieve(
				'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
				os.path.join(vgg_path, vgg_filename),
				pbar.hook)

		# Extract vgg
		print('Extracting model...')
		zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
		zip_ref.extractall(data_dir)
		zip_ref.close()

		# Remove zip file to save space
		os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
	"""
	Generate function to create batches of training data
	:param data_folder: Path to folder that contains all the datasets
	:param image_shape: Tuple - Shape of image
	:return:
	"""
	def get_batches_fn(batch_size):
		"""
		Create batches of training data
		:param batch_size: Batch Size
		:return: Batches of training data
		"""
		# Grab image and label paths
		image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
		label_paths = {
			re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
			for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
		background_color = np.array([255, 0, 0])    # the background color is red. The default color scipy space ir RGB.
		other_road_color = np.array([0,0,0])        # the other lane color is black. 
		road_color = np.array([0xff, 0x46, 0xf9])   # the color for my road . 
		
		# Shuffle training data
		random.shuffle(image_paths)
		# Loop through batches and grab images, yielding each batch
		for batch_i in range(0, len(image_paths), batch_size):
			images = []
			gt_images = []
			for image_file in image_paths[batch_i:batch_i+batch_size]:
				gt_image_file = label_paths[os.path.basename(image_file)]
				# Re-size to image_shape
				image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
				gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

				# Create "one-hot-like" labels by class
                # Background
				# gt_bg = np.all(gt_image == background_color, axis=2)			
				# gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
				
                # other road
				gt_or = np.all(gt_image == other_road_color, axis=2)	
				gt_or = gt_or.reshape(*gt_or.shape,1)

                # my road
				gt_mr = np.all(gt_image == road_color, axis=2)				
				gt_mr = gt_mr.reshape(*gt_mr.shape,1)
				                
                # first column: Background
                # second column: Other road
                # third column: My road, the road where the camera is.
				# gt_image = np.concatenate((gt_bg, gt_or, gt_mr), axis=2) 						
				gt_image = np.concatenate((np.invert(gt_or+gt_mr), gt_or, gt_mr), axis=2) 										
				images.append(image)
				gt_images.append(gt_image)

			yield np.array(images), np.array(gt_images)
	return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
	"""
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""
	for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
		image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

		# Run inference
		im_softmax = sess.run(
			[tf.nn.softmax(logits)],
			{keep_prob: 1.0, image_pl: [image]})

        # Splice out first (background), reshape output back to image_shape
		im_background = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])

        # Splice out second column (other road), reshape output back to image_shape
		im_other_lane_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])

		# Splice out third column (my road), reshape output back to image_shape
		im_my_softmax = im_softmax[0][:, 2].reshape(image_shape[0], image_shape[1])

        # If background softmax > 0.5, prediction is background
		segmentation_background = (im_background > 0.8).reshape(image_shape[0], image_shape[1], 1)

		# If other road softmax > 0.5, prediction is road
		segmentation_other = (im_other_lane_softmax > 0.05).reshape(image_shape[0], image_shape[1], 1)

        # If my road softmax > 0.5, prediction is road
		segmentation_my = (im_my_softmax > 0.05).reshape(image_shape[0], image_shape[1], 1)

		# Create mask based on segmentation to apply to original image
        # create the mask for the background
		mask_background = np.dot(segmentation_background, np.array([[255, 0, 0, 100]]))    # use red for the background
		mask_background = scipy.misc.toimage(mask_background, mode="RGBA")		
		street_im = scipy.misc.toimage(image)
		street_im.paste(mask_background, box=None, mask=mask_background)
        # create the mask for other road
		mask_other = np.dot(segmentation_other, np.array([[0, 0, 255, 127]]))    # use blue color for the other road
		mask_other = scipy.misc.toimage(mask_other, mode="RGBA")		
		street_im.paste(mask_other, box=None, mask=mask_other)
        # create the mask for my road
		mask_my = np.dot(segmentation_my, np.array([[0, 255, 0, 127]]))  # use green color for my road
		mask_my = scipy.misc.toimage(mask_my, mode="RGBA")
		street_im.paste(mask_my, box=None, mask=mask_my)

		yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, create_video):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	print("Start saving images\n")
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to HD
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = gen_test_output(
		sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
	if create_video:
		import imageio 
		writer = imageio.get_writer(str(time.time()) + 'output_video.mp4', fps=10)  

	for name, image in image_outputs:
		print("Current image: ", os.path.join(output_dir, name), "\n")
		scipy.misc.imsave(os.path.join(output_dir, name), image)
		writer.append_data(image)

	if create_video:
		writer.close()