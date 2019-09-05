
#!git clone https://github.com/26medias/keras-face-toolbox.git
#!mv keras-face-toolbox/models models
#!mv keras-face-toolbox/utils utils
#!rm -r keras-face-toolbox
#!gdown https://drive.google.com/uc?id=1H37LER8mRRI4q_nxpS3uQz3DcGHkTrNU
#!mv lresnet100e_ir_keras.h5 models/verifier/insightface/lresnet100e_ir_keras.h5
#!pip install git+https://github.com/rcmalli/keras-vggface.git
#!pip show keras-vggface
#!pip install matplotlib
#!pip install mtcnn
#!pip install bs4
#!pip install selenium



from IPython.display import HTML, display
import time
import requests
import ntpath
import cv2
import math
import os, sys
from matplotlib import pyplot
from PIL import Image
import numpy as np
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import glob
import mtcnn
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.cluster import	hierarchy
from bs4 import BeautifulSoup
from selenium import webdriver
import csv
from models.detector import face_detector
from models.parser import face_parser
from utils.visualize import show_parsing_with_annos



class builder():
	def __init__(self, VIDEO_QUALITY="720", FRAME_PERCENTAGE=40, DIR_VIDEOS="Videos", DIR_FACES="Faces"):
		# The variables
		self.VIDEO_QUALITY     = VIDEO_QUALITY     # The trailer quality we'll download: 480, 720 or 1080
		self.FRAME_PERCENTAGE  = FRAME_PERCENTAGE  # from 0.1 to 100: The percentage of frames that will be analyzed in the video
		self.DIR_VIDEOS        = DIR_VIDEOS
		self.DIR_FACES         = DIR_FACES

		if not os.path.isdir(self.DIR_VIDEOS):
			os.mkdir(self.DIR_VIDEOS, 755);
		if not os.path.isdir(self.DIR_FACES):
			os.mkdir(self.DIR_FACES, 755);
		
		# Create the detector, using default weights
		print("Creating the detector model")
		self.detector = MTCNN()

		# Create a vggface model
		print("Creating the face embedding model")
		self.embedding_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

		# Create a face detector
		print("Creating the face detector model")
		self.fd = face_detector.FaceAlignmentDetector(lmd_weights_path="models/detector/FAN/2DFAN-4_keras.h5")

		# Create a face parser (segmentation)
		print("Creating the face segmentation model")
		self.prs = face_parser.FaceParser()

	# The methods
	# ===========

	# Colab progress bar
	def progress(self, value, max=100):
		return HTML('<progress value="{value}" max="{max}" style="width: 50%"> {value}</progress>'.format(value=value, max=max))

	# Convert a value from one range to another
	def rangeConvert(self, x, in_min, in_max, out_min, out_max):
		return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

	# Get the directory of a filename
	def getDir(self, filename):
		p = Path(filename);
		return p.parts[len(p.parts)-2]

	# Dowload a video from a url
	def downloadFile(self, url):
		print("Downloading ", url)
		filename = self.DIR_VIDEOS+"/"+ntpath.basename(url)
		if os.path.exists(filename):
			return filename
		myfile = requests.get(url)
		open(filename, 'wb').write(myfile.content)
		print(filename," downloaded.")
		return filename

	# Resize an image
	def resize_image(self, im, max_size=768):
		if np.max(im.shape) > max_size:
			ratio = max_size / np.max(im.shape)
			print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
			return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
		return im


	def imageFilesToGrid(self, directory, outputFilename):
		filenames = glob.glob(directory+'/*.jpg')
		#print(directory, ": ", len(filenames), " images")
		if len(filenames) < 4:
			return False
		result_figsize_resolution = 10 # 1 = 100px
		
		images_count = len(filenames)
		# Calculate the grid size:
		grid_size = math.ceil(math.sqrt(images_count))
		
		# Create plt plot:
		fig, axes = pyplot.subplots(grid_size, grid_size, figsize=(result_figsize_resolution, result_figsize_resolution))
		
		current_file_number = 0
		for image_filename in filenames:
			x_position = current_file_number % grid_size
			y_position = current_file_number // grid_size
			plt_image = pyplot.imread(image_filename)
			axes[x_position, y_position].imshow(plt_image)
			current_file_number += 1
		pyplot.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
		pyplot.savefig(outputFilename)
		#pyplot.show()

	def exportImageGrids(self, directory, outputDirectory):
		print("Exporting image grids...")
		dirs = os.listdir(directory)
		dirs.sort()
		ndirs = len(dirs)
		for n,dir in enumerate(dirs):
			if dir is not "ALL":
				self.imageFilesToGrid(directory+"/"+dir, outputDirectory+"/"+dir+".jpg");
			self.progress(n, ndirs)

	# Extract the faces from an image, return an array of numpy faces
	def extractFacesFromImage(self, pixels, required_size=(224, 224), limit=50):
		results = self.detector.detect_faces(pixels)
		faces = []
		errors = 0
		for i,faceData in enumerate(results):
			if len(faces) > limit:
				break
			x1, y1, width, height = faceData['box']
			x2, y2 = x1 + width, y1 + height
			# extract the face
			face = pixels[y1:y2, x1:x2]
			# resize pixels to the model size
			try:
				image = Image.fromarray(face)
				image = image.resize(required_size)
				face_array = asarray(image)
				faces.append(face_array)
				if limit==1:
					return face_array
			except:
				errors+=1
		if limit==1 and len(faces)==0:
			return False
		return faces;

	# Extract the faces from an image, return an array of numpy faces & landmarks
	def extractFacesAndLandmarksFromImage(self, pixels, required_size=(224, 224), limit=50):
		rw, rh = required_size
		results, landmarks = self.fd.detect_face(pixels, with_landmarks=True)
		nResults = len(results)
		faces = []
		errors = 0
		for i,bbox in enumerate(results):
			if len(faces) > limit:
				break
			# Get the face
			x0, y0, x1, y1, score = bbox
			# Find the center of the face
			w			 = x1-x0
			h			 = y1-y0
			xCenter = x0+int(w/2)
			yCenter = y0+int(h/2)
			if w>h:
				y0 = yCenter-int(w/2)
				y1 = yCenter+int(w/2)
			if h>w:
				x0 = xCenter-int(h/2)
				x1 = xCenter+int(h/2)
			x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
			face = pixels[x0:x1, y0:y1, :]
			# Recalculate the landmarks coordinates
			for li in range(len(landmarks[i])): 
				landmark = landmarks[i][li]
				lx, ly = landmark
				landmarks[i][li] = (self.rangeConvert(lx-x0, 0, face.shape[1], 0, rw), self.rangeConvert(ly-y0, 0, face.shape[0], 0, rh))
			# Resize pixels to the model size
			try:
				image = Image.fromarray(face)
				image = image.resize(required_size)
				face_array = asarray(image)
				faces.append(face_array)
				if limit==1:
					return face_array
			except:
				errors+=1
		if limit==1 and len(faces)==0:
			return False
		return faces, landmarks

	# Extract the faces from an image, return an array of numpy faces & landmarks
	def extractFacesLandmarksAndSegmentationFromImage(self, pixels, required_size=(224, 224), limit=50):
		rw, rh = required_size
		results, landmarks = self.fd.detect_face(pixels, with_landmarks=True)
		nResults = len(results)
		faces = []
		segmentations = []
		errors = 0
		for i,bbox in enumerate(results):
			if len(faces) > limit:
				break
			# Get the face
			x0, y0, x1, y1, score = bbox
			# Find the center of the face
			w			 = x1-x0
			h			 = y1-y0
			xCenter = x0+int(w/2)
			yCenter = y0+int(h/2)
			if w>h:
				y0 = yCenter-int(w/2)
				y1 = yCenter+int(w/2)
			if h>w:
				x0 = xCenter-int(h/2)
				x1 = xCenter+int(h/2)
			x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
			face = pixels[x0:x1, y0:y1, :]
			# Recalculate the landmarks coordinates
			for li in range(len(landmarks[i])): 
				landmark = landmarks[i][li]
				lx, ly = landmark
				landmarks[i][li] = (self.rangeConvert(lx-x0, 0, face.shape[1], 0, rw), self.rangeConvert(ly-y0, 0, face.shape[0], 0, rh))
			# Resize pixels to the model size
			try:
				image = Image.fromarray(face)
				image = image.resize(required_size)
				face_array = asarray(image)
				faces.append(face_array)
				# Get the segmentation on the resized image
				segmentation = self.prs.parse_face(face_array)
				segmentations.append(segmentation)
				if limit==1:
					return face_array
			except:
				errors+=1
		if limit==1 and len(faces)==0:
			return False
		return faces, landmarks, segmentations

	# Export the frames out of a video at a specific fps
	def videoToFaces(self, filename, maxFrame=0):
		print("Extracting faces from the video frames...")
		basename = os.path.splitext(ntpath.basename(filename))[0]
		#print("basename:", basename)
		cap = cv2.VideoCapture(filename)
		# Get the video's FPS
		
		fps = cap.get(cv2.CAP_PROP_FPS)
		nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			
		processFrames = int(nframes*self.FRAME_PERCENTAGE/100)
		skipFrame		 = int(nframes/processFrames)
		print(basename, "fps:", fps, "skipFrame:",skipFrame,"Frames:", str(processFrames)+"/"+str(nframes))
		out = display(self.progress(0, processFrames), display_id=True)
		i = 0
		c = 0
		faces = []
		landmarks = []
		segmentations = []
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == False:
					break
			i+=1
			if maxFrame>0 and i > maxFrame:
				break;
			#print(i, "-", i % skipFrame)
			if (i % skipFrame == 0):
				c+=1
				#print("Checking faces in frame #"+str(i))
				#frameFaces = self.extractFacesFromImage(frame)
				frameFaces, frameLandmarks, frameSegmentations = self.extractFacesLandmarksAndSegmentationFromImage(frame)
				out.update(self.progress(c, processFrames))
				for nf, f in enumerate(frameFaces):
					faces.append(f)
					landmarks.append(frameLandmarks[nf])
					segmentations.append(frameSegmentations[nf])
			else:
				continue
			#cv2.imwrite(DIR_IMAGES+"/"+basename+'/'+str(round((i-1)/fps,2))+'sec.jpg',frame)
		cap.release()
		cv2.destroyAllWindows()
		print(basename, " processed.")
		print(processFrames,"/",nframes," frames analyzed.")
		print(len(faces), " faces found.")
		return faces, landmarks, segmentations


	# Show a few images
	def showImages(self, images, width=4):
		fig = pyplot.figure(figsize=(width, math.ceil(len(images)/width)))
		for i in range(len(images)):
			pyplot.subplot(width, math.ceil(len(images)/width), i+1)
			pyplot.imshow(images[i])
			pyplot.axis('off')
		pyplot.savefig('preview.png')
		pyplot.show()

	# Save an array of images to files
	def saveImages(self, images, dest, names=False, prefix="", showProgress=True):
		if not os.path.isdir(dest):
			os.mkdir(dest, 755);
		nImages = len(images)
		if showProgress is True:
			print("Saving ",nImages," images to ", dest)
			out = display(self.progress(0, nImages), display_id=True)
		filenames = []
		for n, image in enumerate(images):
			if names is False:
				filename = dest+"/"+prefix+('{:04d}'.format(n))+'.jpg'
			else:
				filename = dest+"/"+prefix+str(names[n])+'.jpg'
			cv2.imwrite(filename, image)
			filenames.append(filename)
			if showProgress is True:
				out.update(self.progress(n, nImages))
		return filenames

	# Save Numpy Arrays to files
	def saveNpArrays(self, npArrays, dest, names=False, prefix="", showProgress=True):
		if not os.path.isdir(dest):
			os.mkdir(dest, 755);
		nArrays = len(npArrays)
		if showProgress is True:
			print("Saving ",nArrays," numpy arrays to ", dest)
			out = display(self.progress(0, nArrays), display_id=True)
		filenames = []
		for n, npArray in enumerate(npArrays):
			if names is False:
				filename = dest+"/"+prefix+('{:04d}'.format(n))+'.npy'
			else:
				filename = dest+"/"+prefix+str(names[n])+'.npy'
			np.save(filename, npArray)
			filenames.append(filename)
			if showProgress is True:
				out.update(self.progress(n, nArrays))
		return filenames


	# Extract faces and calculate face embeddings for a list of photo files
	def get_embeddings(self, faces):
		print("Calculating the embeddings...")
		# convert into an array of samples
		samples = asarray(faces, 'float32')
		# prepare the face for the model, e.g. center pixels
		samples = preprocess_input(samples, version=2)
		# perform prediction
		embeddings = self.embedding_model.predict(samples)
		return embeddings


	# Determine if a candidate face is a match for a known face
	def is_match(self, known_embedding, candidate_embedding, threshold=0.5):
		# calculate distance between embeddings
		score = cosine(known_embedding, candidate_embedding)
		return score >= threshold

	# Cluster the faces by cosine distance
	def clusterFaces(self, faces, embeddings, landmarks, segmentations, minFaces=2):
		groups = [] # Array of dict {faces:[], embeddings: []}
		nFaces = len(faces)
		print("Clustering ",nFaces," faces...")
		out = display(self.progress(0, nFaces), display_id=True)
		# For each faces
		for n, face in enumerate(faces):
			out.update(self.progress(n, nFaces))
			if len(groups)==0:
				groups.append({
					"faces":				 [face],
					"names":				 [n],
					"embeddings":		[embeddings[n]],
					"landmarks":		 [landmarks[n]],
					"segmentations": [segmentations[n]]
				})
			else:
				# Not the first face, match it against all the groups, see if the average of cosine distance match an existing face
				scores = [] # array of dict {group: n, embeddings: []}
				for g, group in enumerate(groups):
					groupScores = []
					for embedding in group["embeddings"]:
						groupScores.append(cosine(embedding, embeddings[n]))
					score = np.mean(groupScores)
					scores.append({
							"group": g,
							"score": score
					})
				# Sort the scores for each group by lowest score, check if that score is below the threshold
				scores = sorted(scores, key = lambda i: i["score"], reverse=False)
				if scores[0]["score"] <= 0.5:
					# Add to the existing group the face matches
					groups[scores[0]["group"]]["landmarks"].append(landmarks[n])
					groups[scores[0]["group"]]["embeddings"].append(embeddings[n])
					groups[scores[0]["group"]]["segmentations"].append(segmentations[n])
					groups[scores[0]["group"]]["faces"].append(face)
					groups[scores[0]["group"]]["names"].append(n)
					#print("[Matched] face #", n, " to group #", scores[0]["group"], "score:", scores[0]["score"])
				else:
					groups.append({
						"faces":			 [face],
						"names":			 [n],
						"embeddings":	[embeddings[n]],
						"landmarks":	 [landmarks[n]],
						"segmentations": [segmentations[n]]
					})
					#print("[New face] face #", n, " / Best score:", scores[0]["score"])
		# Filter out the groups that don't have enough faces
		return [item for item in groups if len(item["faces"]) >= minFaces]
		#return groups;

	# Cluster all the faces from a remote video
	def clusterFacesOnVideo(self, url):
		print("Processing ", url);
		# Download the video
		videoFilename = self.downloadFile(url)
		
		# Get the directories name for that video
		# /Faces/[dirname]/Faces
		# /Faces/[dirname]/Embeddings
		# /Faces/[dirname]/Landmarks
		# /Faces/[dirname]/Segmentations
		# /Faces/[dirname]/Previews
		dirname					= os.path.splitext(ntpath.basename(videoFilename))[0]
		
		dirClustered		 = self.DIR_FACES+"/"+dirname
		dirFaces				 = dirClustered+"/Faces/"
		dirEmbeddings		= dirClustered+"/Embeddings/"
		dirLandmarks		 = dirClustered+"/Landmarks/"
		dirSegmentations = dirClustered+"/Segmentations/"
		dirPreviews			= dirClustered+"/Previews/"
		
		if os.path.exists(dirPreviews):
			# Video already processed, go to the next one
			print("Video already processed.")
			#return False
		
		# Create the directories
		if not os.path.isdir(dirClustered):
			os.mkdir(dirClustered, 755);
		if not os.path.isdir(dirFaces):
			os.mkdir(dirFaces, 755);
		if not os.path.isdir(dirEmbeddings):
			os.mkdir(dirEmbeddings, 755);
		if not os.path.isdir(dirLandmarks):
			os.mkdir(dirLandmarks, 755);
		if not os.path.isdir(dirSegmentations):
			os.mkdir(dirSegmentations, 755);
		if not os.path.isdir(dirPreviews):
			os.mkdir(dirPreviews, 755);
		
		# Open a CSV to save the datasets
		with open(dirClustered+"/"+dirname+".csv", "w") as csvfile:
			fieldnames = ["video_name", "face_group", "image_filename", "embeddings_filename", "landmarks_filename", "segmentations_filename"]
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			
			# Find the faces on the video
			faces, landmarks, segmentations	= self.videoToFaces(videoFilename)
			nFaces			 = len(faces)
			print(nFaces," faces detected")

			# Get the embedding for all the faces
			embeddings		= self.get_embeddings(faces)
			
			# Cluster the faces using cosine distance
			clusters			= self.clusterFaces(faces, embeddings, landmarks, segmentations, minFaces=5)
			nClusters		 = len(clusters)

			# Export each face group
			print("Saving ",nClusters," face clusters...")
			for n, group in enumerate(clusters):
				
				ngImg = len(group["faces"])
				ngEbd = len(group["embeddings"])
				ngldk = len(group["landmarks"])
				
				# Save the face as an image
				images_filenames				= self.saveImages(group["faces"], dirFaces+('{:04d}'.format(n)), showProgress=False)
				
				# Save the embedding as a numpy array
				embeddings_filenames		= self.saveNpArrays(group["embeddings"], dirEmbeddings+('{:04d}'.format(n)), showProgress=False)
				
				# Save the landmarks as a numpy array
				landmarks_filenames		 = self.saveNpArrays(group["landmarks"], dirLandmarks+('{:04d}'.format(n)), showProgress=False)
				
				# Save the segmentations as a numpy array
				segmentations_filenames = self.saveNpArrays(group["segmentations"], dirSegmentations+('{:04d}'.format(n)), showProgress=False)
				
				
				# Update the CSV
				for i, image_filename in enumerate(images_filenames):
					writer.writerow({
						"video_name": dirname,
						"face_group": n,
						"image_filename": image_filename,
						"embeddings_filename": embeddings_filenames[i],
						"landmarks_filename": landmarks_filenames[i],
						"segmentations_filename": segmentations_filenames[i]
					})


			# Build grids to show each face groups
			self.exportImageGrids(dirFaces, dirPreviews)


	def clusterFacesFromVideos(self, urls):
		nUrls = len(urls)
		for n,url in enumerate(urls):
			self.clusterFacesOnVideo(url)

	def fetchAllHDVideos(self, url):
		response = requests.get(url)
		soup = BeautifulSoup(response.content, "html5lib")
		links = soup.find_all('a')
		videos = []
		for tag in links:
			link = tag.get('href', None)
			if link is not None and 'h'+str(self.VIDEO_QUALITY)+'p' in link:
				videos.append(link)
		return videos