from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import json
import os
from model import build_model
from constants import *
import numpy as np


print('\n--- Reading questions...')
def read_questions(path):
  with open(path, 'r') as file:
    qs = json.load(file)
  texts = [q[0] for q in qs]
  answers = [q[1] for q in qs]
  image_ids = [q[2] for q in qs]
  return (texts, answers, image_ids)
train_qs, train_answers, train_image_ids = read_questions('data/train/questions.json')
test_qs, test_answers, test_image_ids = read_questions('data/test/questions.json')
all_qs = train_qs + test_qs
print(f'Read {len(train_qs)} training questions and {len(test_qs)} testing questions.')


print('\n--- Reading answers...')
with open('data/answers.txt', 'r') as file:
  all_answers = [a.strip() for a in file]
num_answers = len(all_answers)
print(f'Found {num_answers} total answers.')


print('\n--- Reading/processing training images...')
def normalize_img(im):
  return im / 255 - 0.5
def read_images(dir):
  ims = {}
  for filename in os.listdir(dir):
    if filename.endswith('.png'):
      image_id = int(filename[:-4])
      ims[image_id] = normalize_img(img_to_array(load_img(os.path.join(dir, filename))))
  return ims
train_ims = read_images('data/train/images')
test_ims = read_images('data/test/images')
im_shape = train_ims[0].shape
print(f'Read {len(train_ims)} training images and {len(test_ims)} testing images.')
print(f'Each image has shape {im_shape}.')


print('\n--- Fitting question tokenizer...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_qs)
vocab_size = len(tokenizer.word_index)
print(f'Found {vocab_size} words total.')


print('\n--- Converting questions to bags of words...')
def seq_to_bow(seq):
  bow = np.zeros(vocab_size)
  for i in range(vocab_size):
    bow[i] = seq.count(i + 1)
  return bow
def texts_to_bows(texts):
  seqs = tokenizer.texts_to_sequences(texts)
  return [seq_to_bow(seq) for seq in seqs]
train_X_seqs = texts_to_bows(train_qs)
test_X_seqs = texts_to_bows(test_qs)
print(f'Example question bag of words: {train_X_seqs[0]}')


print('\n--- Creating model input images...')
train_X_ims = [train_ims[id] for id in train_image_ids]
test_X_ims = [test_ims[id] for id in test_image_ids]


print('\n--- Creating model outputs...')
train_answer_indices = [all_answers.index(a) for a in train_answers]
test_answer_indices = [all_answers.index(a) for a in test_answers]
train_Y = to_categorical(train_answer_indices)
test_Y = to_categorical(test_answer_indices)
print(f'Example model output: {train_Y[0]}')


print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers)

model.load_weights('model_weights')
predictions = model.predict([train_X_ims, train_X_seqs])

# for idx in range(num_answers):
# 	pred_values = predictions[:, idx]
# 	answer = all_answers[idx]
# 	print(f'\nStatistics for answer {idx}, answer {answer}')
# 	min = np.amin(pred_values)
# 	max = np.amax(pred_values)
# 	mean = np.mean(pred_values)
# 	print(f'\nMin: {min}, Max: {max}, Mean: {mean}')

for idx in range(len(train_answer_indices)):
	# answer numbers for triangle, circle, rectangle
	answer = train_answer_indices[idx]
	if answer == 5 or answer == 9 or answer == 12:
		print(f"Answer {answer}, predictions {predictions[idx][5]}, {predictions[idx][9]}, {predictions[idx][12]}")