from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import argparse
import json
import os
from model import build_model
from constants import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--full-model', action='store_true')
parser.add_argument('--model-weights', help='model weights file')
args = parser.parse_args()
print('\n--- Calling train with full_model: {}'.format(args.full_model))
print('\n--- Model weights file: {}'.format(args.model_weights))

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
vocab_size = len(tokenizer.word_index) + 1
print(f'Vocab Size: {vocab_size}')
print(tokenizer.word_index)


print('\n--- Converting questions to bags of words...')
train_X_seqs = tokenizer.texts_to_matrix(train_qs)
test_X_seqs = tokenizer.texts_to_matrix(test_qs)
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
model = build_model(im_shape, vocab_size, num_answers, args.full_model)

model.load_weights(args.model_weights)
predictions = model.predict([test_X_ims, test_X_seqs])

# for idx in range(num_answers):
# 	pred_values = predictions[:, idx]
# 	answer = all_answers[idx]
# 	print(f'\nStatistics for answer {idx}, answer {answer}')
# 	min = np.amin(pred_values)
# 	max = np.amax(pred_values)
# 	mean = np.mean(pred_values)
# 	print(f'\nMin: {min}, Max: {max}, Mean: {mean}')

shapes = []
yesno = []
for i in range(num_answers):
  if (all_answers[i] == 'rectangle' or all_answers[i] == 'circle' or all_answers[i] == 'triangle'):
    shapes.append(i)
  elif all_answers[i] == 'yes' or all_answers[i] == 'no':
    yesno.append(i)

def return_class(answer):
  if answer in shapes:
    return 0
  if answer in yesno:
    return 1
  return 2
error_matrix = [[0 for _ in range(3)] for _ in range(3)]
total_errors = 0

color_error_matrix = [[0 for _ in range(num_answers)] for _ in range(num_answers)]
questions_wrong = {}

for idx in range(len(test_answer_indices)):
  # answer numbers for triangle, circle, rectangle
  answer = test_answer_indices[idx]
  pred = np.argmax(predictions[idx])
  if not answer == pred:
    total_errors += 1
    error_matrix[return_class(answer)][return_class(pred)] += 1
    color_error_matrix[answer][pred] += 1
    if (return_class(answer) == 1 and return_class(pred) == 1):
      if test_qs[idx] in questions_wrong:
        questions_wrong[test_qs[idx]] += 1
      else:
        questions_wrong[test_qs[idx]] = 1

print('total error: {}'.format(total_errors / len(test_answer_indices)))
print('Indexes are, in order, shapes, yes/no, colors')
print('Rows are class of answer, columns are class of prediction')
for i in range(3):
  print('{}\t{}\t{}\n'.format(error_matrix[i][0] / total_errors, error_matrix[i][1] / total_errors, error_matrix[i][2]/ total_errors))
print('-------------')
for i in range(num_answers):
  to_print = ''
  for j in range(num_answers):
    to_print += str(color_error_matrix[i][j]) + '\t'
  print(to_print)
print('-------------')
print(questions_wrong)
