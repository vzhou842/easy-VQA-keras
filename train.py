from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import argparse
import json
import os
from model import build_model
import numpy as np
from easy_vqa import get_train_questions, get_test_questions, get_train_image_paths, get_test_image_paths, get_answers

parser = argparse.ArgumentParser()
parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()
print(f'\n--- Using big model: {args.big_model}, using data directory: {args.use_data_dir}')

print('\n--- Reading questions...')
if args.use_data_dir:
  def read_questions(path):
    with open(path, 'r') as file:
      qs = json.load(file)
    texts = [q[0] for q in qs]
    answers = [q[1] for q in qs]
    image_ids = [q[2] for q in qs]
    return (texts, answers, image_ids)
  train_qs, train_answers, train_image_ids = read_questions('data/train/questions.json')
  test_qs, test_answers, test_image_ids = read_questions('data/test/questions.json')
else:
  train_qs, train_answers, train_image_ids = get_train_questions()
  test_qs, test_answers, test_image_ids = get_test_questions()
print(f'Read {len(train_qs)} training questions and {len(test_qs)} testing questions.')


print('\n--- Reading answers...')
if args.use_data_dir:
  with open('data/answers.txt', 'r') as file:
    all_answers = [a.strip() for a in file]
else:
  all_answers = get_answers()
num_answers = len(all_answers)
print(f'Found {num_answers} total answers:')
print(all_answers)


print('\n--- Reading/processing images...')
def load_and_proccess_image(image_path):
  # Load image, then scale and shift pixel values to [-0.5, 0.5]
  im = img_to_array(load_img(image_path))
  return im / 255 - 0.5

def read_images(paths):
  # paths is a dict mapping image ID to image path
  # Returns a dict mapping image ID to the processed image
  ims = {}
  for image_id, image_path in paths.items():
    ims[image_id] = load_and_proccess_image(image_path)
  return ims

if args.use_data_dir:
  train_ims = read_images(os.listdir('data/train/images'))
  test_ims  = read_images(os.listdir('data/test/images'))
else:
  train_ims = read_images(get_train_image_paths())
  test_ims = read_images(get_test_image_paths())
im_shape = train_ims[0].shape
print(f'Read {len(train_ims)} training images and {len(test_ims)} testing images.')
print(f'Each image has shape {im_shape}.')


print('\n--- Fitting question tokenizer...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_qs)

# We add one because the Keras Tokenizer reserves index 0 and never uses it.
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
model = build_model(im_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)


print('\n--- Training model...')
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=5,
  callbacks=[checkpoint],
)
