from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img, img_to_array
import json
import os


print('\n--- Reading questions...')
def read_questions(path):
  with open(path, 'r') as file:
    qs = json.load(file)
  return qs
train_qs = read_questions('data/train/questions.json')
test_qs = read_questions('data/test/questions.json')
all_qs = train_qs + test_qs
print(f'Read {len(train_qs)} training questions and {len(test_qs)} testing questions.')


print('\n--- Reading answers...')
with open('data/answers.txt', 'r') as file:
  answers = [a.strip() for a in file]
  print(answers)
print(f'Found {len(answers)} total answers.')


print('\n--- Reading training images...')
def read_images(dir):
  ims = []
  for filename in os.listdir(dir):
    if filename.endswith('.png'):
      ims.append(img_to_array(load_img(os.path.join(dir, filename))))
  return ims
train_ims = read_images('data/train/images')
test_ims = read_images('data/test/images')
print(f'Read {len(train_ims)} training images and {len(test_ims)} testing images.')


print('\n--- Processing questions...')
texts = list(map(lambda q: q[0], all_qs))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print(f'Found {len(word_index)} words total')


