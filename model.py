from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM, Dropout, Multiply
from constants import *

def build_model(im_shape, vocab_size, num_answers):
  # The CNN
  im_input = Input(shape=im_shape)
  x1 = Conv2D(8, 3, padding='same')(im_input)
  x1 = MaxPooling2D()(x1)
  x1 = Conv2D(8, 3, padding='same')(im_input)
  x1 = Flatten()(x1)
  x1 = Dense(64, activation='tanh')(x1)

  # The RNN
  q_input = Input(shape=(MAX_QUESTION_LEN,))
  x2 = Embedding(vocab_size, 32)(q_input)
  x2 = LSTM(32)(x2)
  x2 = Dense(64, activation='tanh')(x2)

  # Merge -> output
  out = Multiply()([x1, x2])
  out = Dense(num_answers, activation='softmax')(out)

  model = Model(inputs=[im_input, q_input], outputs=out)
  model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model
