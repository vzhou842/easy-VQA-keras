from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam

def build_model(im_shape, vocab_size, num_answers, big_model):
  # The CNN
  im_input = Input(shape=im_shape)
  x1 = Conv2D(8, 3, padding='same')(im_input)
  x1 = MaxPooling2D()(x1)
  x1 = Conv2D(16, 3, padding='same')(x1)
  x1 = MaxPooling2D()(x1)
  if big_model:
    x1 = Conv2D(32, 3, padding='same')(x1)
    x1 = MaxPooling2D()(x1)
  x1 = Flatten()(x1)
  x1 = Dense(32, activation='tanh')(x1)

  # The question network
  q_input = Input(shape=(vocab_size,))
  x2 = Dense(32, activation='tanh')(q_input)
  x2 = Dense(32, activation='tanh')(x2)

  # Merge -> output
  out = Multiply()([x1, x2])
  out = Dense(32, activation='tanh')(out)
  out = Dense(num_answers, activation='softmax')(out)

  model = Model(inputs=[im_input, q_input], outputs=out)
  model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

  return model
