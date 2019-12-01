word_index = {'the': 1, 'shape': 2, 'is': 3, 'a': 4, 'image': 5, 'what': 6, 'in': 7, 'does': 8, 'contain': 9, 'present': 10, 'there': 11, 'not': 12, 'color': 13, 'no': 14, 'triangle': 15, 'circle': 16, 'rectangle': 17, 'of': 18, 'green': 19, 'brown': 20, 'teal': 21, 'gray': 22, 'blue': 23, 'black': 24, 'yellow': 25, 'red': 26}
answers = ['teal', 'yellow', 'triangle', 'no', 'yes', 'brown', 'circle', 'blue', 'black', 'red', 'gray', 'green', 'rectangle'];

function getBagOfWords(str) {
  str = str.replace(/[^\w\s]|_/g, '')
           .replace(/\s+/g, ' ')
           .replace(/[0-9]/g, '')
           .toLowerCase();
  
  let tokens = str.split(' ');
  let bagOfWords = Array(Object.keys(word_index).length).fill(0);
  tokens.forEach(token => {
    if (token in word_index) {
      bagOfWords[word_index[token]] += 1;
    }
  })
  return bagOfWords;
}

function scaleImageData(bigCanvas) {
  let canvas = document.getElementById('smallCanvas');
  const ctx = canvas.getContext('2d');
  ctx.scale(0.5, 0.5);
  ctx.drawImage(bigCanvas, 0, 0);        
  return canvas;
}

function getInference(imageData, questionBOW) {
  console.log('called getInference');
  tf.loadLayersModel('https://atlantic-question.glitch.me/assets/model.json').then(model => {
    console.log('Successfully loaded weights');
    let imageTensor = tf.browser.fromPixels(imageData, 3);
    imageTensor = imageTensor.expandDims(0);
    imageTensor.print();
    let questionTensor = tf.tensor(questionBOW);
    questionTensor = questionTensor.expandDims(0);
    let output = model.predict([imageTensor, questionTensor]);
    let finalIdx = output.argMax(1).arraySync();
    console.log(answers[finalIdx[0]]);
  })
  .catch(err => console.log(err));
}

function onSubmit() {
  console.log('called onSubmit');
  let canvas = document.getElementById('myCanvas');
  const ctx = canvas.getContext('2d');
  let smallCanvas = scaleImageData(canvas);
  questionBOW = getBagOfWords("What color is the shape?");
  console.log('bow: ', questionBOW);
  getInference(smallCanvas, questionBOW);
}
