from __future__ import print_function, division
from builtins import range
import os
import traceback
from markupsafe import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class Chatbot:
    def __init__(self):
        self.MAX_WORDS = 60000
        self.MAX_LINE_LENGTH = 12
        self.EMBEDDING_DIM = 50
        self.encoder_states = None
        self.decoder_lstm = None
        self.decoder_embedded = None
        self.decoder_model = None
        self.full_tokenizer = None
        self.encoder_model = None
        self.tokenizer_answers = None
        self.dense = None
        self.tokenizer_questions = None
        self.model = self.load_model_if_found()
        self.decoder_states = None
        
    def load_model_if_found(self):
        try:
            print('Attempting to load model')
            return tf.keras.models.load_model('saved_model')
        except:
            print('No model found, setting model to None')
            return None

    def load_corpus_and_train(self):
        lines = open('corpus/movie_lines.txt', encoding='utf-8',
                    errors='ignore').read().split('\n')
        # print(lines[-1])
        conversations = open('corpus/movie_conversations.txt',
                            encoding='utf-8', errors='ignore').read().split('\n')

        for i in range(0, len(conversations) - 1):
            conversation = conversations[i].split(' +++$+++ ')[-1][1:-1]
            conversations[i] = conversation.replace(
                "'", "").replace(",", "").split()

        dialogue = {}
        for line in lines:
            dialogue[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

        questions = []
        answers = []

        for conversation in conversations:
            # only train on a fraction of the data
            for i in range(len(conversation)//4):
                questions.append(dialogue[conversation[i]])
                answers.append(dialogue[conversation[i+1]])

        del(dialogue)

        # convert text to lowercase
        for line in questions:
            line.lower()
        for line in answers:
            line.lower()

        for i in range(len(answers)):
            answers[i] = '<sos> ' + answers[i] + ' <eos>'

        # self.full_tokenizer = Tokenizer(
        #     num_words=self.MAX_WORDS, oov_token='<oov>', filters='')
        # self.full_tokenizer.fit_on_texts(questions + answers)

        # print(full_tokenizer.word_index)

        self.tokenizer_questions = Tokenizer(
            num_words=self.MAX_WORDS, oov_token='<oov>'
        )
        self.tokenizer_questions.fit_on_texts(questions)
        questions_sequences = self.tokenizer_questions.texts_to_sequences(questions)
        
        del(questions)
        
        self.tokenizer_answers = Tokenizer(
            num_words=self.MAX_WORDS, filters="!'#()*+,-./:;?@^_`{|}~")
        self.tokenizer_answers.fit_on_texts(answers)
        answers_sequences = self.tokenizer_answers.texts_to_sequences(answers)

        del(answers)
        
        # determine maximum length
        # max_length = max(len(s) for s in questions_sequences)

        questions_sequences = pad_sequences(
            questions_sequences, maxlen=self.MAX_LINE_LENGTH, padding='post', truncating='post')
        answers_sequences = pad_sequences(
            answers_sequences, maxlen=self.MAX_LINE_LENGTH, padding='post', truncating='post')

        # make the outputs for teacher forcing
        answers_sequences_outputs = []
        for i in answers_sequences:
            answers_sequences_outputs.append(i[1:])

        answers_sequences_outputs = pad_sequences(answers_sequences_outputs, maxlen=self.MAX_LINE_LENGTH, padding='post', truncating='post')

        answers_sequences_outputs = to_categorical(answers_sequences_outputs, self.MAX_WORDS)

        # load pretrained word vectors
        word2vec = {}
        with open(os.path.join('./Glove/glove.6B.%sd.txt' % self.EMBEDDING_DIM), encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        del(f)
        
        print(self.model)
        if not self.model:
            # prepare embedding matrix
            embedding_matrix = np.zeros((self.MAX_WORDS, self.EMBEDDING_DIM))
            for word, i in self.tokenizer_questions.word_index.items():
                if i < self.MAX_WORDS:
                    embedding_vector = word2vec.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all zeros.
                        embedding_matrix[i] = embedding_vector

            # BUILD THE MODEL 

            # shape is 50 (MAX_LINE_LENGTH) because thats the length of the input we pass in
            encoder_input = Input(shape=(self.MAX_LINE_LENGTH, ))
            decoder_input = Input(shape=(self.MAX_LINE_LENGTH, ))

            del(word2vec)

            # create embedding layer
            embedding_layer = Embedding(
                self.MAX_WORDS,
                self.EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=self.MAX_LINE_LENGTH,
                # trainable=True
            )

            del(embedding_matrix)

            encoder_embedded = embedding_layer(encoder_input)

            # 400 LSTM cells, return_sequences=true means that it returns the hidden state output at every time step, which is necessary in order to stack lstm layers
            # return state is true because we need it for the decoder. returns the last state as well as the output at the last time step.
            encoder_lstm = LSTM(100, return_sequences=True, return_state=True)
            encoder_output, h_enc, c_enc = encoder_lstm(encoder_embedded)
            self.encoder_states = [h_enc, c_enc]

            self.decoder_embedded = embedding_layer(decoder_input)
            self.decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
            decoder_output, decoder_h, decoder_c = self.decoder_lstm(self.decoder_embedded, initial_state=self.encoder_states)

            self.dense = Dense(self.MAX_WORDS, activation='softmax')

            dense_output = self.dense(decoder_output)

            self.model = Model([encoder_input, decoder_input], dense_output)

            print('Compiling model')
            self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

            print('Training model')
            # 30 epochs is too little, 100 should be a good number
            self.model.fit([questions_sequences, answers_sequences], answers_sequences_outputs, epochs=20, verbose=1)
            print('Finished training')
            self.model.save('saved_model')
        else:
            print('Using found model')


    # Took heavy inspiration (AKA) from https://github.com/Pawandeep-prog/chatbot/blob/main/inference.py
    def get_response(self, message):
        try:
            embedding_layer = self.model.get_layer('embedding')

            encoder_input = Input(shape=(self.MAX_LINE_LENGTH, ))
            decoder_input = Input(shape=(self.MAX_LINE_LENGTH, ))
            encoder_embedded = embedding_layer(encoder_input)
            self.decoder_embedded = embedding_layer(decoder_input)

            encoder_lstm = self.model.get_layer('lstm')
        
            encoder_output, h_enc, c_enc = encoder_lstm(encoder_embedded)
            self.encoder_states = [h_enc, c_enc]
            
            self.encoder_model = Model([encoder_input], self.encoder_states)
            
            decoder_state_input_h = Input(shape=(100,))
            decoder_state_input_c = Input(shape=(100,))
            
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            
            self.decoder_lstm = self.model.get_layer('lstm_1')
            
            
            decoder_outputs, dec_h, dec_c = self.decoder_lstm(self.decoder_embedded,
                                                                initial_state=decoder_states_inputs)

            self.decoder_states = [dec_h, dec_c]

            self.decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + self.decoder_states)
            
            self.dense = self.model.get_layer('dense')

            print('Generating response for input: ', message)
                        
            done = False
            while done != True:
                preprocessed_msg = message.lower().translate(str.maketrans('', '', string.punctuation))

                txt = []
                lst = []
                for y in preprocessed_msg.split():
                    try:
                        lst.append(self.tokenizer_questions.word_index[y])
                    except:
                        lst.append(self.tokenizer_questions.word_index['<oov>'])
                txt.append(lst)
                txt = pad_sequences(txt, self.MAX_LINE_LENGTH, padding='post')

                state = self.encoder_model.predict(txt)

                # empty target sequence looking like [0]
                empty_target_seq = np.zeros((1, 1))

                empty_target_seq[0, 0] = self.tokenizer_answers.word_index['<sos>']

                stop_condition = False
                decoded_translation = ''

                while not stop_condition:
                    dec_outputs, h, c = self.decoder_model.predict(
                        [empty_target_seq] + state
                    )
                    # run the decoder output through the dense
                    decoder_concat_input = self.dense(dec_outputs)

                    # index 0 doesn't exist in the word index so we start add 2 to the chosen index to compensate for that, 
                    sampled_word_index = np.argmax(decoder_concat_input[0, 0, 2:-1]) + 2
                    
                    # will fail because there is no '0', that is reserved for padding
                    sampled_word = self.tokenizer_answers.index_word[sampled_word_index]
                    if sampled_word != '<eos>':
                        decoded_translation += sampled_word + ' '

                    if (sampled_word == '<eos>') or (len(decoded_translation.split()) > self.MAX_LINE_LENGTH):
                        stop_condition = True
                        done = True
                    
                    # empty target sequence looking like [0]
                    empty_target_seq = np.zeros((1, 1))
                    empty_target_seq[0, 0] = sampled_word_index
                    state = [h, c]
            return decoded_translation
        except Exception as e:
            print(traceback.format_exc())
            print('exception:', e)
            return 'Encountered error getting response, see console for details'
