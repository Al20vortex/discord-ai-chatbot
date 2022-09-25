This is a project where I built a conversational discord bot for me and my friends. It works through a seq2seq model using LSTMs and tensorflow, 
Glove word vector data, and the movie conversations dataset. Due to file size limitations on GitHub, I couldn't upload the working model or datasets. 
To get the model working, you will need to create a discord bot and insert the token in main.py, in the function below the TODO INSERT BOT TOKEN. 
Next you'll need to download the Glove word embeddings from here (https://nlp.stanford.edu/projects/glove/) and put the files (glove.6B.50d.txt, etc) in a 
folder called Glove in the root project directory. Next make a folder called corpus, and in there insert the files 
movie_conversations.txt and movie_lines.txt which can be downloaded from the Cornell Movie Dialogues corpus found here:
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

After that, have fun! Try experimenting with different parameters, changing the size of the dataset for training (I found using only a quarter of 
the dataset was sufficient, otherwise I run out of ram on my laptop :/ ). 
