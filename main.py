import discord
from chatbot import Chatbot

# initialize
bot = Chatbot()
# load in corpus, train chatbot if necessary
bot.load_corpus_and_train()
# get test response on init
bot.get_response('Hi there')
client = discord.Client()


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    else:
        res = bot.get_response(message.content)
        print(res)
        await message.channel.send(res)
        return

# TODO ENTER BOT TOKEN HERE
client.run('Insert Bot Token Here')