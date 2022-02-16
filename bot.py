import requests, nextcord, asyncio, random, time
from nextcord.ext import commands
from os import listdir
from os.path import isfile, join

token="ODIzNzkzOTcyMTQwMDQ4NDA2.YFl_7A.zYRliAd1ndJ1pdtdRyzLsW3hOtc"
game = nextcord.Game("인공지능 모드 작동")
bot=commands.Bot(command_prefix="알럽파 ", status=nextcord.Status.online, activity=game)

@bot.event
async def on_ready():
    await bot.change_presence(activity=game, status=nextcord.Status.online)
    print("ready")

@bot.event
async def on_message(ctx):
    pass

bot.run(token)