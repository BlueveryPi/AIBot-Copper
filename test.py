import torch
import asyncio
import nextcord
import asyncio
import random
import math
import numpy as np
import pandas as pd
from nextcord.ext import commands
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

print("loading tokenizer...")
TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
print("tokenizer loaded, loading model...")

class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q + \
                                          self.sent_token + sentiment)   
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv(r'D:\anaconda3\envs\kogpt\KoGPT2-chatbot\Chatbot_data\ChatbotData.csv')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    async def chat(self, sent='0', q=''):
        tok = TOKENIZER
        with torch.no_grad():
            q = q.strip()
            a = ''
            i=0
            while 1:
                i+=1
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = model(input_ids)
                
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                if i>50:
                    return ""
                a += gen.replace('▁', ' ')
            return a.strip().replace("<unused3>", "\n").replace("<unk>", "")

model = KoGPT2Chat.load_from_checkpoint('KoGPT2-chatbot/model_chp/model_-last.ckpt')
print("model loaded, initalizing bot...")

def join(n, end=" "):
    txt=""
    for t in n:
        if n[-1]!=t:
            txt+=str(t)+end
        else:
            txt+=str(t)
    return txt

ok=["901127638851129354"]

token="???"
game = nextcord.Game("인공지능 모드 작동")
bot=commands.Bot(command_prefix="알럽파 ", status=nextcord.Status.online, activity=game)
armed=False
recents=[]

@bot.event
async def on_ready():
    await bot.change_presence(activity=game, status=nextcord.Status.online)
    print("ready")

@bot.command(aliases=["조용", "닥쳐", "안닥쳐?", "아가리", "전원끄기", "poweroff"])
async def 쉿(ctx):
    global armed
    if armed:
        armed=False
        msg = await ctx.channel.send("sudo poweroff")
        txt=msg.content
        for i in range(5):
            await msg.edit(txt+".")
            txt=txt+"."
            await asyncio.sleep(0.3)
        await msg.edit(txt+"done")

@bot.command(aliases=["켜져라", "부팅", "bootup", "전원온", "전원켜기", "말해"])
async def 얍(ctx):
    global armed
    if not armed:
        armed=True
        msg = await ctx.channel.send("beep")
        txt = msg.content
        for i in range(5):
            await msg.edit(txt+".")
            txt=txt+"."
            await asyncio.sleep(0.3)
        await msg.edit(txt+"bootup complete")

@bot.event
async def on_message(message):
    if str(message.author.id) != "823793972140048406":
        if str(message.channel.id) in ok:
            global recents
            '''
            recents.append(message.content)
            if len(recents)>50:
                recents.pop(0)
            '''
            recents = await message.channel.history(limit=50).flatten()
            for i, recent in enumerate(recents):
                recents[i]=recent.content

            #print(recents)

            ctx = await bot.get_context(message)
            if ctx.valid:
                await bot.process_commands(message)
            else:
                if armed:
                    msgc=await model.chat(join(message.content, end="<unused3>"))
                    async with message.channel.typing():
                        wait=math.ceil(abs(0.3*len(msgc)+random.random()-0.5))
                        #await asyncio.sleep(wait)
                    if msgc!="":
                        await message.channel.send(msgc)

bot.run(token)