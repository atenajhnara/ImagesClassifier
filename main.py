#ربات تلگرام با پردازش تصویر یعنی عکس مثلا حیوان یا وسیله یا میوه یا اشیا به ربات میدیم بعد اون تشخیص میده عکس چی هست

import torch
from PIL import Image
from torchvision import models , transforms , datasets
from io import BytesIO
from PIL import Image
from telegram import Update
from telegram.ext import Updater,CommandHandler,MessageHandler,filters, CallbackContext
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os



weights=MobileNet_V2_Weights.DEFAULT
model=mobilenet_v2(weights=weights)
model.eval()

preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


import json
import urllib.request

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = urllib.request.urlopen(url)
labels = json.load(response) 

#labels=datasets.ImageNet(root='./data', split='val', download=True).classes
load_dotenv()
TOKEN=os.getenv("TOKEN")

async def start(update:Update,context:CallbackContext):
    await update.message.reply_text("سلام عکس بده تا تشخیص بدم چی هست")


async def handle_photo(update,context):
    await update.message.reply_text("عکس دریافت شد دارم پردازش میکنم")
    photo_file=await update.message.photo[-1].get_file()  #بالاترین کیفیت عکس رو میگیره
    photo_bytes=await photo_file.download_as_bytearray()
    img=Image.open(BytesIO(photo_bytes))  # عکس رو از بایت ها به فرمت قابل پردازش تبدیل میکنه
    input_tensor=preprocess(img).unsqueeze(0)


    with torch.no_grad():  #محاسبه گرادیان نمیکنه فقط پیش بینی میکنه
        outputs=model(input_tensor)
    probabilities=torch.nn.functional.softmax(outputs[0],dim=0)
    top_prob,top_catid=torch.topk(probabilities,1)


    class_name=labels[top_catid[0]] #شماره کلاس
    confidence=top_prob[0].item()*100  #احتمال کلاس
    await update.message.reply_text(f"احتمال بالا:{class_name} ({confidence:.2f}%)")

def main():
    
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",start))
    app.add_handler(MessageHandler(filters.PHOTO,handle_photo))

    app.run_polling()  #ربات شروع به گوش دادن پیام ها میکنه

if __name__=="__main__":
    main()







    



