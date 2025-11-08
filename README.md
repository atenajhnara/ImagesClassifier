# 🤖 ImagesClassifier Telegram Bot | ربات تلگرام تشخیص تصویر

A smart Telegram bot that receives an image and classifies it using MobileNetV2.  
The bot predicts the object in the image (animal, fruit, object, etc.) and sends the top prediction back to the user.

ربات تلگرامی هوشمند که عکس دریافت می‌کند و با استفاده از مدل MobileNetV2 کلاس تصویر را پیش‌بینی می‌کند.  
این ربات می‌تواند اشیاء، حیوانات، میوه‌ها و … را تشخیص دهد و پیش‌بینی را برای کاربر ارسال کند.

---

## 🧠 Technologies Used | تکنولوژی‌های استفاده‌شده

- Python 3.10+  
- PyTorch & torchvision (مدل پیش‌آموزش دیده MobileNetV2)  
- Pillow (PIL) (پردازش تصویر)  
- python-telegram-bot (ساخت ربات تلگرام)  
- dotenv (مدیریت کلیدهای محیطی)

---

## ⚙️ How It Works | نحوه کار

1. User sends a photo to the Telegram bot.  
2. Bot preprocesses the image (resize, crop, normalize).  
3. Bot predicts the class using MobileNetV2.  
4. Bot sends the top prediction and confidence percentage to the user.

مراحل کار:  
1. کاربر عکس ارسال می‌کند  
2. ربات تصویر را پردازش می‌کند (Resize, CenterCrop, Normalize)  
3. پیش‌بینی کلاس با مدل MobileNetV2 انجام می‌شود  
4. ربات پیش‌بینی و درصد اطمینان را به کاربر ارسال می‌کند

---

## 🧩 Key Code Structure | ساختار اصلی کد

```python
# Load MobileNetV2 pretrained model
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load ImageNet labels
labels = ...  # from JSON or dataset

# Telegram Handlers
async def start(update, context):
    await update.message.reply_text("سلام عکس بده تا تشخیص بدم چی هست")

async def handle_photo(update, context):
    # Receive photo, preprocess, predict, send top class with confidence
    ...
