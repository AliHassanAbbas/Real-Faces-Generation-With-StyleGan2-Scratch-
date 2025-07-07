# 🖼️ Real Faces Generation with StyleGAN2 (from Scratch)

Welcome to **Real Faces Generation with StyleGAN2**, a PyTorch project that trains StyleGAN2 from scratch to generate realistic 64x64 facial images, complete with an interactive Streamlit app.

## 📂 Project Structure
```
Real-Faces-StyleGAN2/
├── datasets/
│   └── celeba/                # Your CelebA images go here
├── checkpoints/               # Saved model weights
├── logs/                      # Generated samples during training
├── models/
│   ├── generator.py           # StyleGAN2 Generator
│   └── discriminator.py       # StyleGAN2 Discriminator
├── utils/
│   └── dataloader.py          # CelebA Dataset loader
├── train.py                   # Training script
├── viewer.py                  # Streamlit app for face generation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📥 Dataset
This project uses the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), containing **200,000+ real celebrity images**.

➡️ **Direct download link:** [Download CelebA images (Google Drive)]([https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing))

After downloading, extract the `img_align_celeba.zip` archive into:
```
datasets/celeba/
```

Your directory should look like this afterward:
```
datasets/celeba/
├── 000001.jpg
├── 000002.jpg
├── ...
```

## 🚀 Usage

1️⃣ **Install dependencies**
```
pip install -r requirements.txt
```

2️⃣ **Train the model**
```
python train.py
```

3️⃣ **Launch the Streamlit app**
```
streamlit run viewer.py
```

Enjoy generating realistic faces! 🎨
