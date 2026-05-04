# HYPERPARAMETER – TUNED CNN MODEL FOR ACCURATE FACE MASK DETECTION

> Real-time face mask detection powered by a hyperparameter optimized CNN.

Detects whether a person is wearing a face mask in **real-time** using a live webcam feed. Draws a 🟢 green box for masked faces and a 🔴 red box for unmasked ones instantly.

---

## Highlights

- **99.59% accuracy** on the test set 
- Works under **low-light conditions**
- Hyperparameter-tuned via **Keras Tuner Random Search** over 400 trials
- Real-time inference using **OpenCV Haar Cascade** for face detection

---

## Dataset

[Face Mask ~12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) from Kaggle

Structure of Dataset 

```bash
dataset/
  Train/
    WithMask/
    WithoutMask/
  Validation/
    WithMask/
    WithoutMask/
  Test/
    WithMask/
    WithoutMask/
```

| Split      | With Mask | Without Mask |
|------------|-----------|--------------|
| Train      | 5,000     | 5,000        |
| Validation | 400       | 400          |
| Test       | 483       | 509          |

---

## Setup & Run

```bash
git clone https://github.com/Tata-Umesh/Face_Mask_Detection.git
cd Face_Mask_Detection
```

1. Download the Dataset from Kaggle and place it in the folder.
2. Train the Model : Open and run the `train.ipynb` notebook cell by cell. This will run hyperparameter tuning and save the best model weights locally.
3. Run Real-Time Detection
```bash
python detect.py
```
Make sure your webcam is connected.
---

## References

1. Kanavos et al. (2024). *Real-Time Detection of Face Mask Usage Using Convolutional Neural Networks.* Computers, 13, 182.
