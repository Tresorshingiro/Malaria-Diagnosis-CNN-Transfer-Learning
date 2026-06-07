# Malaria Diagnosis with CNNs & Transfer Learning

Automated detection of malaria from microscopic blood-smear cell images using Convolutional
Neural Networks (CNNs) and transfer learning in TensorFlow/Keras. This is a group project for the
Machine Learning course at **African Leadership University (ALU)**.

Each cell image is classified as one of two classes: **Parasitized** (infected) or **Uninfected**.

---

## The Problem

Malaria is a life-threatening disease caused by *Plasmodium* parasites transmitted through the
bites of infected *Anopheles* mosquitoes. Diagnosis traditionally relies on a trained
microscopist manually examining stained blood smears — a slow, labour-intensive process that is
hard to scale in the low-resource regions where malaria is most prevalent. Automating this
classification with CNNs can support faster, more consistent, and more accessible screening.

This project simulates a real-world medical-AI workflow: systematic experimentation, rigorous
evaluation, and clear communication of results across a team.

---

## Dataset

We use the public **NIH Malaria Cell Images** dataset (National Library of Medicine, in
collaboration with Mahidol-Oxford and Chittagong Medical College Hospital).

- **~27,558 images**, balanced between `Parasitized` and `Uninfected`.
- Downloaded automatically inside each notebook:

```bash
wget https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
unzip cell_images.zip
```

The extracted `cell_images/` folder and the `.zip` are **not** committed to the repo (they are
regenerated on download). The current `.gitignore` only excludes `venv`; consider adding
`cell_images/` and `*.zip` to keep the repo lightweight.

**Citation:** Rajaraman, S., et al. (2018). *Pre-trained convolutional neural networks as feature
extractors toward improved malaria parasite detection in thin blood smear images.* PeerJ.

---

## Models

The team builds **five distinct models**. Every model owner runs **at least seven experiments**
and reports accuracy, precision, recall and F1-score, plus learning curves, confusion matrices,
and ROC/AUC curves.

| # | Model | Notebook | Type | Owner |
|---|-------|----------|------|-------|
| 1 | Baseline CNN | `Baseline_CNN__Model_1.ipynb` | Simple CNN (group-built) | Heroine|
| 2 | Advanced CNN | `Advanced_CNN.ipynb` | Deeper CNN + augmentation/dropout | Celine|
| 3 | VGG16 | `model3_vgg16.ipynb` | Pretrained transfer learning | Emmanuel |
| 4 | ResNet50 | `model4_resnet50.ipynb` | Pretrained transfer learning | Tresor |
| 5 | **MobileNetV2** | `model5_mobilenetv2.ipynb` | Pretrained transfer learning | **Jacques** |

> Update the **Owner** column as teammates confirm their models.

---

## Shared Methodology

All notebooks follow the same protocol so results are directly comparable:

- **Data pipeline:** `tf.keras.utils.image_dataset_from_directory` with an 80/20 train/validation
  split, `IMG_SIZE = 128×128`, `BATCH_SIZE = 32`, `SEED = 42`, and a cached/shuffled/prefetched
  `tf.data` pipeline.
- **Output head:** single sigmoid unit with `binary_crossentropy` loss.
- **7-experiment protocol:** each experiment varies one design dimension (learning rate,
  optimizer, head capacity, dropout, data augmentation, fine-tuning depth) so its effect can be
  isolated.
- **Evaluation:** a consolidated metrics table (Accuracy / Precision / Recall / F1) plus, for each
  experiment, learning curves, a confusion matrix, and a ROC/AUC curve.

**MobileNetV2 note:** Model 5 uses MobileNetV2's official `preprocess_input` (scales pixels to
`[-1, 1]`) instead of `Rescaling(1/255)`, since that is the input range the pretrained weights
expect. During fine-tuning, BatchNormalization layers are kept frozen / in inference mode to
preserve their pretrained statistics.

---

## How to Run

The notebooks were developed for **Google Colab** with a GPU runtime (the full dataset is too
large to train comfortably on CPU).

1. Open the notebook (e.g. `model5_mobilenetv2.ipynb`) in Google Colab.
2. Set **Runtime → Change runtime type → GPU**.
3. **Runtime → Run all.** The dataset downloads automatically on the first run.

---

## Repository Structure

```
.
├── Baseline_CNN__Model_1.ipynb   # Model 1 — baseline CNN
├── Advanced_CNN.ipynb            # Model 2 — advanced CNN
├── model3_vgg16.ipynb            # Model 3 — VGG16 transfer learning
├── model4_resnet50.ipynb         # Model 4 — ResNet50 transfer learning
├── model5_mobilenetv2.ipynb      # Model 5 — MobileNetV2 transfer learning
├── activity.txt                  # Assignment brief
└── README.md
```
