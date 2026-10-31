# Indian Currency Detection Model

A Deep Learning-based Indian Currency Detection System designed to identify and classify Indian currency notes from images. The project uses image classification techniques to recognize different denominations and provides audio feedback using Text-to-Speech functionality.

## Project Overview

This project aims to develop an automated currency recognition system that can identify Indian currency denominations using a trained deep learning model.

The system processes currency note images, classifies them into their respective denominations, and provides audio output to announce the detected currency.

## Features

- Indian currency denomination detection and classification.
- Image-based currency recognition.
- Deep Learning model training using TensorFlow.
- Text-to-Speech functionality using gTTS.
- Audio playback of detected currency denominations.
- Support for training with a custom currency dataset.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep Learning model development |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Pillow (PIL) | Image processing |
| gTTS | Text-to-Speech conversion |
| playsound | Audio playback |

## Dependencies

Install the required Python libraries:

```bash
pip install tensorflow numpy matplotlib gTTS playsound pillow
```

Alternatively, install dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

##  Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd <PROJECT_FOLDER>
```

### 2. Collect the Dataset

Gather images of Indian currency notes in JPG or PNG formats.

Organize the dataset into separate folders for each denomination.

Example:

```text
dataset/
│
├── 10-rupees/
├── 20-rupees/
├── 50-rupees/
├── 100-rupees/
├── 200-rupees/
└── 500-rupees/
```

### 3. Update the Dataset Path

Update the dataset directory path in the training script according to your local system.

Ensure that the dataset is correctly structured and accessible.

### 4. Train the Model

Run the training script to train the deep learning model using the prepared dataset.

```bash
python train.py
```

Adjust the model architecture and hyperparameters if required.

### 5. Run Currency Detection

After training, run the prediction script to classify currency note images.

```bash
python predict.py
```

Refer to the source code for the appropriate inference procedure and input requirements.

## Audio Assistance

The project uses Google Text-to-Speech (gTTS) to convert the predicted currency denomination into speech.

The generated audio is played using the playsound library, allowing users to hear the detected denomination.

## Contributors

This project was collaboratively developed by:

- **Senthil Velan** – [@Senthil-velan19](https://github.com/Senthil-velan19)
- **DUCTOU** – [@DUCTOU](https://github.com/DUCTOU)

Both contributors collaborated on the development of the Indian Currency Detection Model.

## Future Improvements

- Improve classification accuracy using larger and more diverse datasets.
- Enhance recognition under different lighting conditions.
- Optimize the model for real-time currency detection.
- Develop a user-friendly mobile application.
- Improve accessibility features for visually impaired individuals.

## Contributions

Contributions, suggestions, and improvements are welcome. Feel free to open an issue or submit a pull request.

---

**Developed as a collaborative Deep Learning project.**
