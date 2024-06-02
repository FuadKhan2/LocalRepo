
# DIP Lab Task 1 Documentation

Hello, everyone. In this Documentation, I'll walk you through a series of image processing tasks using a denoising dataset from Kaggle. We'll cover noise reduction, change detection, and masking techniques. Let's get started with the code.

## Code Walkthrough

### 1. Kaggle Dataset Setup and Installing Necessary Libraries



```bash
  !pip install kaggle
  !pip install opencv-python-headless

```
- I start by installing the required libraries: kaggle to download datasets and opencv-python-headless for image processing.

### 2. Mounting Google Drive

```bash
  from google.colab import drive
  drive.mount('/content/drive')

```
- I mount Google Drive to access the Kaggle credentials stored there.

### 3. Setting Up Kaggle Configuration

```bash
  ! mkdir ~/.kaggle
  !cp '/content/drive/MyDrive/Colab Notebooks/kaggle_credentials/kaggle.json' ~/.kaggle/
  ! chmod 600 ~/.kaggle/kaggle.json
  ! mkdir /content/kaggle_data

```
- I create a directory for the Kaggle configuration file and copy the kaggle.json file from Google Drive. I also set permissions and create a directory to store the dataset.

### 4. Downloading and Unzipping the Dataset

```bash
  %%shell
  ls /
  if [ ! -d "/root/.kaggle/kaggle.json" ]; then
  echo "$DIRECTORY does not exist."
  fi
  ! kaggle datasets download -p /content/kaggle_data tenxengineers/  denoising-dataset-multiple-iso-levels
  ! unzip /content/kaggle_data/denoising-dataset-multiple-iso-levels.zip -d /content/kaggle_data/denoising-dataset-multiple-iso-levels/

```
- I verify the Kaggle configuration, download the dataset, and unzip it to a specified directory.

### 5. Preparing the Image List

```bash
  import os
  import numpy as np
  import matplotlib.pyplot as plt
  import cv2
  from google.colab.patches import cv2_imshow
  from skimage import io

  txt_files = []
  for root, dirs, files in os.walk("/content/kaggle_data/denoising-dataset-multiple-iso-levels/AlphaISP - Denoising Dataset/AlphaISP - Denoising Dataset"):
      for file in files:
          if file.endswith("AlphaISP_2592x1536_8bits_Scene44.png"):
              txt_files.append(os.path.join(root, file))

```
- I import necessary libraries and create a list of paths for the specific images we will process.

### 6. Display Function

```bash
  def show_image(title, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

```
- This function helps display images with a title.

## Task Demonstration

### 7. Displaying Original Images

```bash
  for noisy_image in txt_files:
    print(noisy_image)
    plt.figure()
    plt.imshow(io.imread(noisy_image))
  plt.show()

```
- I print and display the original images from our list.

### 8. Noise Reduction using Addition
In the noise reduction section, we add random noise to the original image to simulate a noisy environment. This process involves:
```bash
  for image_path in txt_files:
    image = cv2.imread(image_path)
    show_image('Original Image', image)

    # Adding Noise
    noise = np.random.randint(0, 25, image.shape, dtype=np.uint8)
    noisy_image = cv2.add(image, noise)
    show_image('Noisy Image', noisy_image)

```
- I load and display the original image. Then, I generate random noise and add it to the original image using 'cv2.add', then display the noisy image.

### 9. Change Detection using Subtraction
In the change detection section, we identify differences between the original and a modified (dirty) image:
#### Loading the Corresponding Modified Image:
```bash
    dirty_image_path = image_path.replace("AlphaISP_2592x1536_8bits_Scene44.png", "AlphaISP_2592x1536_8bits_Scene44_dirty.png")
    if os.path.exists(dirty_image_path):
        dirty_image = cv2.imread(dirty_image_path)
        show_image('Modified Image for Change Detection', dirty_image)

```
- I generate the path for the modified image and check if it exists. If it does, I load and display it.

#### Detecting Changes:
```bash
   # Detecting Changes
        change_detected = cv2.subtract(image, dirty_image)
        show_image('Change Detected', change_detected)
    else:
        print(f"Clean image not found for {image_path}")
        change_detected = None

```
- Using 'cv2.subtract', I compute the difference between the original and modified images to highlight changes, then display the result.

### 10. Masking using Multiplication
In the masking section, we apply a binary mask to the images to isolate specific regions:
#### Creating and Applying Mask
```bash
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (700, 0), (1250, 240), 255, -1)
    show_image('Mask', mask)

```
- I create a binary mask (initially all zeros) and draw a white rectangle on it to define the region of interest, then display the mask.

#### Applying the Mask to the Noisy Image:
```bash
    masked_image = cv2.bitwise_and(noisy_image, noisy_image, mask=mask)
    show_image('Masked Noisy Image', masked_image)

```
- I use 'cv2.bitwise_and' to apply the mask to the noisy image, displaying the masked result.

#### Applying the Mask to the Modified Image (if available):
```bash
    # Applying Mask to Modified Image
    if change_detected is not None:
        masked_dirty_image = cv2.bitwise_and(dirty_image, dirty_image, mask=mask)
        show_image('Masked Modified Image', masked_dirty_image)
    break

print("Processing completed.")

```
- If a modified image is available, I apply the same mask to it and display the result.

## Alternative Approaches
To achieve similar or better results, I could explore other noise reduction techniques such as Gaussian Blur or advanced algorithms like Non-Local Means Denoising. Additionally, machine learning-based approaches using convolutional neural networks could significantly improve change detection and masking accuracy.

## Conclusion
To sum up, I've successfully demonstrated noise reduction, change detection, and masking on a denoising dataset. These techniques are fundamental in image processing and have numerous applications in fields such as photography, medical imaging, and surveillance. Thank you for watching!

## Link

[Google Colab](https://colab.research.google.com/drive/1bA1BbrfF_5cUylzzdkad6Vp-OtBHmLBP#scrollTo=nmsFgEuubDX2)


