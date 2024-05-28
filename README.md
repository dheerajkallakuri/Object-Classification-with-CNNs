# Object Classification with Convolutional Neural Networks and Comparisons

## Overview

This project aims to implement and evaluate various iterations of Convolutional Neural Networks (CNNs) for object classification on the CIFAR-10 dataset. The dataset consists of 60,000 images categorized into 10 distinct classes. The project is divided into four phases, each progressively enhancing the CNN architecture and evaluating its performance.

## Project Phases

### Phase 1: Basic CNN Construction
- Construct a simple CNN with a few convolutional layers followed by fully-connected layers.
- Train the model on the CIFAR-10 dataset.
- Report the results and performance metrics.

### Phase 2: Adding Pooling Layers
- Enhance the CNN by adding appropriate pooling layers after the convolutional layers.
- Train the modified model on the CIFAR-10 dataset.
- Observe and report the changes in performance.

### Phase 3: Incorporating Dropout
- Incorporate dropout layers into the CNN architecture to reduce overfitting.
- Train the updated model on the CIFAR-10 dataset.
- Report the benefits of using dropout and its impact on model performance.

### Phase 4: Using ShuffleNet Architecture
- Implement a well-known CNN architecture, ShuffleNet.
- Train ShuffleNet on the CIFAR-10 dataset.
- Report the results and compare the performance with previous models.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. You may choose an optimal train-validation-test split and discuss your configuration in the report.

<img width="402" alt="dataset" src="https://github.com/dheerajkallakuri/Object-Classification-with-CNNs/assets/23552796/f43c1320-cf1a-4916-b32c-3f90876ef32b">

## Usage

1. **Train Basic CNN:**
   ```bash
   python projcnn1.py
   ```

2. **Train CNN with Pooling Layers:**
   ```bash
   python projcnn2.py
   ```

3. **Train CNN with Dropout:**
   ```bash
   python projcnn3.py
   ```

4. **Train ResNet50:**
   ```bash
   python projcnnResNet50_TL.py
   ```
   
5. **Train ShuffleNet:**
   ```bash
   python projcnnShuffleNet_TL.py
   ```

## Results and Reporting

- **Phase 1: Basic CNN**
 <img width="534" alt="part1" src="https://github.com/dheerajkallakuri/Object-Classification-with-CNNs/assets/23552796/922ed681-9c95-4538-be64-65fca52dc0b7">


- **Phase 2: CNN with Pooling Layers**
 <img width="535" alt="part2" src="https://github.com/dheerajkallakuri/Object-Classification-with-CNNs/assets/23552796/94e75e36-70aa-4c87-9cef-0be074757f60">


- **Phase 3: CNN with Dropout**
 <img width="532" alt="part3" src="https://github.com/dheerajkallakuri/Object-Classification-with-CNNs/assets/23552796/af7f140e-2da6-45fc-ab4e-10f3986d1aa9">


- **Phase 4: ResNet50 and ShuffleNet**
 <img width="526" alt="part4a" src="https://github.com/dheerajkallakuri/Object-Classification-with-CNNs/assets/23552796/6b4bc906-9947-4e8b-9b8e-45951050af6d">

 <img width="534" alt="part4b" src="https://github.com/dheerajkallakuri/Object-Classification-with-CNNs/assets/23552796/684bfbb2-ad2e-45ca-9c45-da366f43a494">
