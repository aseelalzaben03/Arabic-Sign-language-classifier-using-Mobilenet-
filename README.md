# Arabic-Sign-language-classifier-using-Mobilenet-

This application is designed to recognize and classify hand signs from Arabic Sign Language using deep learning techniques. It is built upon a dataset consisting of labeled images, where each image represents a hand gesture corresponding to one of the Arabic alphabet letters.

The core functionality of the application allows users to upload  an image showing a hand gesture representing an Arabic letter. Once the image is provided, the trained deep learning model processes it and predicts which Arabic letter the gesture corresponds to.

Key Features: Input Image Handling: Users can upload an image from their device .

Preprocessing Pipeline: The input image undergoes several preprocessing steps such as resizing, normalization ,ect.. to match the format used during model training.

Model Inference: A MobileNet model, trained on a dataset of Arabic Sign Language images, is used to perform classification. The model outputs the most likely Arabic letter associated with the given hand sign.

User Interface: The app includes a simple and intuitive interface to facilitate interaction, making it easy for users of all ages to use the system effectively.

Prediction Output: After analysis, the predicted Arabic letter is displayed on the screen, possibly along with a confidence score indicating the model's certainty.

Dataset: The dataset used for training contains Hundreds of images of hand gestures, each labeled with a corresponding Arabic letter. The images cover various lighting conditions, backgrounds, and hand orientations to help the model generalize better in real-world scenarios. Dataset Link:https://www.kaggle.com/datasets/muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset

Objective: The goal of this project is to assist in communication between hearing-impaired individuals and others by providing an automatic recognition tool for Arabic Sign Language letters. It can also serve as an educational resource for people who want to learn and practice Arabic Sign Language.

