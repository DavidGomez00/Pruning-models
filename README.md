# Pruning models
 El objetivo de este repositorio es hacre un experimento de fine-tune a una red neuronal convolucional para luego aplicarle model-pruning.

 El modelo base utilizado para realizar la clasificación binaria entre perros y gatos es VGG-16: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html.
 El dataset utilizado es Cats and Dogs image classification: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification/data.

 Para el entrenamiento del modelo, se ha modificado la última capa completamente conectada de la red, dejando la capa de salida en una única neurona. Se ha realizado un 5-fold cross validation.
