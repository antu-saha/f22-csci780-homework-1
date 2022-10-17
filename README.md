# f22-csci780-homework-1

Overview:
=========
I designed a convolutional neural network (CNN).
I applied dropout, ADAM optimizer, max-pooling, tensorboard, wandb, Xavier Initialization, and batch normalization in my model architecture.
For the training and testing of my model, I used CIFAR 10 dataset.  
Then I replace my model with the pre-trained ResNet-18 model and use the same dataset to see the performance. 
Finally, I implemented Data Parallel Training (DDP) for the training purpose. I do so for both the CNN and ResNet models.

Usage:
=====
1. For single GPU training:
		Run the main.py file.
				This file will ask for a user input like below:
				
				Please select one from the following menu:
				For CNN, press: 1
				For ResNet-18, press: 2
				Enter your choice:
				
				If you enter 1, then the model will be initilized with the CNN model.
				If you enter 2, then the model will be initilized with the ResNet-18 model.
				
2. For mulple GPU training:
		Run the ddp_training.py file.
				This file will also work as before. 
				But in this time, it will use multile GPU (if available) for the training purpose.  
