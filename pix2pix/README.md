# Pix2pix

- I choose the map-to-aerials translation for testing the model. The dataset link is [here](https://www.kaggle.com/datasets/alincijov/pix2pix-maps)

- In the original paper the author train the model for 200 epochs but because of the lack of training resources i am just be able to train it for 70 epochs. The result looks good when translating tile in the map images but about the model still can not capture the highway part in the map images.

- For the pretrained weight of the map-aerials translation model you can download it [here](https://github.com/aladdinpersson/Machine-Learning-Collection/releases/download/1.0/Pix2Pix_Weights_Satellite_to_Map.zip)

|1st Column: Input / 2nd Column: Target / 3rd Column: Generated|
|:---:|
|![Pix2pix training process](pix2pix_training_process.gif)|