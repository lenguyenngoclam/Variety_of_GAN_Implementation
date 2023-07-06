# Variety_of_GAN_Implementation
In this repository, I will implement variety of GAN model using Tensorflow framework

1. The FC_GAN is implemented by the following paper: [Generative Adversarial Model](https://arxiv.org/abs/1406.2661)
2. The DCGAN is implemented by the following paper: [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Network](https://arxiv.org/abs/1511.06434)
3. The WGAN is implemented by the following paper: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- The paper contain a lot of math. To mark down these are thing that really make WGAN stand out compare to traditional GAN:
  + They use EM (Earth Mover) distance as the value function.
  + The Wasserstein distance is differentiable everywhere so it can reduce the optimal trap when training discriminator in traditional GAN. What i mean by optimal trap is at some points of training where the generator find the most plausible output to trick the discriminator and the discriminator realize that but the discriminator have already been trained to optimal point and saturates, this lead to vanishing gradients so that it can not escape from this optimal point. By using this EM distance, it will avoid mode collapse problem when training traditional GAN model.
  + The loss of WGAN is meaningful. The value of the loss correlate well with the visual quality of the generated samples.
  + The WGAN is more stable to train compare to traditional GAN.
4.  The WGAN_GP is implemented by the following paper: [Improved Training Of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
  + WGAN requires that the discriminator(critic) must lie within space of 1-Lipschits functions, which the author in Wasserstein paper enforce through weight clipping. The author in Wasserstein GAN paper also said that weight clipping is clearly a terrible way to enforce a Lipschits constraint. If the clipping value is large, it will take longer time to train the weight to reach its limit. If the clipping value is small, this can lead to vanishing gradients when the number of layers is big or batch normalization is not used.
  + In this paper, the author proposed using gradient penalty constraint instead of weight clipping.
  + The main idea is based on this phrase in the paper: Any differentiable functions is 1-Lipschits if and only if it has gradient norm at most 1 everywhere.
  + From the idea above we will add the constraint to loss function of the critic force the gradient norm of the critics with respect to its input is at most 1.