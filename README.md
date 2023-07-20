# Variety_of_GAN_Implementation
In this repository, I will implement variety of GAN model using Tensorflow framework

1. The FC_GAN is implemented by the following paper: [Generative Adversarial Model](https://arxiv.org/abs/1406.2661)
2. The DCGAN is implemented by the following paper: [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Network](https://arxiv.org/abs/1511.06434)

3. The WGAN is implemented by the following paper: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

- The paper contain a lot of math. To mark down these are thing that really make WGAN stand out compare to traditional GAN:

  - They use EM (Earth Mover) distance as the value function.
  - The Wasserstein distance is differentiable everywhere so it can reduce the optimal trap when training discriminator in traditional GAN. What i mean by optimal trap is at some points of training where the generator find the most plausible output to trick the discriminator and the discriminator realize that but the discriminator have already been trained to optimal point and saturates, this lead to vanishing gradients so that it can not escape from this optimal point. By using this EM distance, it will avoid mode collapse problem when training traditional GAN model.
  - The loss of WGAN is meaningful. The value of the loss correlate well with the visual quality of the generated samples.
  - The WGAN is more stable to train compare to traditional GAN.

4. The WGAN_GP is implemented by the following paper: [Improved Training Of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

  - WGAN requires that the discriminator(critic) must lie within space of 1-Lipschits functions, which the author in Wasserstein paper enforce through weight clipping. The author in Wasserstein GAN paper also said that weight clipping is clearly a terrible way to enforce a Lipschits constraint. If the clipping value is large, it will take longer time to train the weight to reach its limit. If the clipping value is small, this can lead to vanishing gradients when the number of layers is big or batch normalization is not used.
  - In this paper, the author proposed using gradient penalty constraint instead of weight clipping.
  - The main idea is based on this phrase in the paper: Any differentiable functions is 1-Lipschits if and only if it has gradient norm at most 1 everywhere.
  - From the idea above we will add the constraint to loss function of the critic force the gradient norm of the critics with respect to its input is at most 1.

5. Pix2pix paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

  - Image-to-Image translation means translating one possible representation of the scene into another. It based on the fact that we have to predict pixels from pixels. And CNN is becoming a workhorse behind all of the image predictions problem. We can take a naive approach that use Euclidian distance as a loss function between predicted pixel and ground-truth pixel but the result will be blurry. And "coming up with loss functions that force the CNN to do what we really want – e.g., output sharp, realistic images is an open problem and generally requires expert knowledge".
  - In the paper, the authors explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model. This makes cGANs suitable for image-to-image translation tasks, where condition on an input image and gen- erate a corresponding output image.
  - I think the part where i got confused when reading this paper the first time is the discriminator the author used which is the patchGAN. After doing research i come down with some notes:
    - The difference between patchGAN and regular GAN discriminator is that regular GAN maps from a 256x256 image to single scalar output, which signifies "real" or "fake", whereas the patchGAN maps from 256x256 images to an array output X, where each X_ij signifies whether patch ij in the image is "real" or "fake".
    - And the patch is just a neuron in a convolutional net so we can trace back for its receptive field in the input (The <i> receptive field </i> is defined as a region in the input space that a particular CNN's feature is looking at). And the size NxN array is defined to be the size of this receptive field that one patch in the output can observe.
    - In this paper, the author choose N=70.

6. CycleGAN paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

  - Image-to-image translation's goal is learning the mapping between the input image and output image using training set of aligned image pairs. However, in real world, for many tasks paired training dataset will not available. So this paper, the author present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples.
  - The term 'cycle' comes from forward and backward mapping. We will train the mapping function G : X -> Y and mapping function F : Y -> X using adversarial loss function and the new loss term which is cycle consistency loss (For more information you can find it in the paper).
  - The idea of cycle consistency loss : With the network that have large capacity we will have indefinite way to map input image to output image that in the target domain (in practice, it will often lead to mode collapse problem as the author said), so to ensure that we get the desired output image y from input image x, the author solve this problem by arguing that the mapping function should be cycle consistency (The inverse mapping F(y) should produce the image that approximately the same as original input image x).
  - And also the things that i don't understand when reading this paper is why do we need to train the cycle consistency loss in both way forward and backward but not in one direction. The reason is very simple because train the model in both way produce the better result compare to just train in one direction.