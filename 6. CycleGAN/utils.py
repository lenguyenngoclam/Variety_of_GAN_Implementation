from IPython import display
import matplotlib.pyplot as plt
import cv2

def plot_generator_loss(G_loss):
    plt.plot(G_loss, label='Generator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def plot_loss(g_losses, d_losses):
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def generate_and_visualize(generator, test_inputs, epoch):
    output_images = generator.predict(test_inputs)

    fig = plt.figure(figsize=(20, 20))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        # Add border for seperate image
        padding = 30
        input_img = cv2.copyMakeBorder((test_inputs[i] * 127.5 + 127.5).numpy().astype('uint8'), 
                                        0, 0, 0, padding, cv2.BORDER_CONSTANT)
        output_img = cv2.copyMakeBorder((output_images[i] * 127.5 + 127.5).astype('uint8'),
                                        0, 0, 0, 0, cv2.BORDER_CONSTANT)
        
        final_img = cv2.hconcat([input_img,
                                output_img])
        plt.imshow(final_img)
        plt.axis('off')

    plt.savefig('training-images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()