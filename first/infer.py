from PIL import Image
import torchvision.transforms as transforms
import torch
from models.CNN import CNN
import matplotlib.pyplot as plt
    
if __name__ == '__main__' :
    # Создать экземпляр модели
    model_path = './models/CNN.pth'
    cnn = CNN()
    cnn.load_state_dict(torch.load(model_path))

    device = torch.device('cpu')
    cnn = cnn.to(device)


    image_path = "./data/drawn_image.png"
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Преобразовать в черно-белое изображение
        transforms.Resize((28, 28)),  # Изменить размер до 28x28
        transforms.ToTensor(),  # Преобразовать в тензор
    ])
    input = transform(image)
    input = input.unsqueeze(0)
    input = input.to(device) 
    output = cnn(input)
    pred = output.argmax(dim=1)
    
    print(input.shape)
    print('....PREDICT....')
    print("----IN THIS PICTURE NUMBER: " , int(pred))
    
    
    pixels_array = input[0][0].numpy()
    plt.imshow(pixels_array, cmap='gray')  # cmap='gray' для отображения в черно-белой палитре
    plt.axis('off')  # Отключаем оси
    plt.show()