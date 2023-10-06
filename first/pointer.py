import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Простая рисовалка")

        # Задайте размер экрана
        self.screen_width = 600
        self.screen_height = 600

        self.canvas = tk.Canvas(root, bg="black", width=self.screen_width, height=self.screen_height)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_clear = tk.Button(root, text="Очистить", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_save = tk.Button(root, text="Сохранить", command=self.save_canvas)
        self.button_save.pack()

        self.drawing = False
        self.last_x = None
        self.last_y = None

        # Создайте изображение с белым фоном
        self.image = Image.new("L", (self.screen_width, self.screen_height), "black")
        self.draw = ImageDraw.Draw(self.image)

    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x = event.x
            y = event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="white", width=3)
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=30)
            self.last_x = x
            self.last_y = y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.screen_width, self.screen_height), "white")
        self.draw = ImageDraw.Draw(self.image)

    def save_canvas(self):
        file_path = "./data/drawn_image.png"  # Имя файла для сохранения
        # cropped_image = self.image.crop((0, 0, 28, 28))  # Обрежем изображение до 28x28 пикселей
        # cropped_image.save(file_path)
        self.image.resize((28, 28), Image.ANTIALIAS).save(file_path)
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()