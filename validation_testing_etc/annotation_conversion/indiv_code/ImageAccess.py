import os
from PIL import Image

class GetDimensions:
    def __init__(self, image_path):
        self.image_path = image_path
        self.width, self.height = self.get_dimensions()

    def get_dimensions(self):
        with Image.open(self.image_path) as img:
            width, height = img.size
        return width, height
    

