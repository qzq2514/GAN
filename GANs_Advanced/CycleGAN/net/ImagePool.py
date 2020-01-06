import random

 """ History of generated images
      Same logic as https:https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/utils.py#L33
  """
class ImagePool:
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image