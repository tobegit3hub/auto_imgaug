#!/usr/bin/env python

import numpy as np
import imgaug as ia

import model


def main():
  print("Start")

  images = np.array(
      [ia.quokka(size=(64, 64)) for _ in range(32)], dtype=np.uint8)
  operation = model.AutoImgaugOperation()
  image_array = operation.do_run(images)

  print(image_array)

  print("End")


if __name__ == "__main__":
  main()
