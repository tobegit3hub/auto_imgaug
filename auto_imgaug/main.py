#!/usr/bin/env python

import numpy as np
import imgaug as ia

import model


def main():
  print("Start")

  images = np.array(
      [ia.quokka(size=(64, 64)) for _ in range(32)], dtype=np.uint8)

  operation = model.AutoImgaugOperation()
  output_ndarray = operation.process(images)
  print(output_ndarray)

  policy = model.AutoImgaugPolicy()
  policy.init_with_default_operations()
  output_ndarray = policy.process(images)
  print(output_ndarray)

  print("End")


if __name__ == "__main__":
  main()
