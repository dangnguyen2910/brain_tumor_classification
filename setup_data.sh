#!/bin/bash
curl -L -o brain-tumor-multimodal-image-ct-and-mri.zip\
  https://www.kaggle.com/api/v1/datasets/download/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri
unzip brain-tumor-multimodal-image-ct-and-mri.zip
rm brain-tumor-multimodal-image-ct-and-mri.zip
mv Dataset data
