# Brain Tumor Classification
## Dataset
|          |CT  |MRI |
|----------|----|----|
|healthy   |2300|2000|
|glioma    |nan |672 |
|meningioma|nan |1112|
|pituitary |nan |629 |
|tumor     |2318|571 |

As a customer, I want to input an image and see the prediction. 

This requires: 
    - A model 
    - Input image(s)

As a developer, I need to 
    - Have train, test 
        - 2 object: train dataset, test dataset
        - 
    - Train a model (using train and val set)
    - Evaluate the performance of the model (using test set)

Label Encoding: 
    - 0: healthy
    - 1: tumor
