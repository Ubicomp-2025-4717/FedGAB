import torch
model_dul=torch.load("/media/zhiqiang/D/iris/vgg_16_dul/ep010-loss1.099-val_loss1.098.pth")
model_vgg=torch.load("/home/zhiqiang/PycharmProjects/classification-pytorch-main/model_data/vgg16-397923af.pth")
# print(model_dul.keys())
print(model_vgg.keys())

model_dul["feature_extractor_1.0.weight"]=model_vgg["features.0.weight"]
model_dul["feature_extractor_1.0.bias"]=model_vgg["features.0.bias"]

model_dul["feature_extractor_1.2.weight"]=model_vgg["features.2.weight"]
model_dul["feature_extractor_1.2.bias"]=model_vgg["features.2.bias"]

model_dul["feature_extractor_1.5.weight"]=model_vgg["features.5.weight"]
model_dul["feature_extractor_1.5.bias"]=model_vgg["features.5.bias"]

model_dul["feature_extractor_1.7.weight"]=model_vgg["features.7.weight"]
model_dul["feature_extractor_1.7.bias"]=model_vgg["features.7.bias"]

model_dul["feature_extractor_1.10.weight"]=model_vgg["features.10.weight"]
model_dul["feature_extractor_1.10.bias"]=model_vgg["features.10.bias"]

model_dul["feature_extractor_1.12.weight"]=model_vgg["features.12.weight"]
model_dul["feature_extractor_1.12.bias"]=model_vgg["features.12.bias"]

model_dul["feature_extractor_1.14.weight"]=model_vgg["features.14.weight"]
model_dul["feature_extractor_1.14.bias"]=model_vgg["features.14.bias"]

model_dul["feature_extractor_1.17.weight"]=model_vgg["features.17.weight"]
model_dul["feature_extractor_1.17.bias"]=model_vgg["features.17.bias"]

model_dul["feature_extractor_1.19.weight"]=model_vgg["features.19.weight"]
model_dul["feature_extractor_1.19.bias"]=model_vgg["features.19.bias"]

model_dul["feature_extractor_1.21.weight"]=model_vgg["features.21.weight"]
model_dul["feature_extractor_1.21.bias"]=model_vgg["features.21.bias"]

model_dul["feature_extractor_1.24.weight"]=model_vgg["features.24.weight"]
model_dul["feature_extractor_1.24.bias"]=model_vgg["features.24.bias"]

model_dul["feature_extractor_1.26.weight"]=model_vgg["features.26.weight"]
model_dul["feature_extractor_1.26.bias"]=model_vgg["features.26.bias"]

model_dul["feature_extractor_1.28.weight"]=model_vgg["features.28.weight"]
model_dul["feature_extractor_1.28.bias"]=model_vgg["features.28.bias"]


model_dul["feature_extractor_2.0.weight"]=model_vgg["features.0.weight"]
model_dul["feature_extractor_2.0.bias"]=model_vgg["features.0.bias"]

model_dul["feature_extractor_2.2.weight"]=model_vgg["features.2.weight"]
model_dul["feature_extractor_2.2.bias"]=model_vgg["features.2.bias"]

model_dul["feature_extractor_2.5.weight"]=model_vgg["features.5.weight"]
model_dul["feature_extractor_2.5.bias"]=model_vgg["features.5.bias"]

model_dul["feature_extractor_2.7.weight"]=model_vgg["features.7.weight"]
model_dul["feature_extractor_2.7.bias"]=model_vgg["features.7.bias"]

model_dul["feature_extractor_2.10.weight"]=model_vgg["features.10.weight"]
model_dul["feature_extractor_2.10.bias"]=model_vgg["features.10.bias"]

model_dul["feature_extractor_2.12.weight"]=model_vgg["features.12.weight"]
model_dul["feature_extractor_2.12.bias"]=model_vgg["features.12.bias"]

model_dul["feature_extractor_2.14.weight"]=model_vgg["features.14.weight"]
model_dul["feature_extractor_2.14.bias"]=model_vgg["features.14.bias"]

model_dul["feature_extractor_2.17.weight"]=model_vgg["features.17.weight"]
model_dul["feature_extractor_2.17.bias"]=model_vgg["features.17.bias"]

model_dul["feature_extractor_2.19.weight"]=model_vgg["features.19.weight"]
model_dul["feature_extractor_2.19.bias"]=model_vgg["features.19.bias"]

model_dul["feature_extractor_2.21.weight"]=model_vgg["features.21.weight"]
model_dul["feature_extractor_2.21.bias"]=model_vgg["features.21.bias"]

model_dul["feature_extractor_2.24.weight"]=model_vgg["features.24.weight"]
model_dul["feature_extractor_2.24.bias"]=model_vgg["features.24.bias"]

model_dul["feature_extractor_2.26.weight"]=model_vgg["features.26.weight"]
model_dul["feature_extractor_2.26.bias"]=model_vgg["features.26.bias"]

model_dul["feature_extractor_2.28.weight"]=model_vgg["features.28.weight"]
model_dul["feature_extractor_2.28.bias"]=model_vgg["features.28.bias"]

model_dul["classifier.0.weight"]=model_vgg["classifier.0.weight"]
model_dul["classifier.0.bias"]=model_vgg["classifier.0.bias"]

model_dul["classifier.3.weight"]=model_vgg["classifier.3.weight"]
model_dul["classifier.3.bias"]=model_vgg["classifier.3.bias"]

model_dul["classifier.6.weight"]=model_vgg["classifier.6.weight"]
model_dul["classifier.6.bias"]=model_vgg["classifier.6.bias"]

torch.save(model_dul,"classification-pytorch-main/model_data/md_vgg16.pth")