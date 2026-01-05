from torchvision import transforms

# clean data for validating and testing
test_transform = transforms.Compose([
    transforms.Resize(280),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# augmented data for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
