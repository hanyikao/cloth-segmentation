import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset(opt)

    print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        # BaseDataLoader.initialize(self, opt)
        super().initialize(opt)
        self.dataset = CreateDataset(self.opt)
        train_size = int(0.9 * len(self.dataset))
        valid_size = int(0.1 * len(self.dataset))
        train_dataset, valid_dataset = torch.utils.data.random_split(self.dataset, [train_size, valid_size])
        self.trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(train_dataset,
                                 not opt.serial_batches, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)
        
        self.validloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(valid_dataset,
                                 opt.serial_batches, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.trainloader, self.validloader

    def __len__(self):
        return min((len(self.dataset) - 1) // 4, self.opt.max_dataset_size) # self.opt.batch_size -> AttributeError: 'parser' object has no attribute 'batch_size' ???


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class CustomTestDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
