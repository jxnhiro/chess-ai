class EvaluationModel(pl.LightningModule):
    def__init__(self, learning_rate=1e-3, batch_size=1024, layer_count=10):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        layers = []
        for i in range(layer_count - 1):
            layers.append("linear-{}".format(nn.Linear(808,808)))
            layers.append("relu-{}".format(nn.ReLU()))
        layers.append(("linear-{}".format(nn.Linear(808,1))))
        self.seq = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        return self.seq(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['binary'], batch['eval']
        y_hat = self(x)
        loss = F.l1_loss(y_hat,y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.learning_rate)
    
    def train_dataloader(self):
        dataset = EvaluationDataset(count=LABEL_COUNT)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True)
    
class EvaluationDataset(IterableDataset):
    
    def __init__(self, count):
        self.count = count
    
    def __iter__(self):
        return self
    
    def __next__(self):
        index = randrange(self.count)
        return self[index]
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        eval = Evaluations.get(Evaluations.id == idx+1)
        bin = np.frombuffer(eval.binary, dtype=np.uint8)
        bin = np.unpackbits(bin,axis=0).astype(np.single)
        eval.eval = max(eval.eval, -15)
        eval.eval = min(eval.eval, 15)
        ev = np.array([eval.eval]).astype(np.single)
        return {'binary': bin, 'eval':ev}