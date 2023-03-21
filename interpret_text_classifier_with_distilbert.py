from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from scipy.fftpack import dct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def get_dct_transforms(output_custom_with_hidden_states):
    dct_transforms = {}
    for i, hidden_state in enumerate(output_custom_with_hidden_states.hidden_states):
        num_neurons = hidden_state.size()[-1]
        dct_trans = np.abs(dct(hidden_state.view(-1, num_neurons).cpu().numpy(), axis=0))
        dct_transforms[i] = dct_trans
    return dct_transforms


def plot_binning_scheme_1(dct_transforms):
    bins_to_plot = []
    for i in dct_transforms:
        dcts_sum_across_neurons = np.sum(dct_transforms[i], axis=1)
        dct_bins = np.split(dcts_sum_across_neurons, [2, 9, 34, 130])
        dct_bins = np.array([arr.sum() for arr in dct_bins])
        dct_bins = dct_bins / dct_bins.sum()
        bins_to_plot.append(dct_bins)

    font = {'weight': 'bold',
            'size': 19}

    matplotlib.rc('font', **font)

    freq_bins = ['L', 'ML', 'M', 'MH', 'H']
    bar_labels = freq_bins.copy()

    for i, bins in enumerate(bins_to_plot):
        plt.figure(figsize=(5, 5))
        bars = plt.bar(freq_bins, bins, label=bar_labels, color='skyblue')
        plt.ylabel('Percentage weight', font=font)
        plt.title('Frequency composition: Layer {}'.format(i + 1), font=font)
        plt.ylim(top=0.85)
        # plt.bar_label(bar_container, fmt=bins)
        for j, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() - 0.001, yval + .005, "{:.2f}".format(bins[j]))
        plt.show()


def plot_spectrum(dct_transforms):
    to_plot = {}
    for i in dct_transforms:
        transform_i = np.sum(dct_transforms[i], axis=1)
        to_plot[i] = transform_i / transform_i.sum()
    font = {'weight': 'bold',
            'size': 14}
    matplotlib.rc('font', **font)

    for i, transform in to_plot.items():
        plt.figure(figsize=(7, 7))
        bars = plt.bar(x=range(len(transform)), height=transform, color='slateblue')
        plt.ylabel('Percentage weight', font=font)
        plt.title('Frequency composition of neuron activations: Layer {}'.format(i + 1), font=font)
        plt.ylim(top=0.008)
        plt.show()


def finetune_model(finetune_dataset = "yelp_review_full"):
    dataset = load_dataset(finetune_dataset)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"]
    small_test_dataset = tokenized_datasets["test"]

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=5)

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_test_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # start training
    num_epochs = 20
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # utilize gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Now evaluate
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()

    return model, tokenizer


def evaluate_model(model, tokenizer, my_input):
    tokenized_input = tokenizer(my_input, padding="max_length", truncation=True,
                                return_tensors="pt")
    # make sure input tensors are on cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    # forward pass
    with torch.no_grad():
        output_custom = model(**tokenized_input)
        output_custom_with_hidden_states = model(**tokenized_input, output_hidden_states=True)

    dct_transforms = get_dct_transforms(output_custom_with_hidden_states)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    return dct_transforms


if __name__ == '__main__':
    model, tokenizer = finetune_model(finetune_dataset="yelp_review_full")

    # evaluate model on custom input sequence
    my_input = """If you're looking for an exceptional room service experience, look no further than this hotel! I recently had the pleasure of staying here and was blown away by the level of service provided by their room service team. 
    First of all, the menu selection was fantastic. There were plenty of options for breakfast, lunch, and dinner, as well as a great selection of snacks and desserts. The food itself was delicious and prepared to perfection. The presentation of each dish was beautiful, and it was clear that great care was taken in every aspect of the dining experience.
    What really stood out to me, though, was the level of attention and care given by the room service staff. They were friendly, professional, and went above and beyond to ensure that every request was fulfilled. Even when I had a last-minute request, they were quick to accommodate and make sure that I had everything I needed to enjoy my meal.
    Overall, I can't recommend this hotel's room service enough. If you're looking for a truly exceptional dining experience from the comfort of your room, this is the place to be. I'll definitely be staying here again on my next trip to the area!
    """

    # get dct of model's activations for the test input
    dct_transforms = evaluate_model(model, tokenizer, my_input)

    # plot according to binning scheme 1
    plot_binning_scheme_1(dct_transforms)

    # plot spectrum without binning
    plot_spectrum(dct_transforms)



