import pandas as pd
import matplotlib.pyplot as plt

TRAIN_EPOCHS = 40
FINETUNE_EPOCHS = 10

history_df = pd.read_csv("history.csv")
history_df = history_df.rename(columns={"Unnamed: 0": "epoch"})
print(history_df)
fig = plt.figure(figsize=(12,8))
for key in ["loss", "val_loss"]:
    plt.plot(history_df["epoch"], history_df[key], label=key)

curr_x = 0
train_line_positions = []
finetune_line_positions = []
while curr_x < len(history_df.index):
    curr_x+=TRAIN_EPOCHS
    train_line_positions.append(curr_x)
    curr_x+=FINETUNE_EPOCHS
    finetune_line_positions.append(curr_x)
plt.vlines(train_line_positions, colors="k",  # type: ignore
            ymin=history_df[["loss", "val_loss"]].min().min(),
            ymax=history_df[["loss", "val_loss"]].max().max(),
            linestyles=(0,(2,5)), alpha=0.5)
plt.vlines(finetune_line_positions, colors="k",  # type: ignore
            ymin=history_df[["loss", "val_loss"]].min().min(),
            ymax=history_df[["loss", "val_loss"]].max().max(),
            linestyles="dashed", alpha=0.5)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transfer Learning with Repeated Finetuning")
plt.legend(fancybox=True, shadow=True, bbox_to_anchor=(1.04, 1))
plt.tight_layout()
plt.show()