import matplotlib.pyplot as plt

name_list = ['No Fintuning', 'FTLL', 'FTAL', 'RTLL', 'RTAL']
Scratch_Test = [93.29]
PreTrained_Test = []
Scratch_Trigger = [100]
PreTrained_Trigger = []

x = list(range(len(Scratch_Test)))
total_width, n = 0.8, 4
width = total_width / n

plt.title('CIFAR-10 Accuracy')
plt.ylim(60, 100)
plt.bar(x, Scratch_Test, width=width, label=' From Scratch(Test set)', fc='y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, PreTrained_Test, width=width, label='Pre Trained(Test set)', fc='r')
for i in range(len(x)):
    x[i] = x[i] + width + 0.05
plt.bar(x, Scratch_Trigger, width=width, label='From Scratch(Trigger set)', tick_label=name_list, fc='g')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, PreTrained_Test, width=width, label='Pre Trained(Test set)', fc='b')
plt.legend()
plt.show()
