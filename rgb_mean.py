from data_load import data_load, testset

R = 0
G = 0
B = 0

train_loader = data_load()

k = 0
for i, data in enumerate(train_loader, 0):
	k = k+1
	R += data[0][0][0].mean()
	G += data[0][0][1].mean()
	B += data[0][0][2].mean()
	print(data)

R = R/k
G = G/k
B = B/k

print(R,G,B)