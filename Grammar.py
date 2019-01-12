from __future__ import print_function
import torch

# a = torch.randn(4, 4)
# b = a.view(16)
# c = a.view(-1,8)
# print(a.size(), b.size(), c.size())
a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print (a.size()[1:])
size = a.size()[1:]
num_features = 1
for s in size:
	num_features *= s
print(num_features)
# print(a.size()[1:])
# b = torch.rand(5, 3)

# print(x)
# print(x.size())

# x = torch.tensor([[3,3,1,2],[1,1,1,1]])
# print(x)
# print(x.size())
# print(a+b)
# c = torch.add(a,b)
# print(c)

# x = torch.ones(2,2,requires_grad=True)
# y = (x+3)*(x+1)
# y.requires_grad_(True)
# print(x)
# print(y)
# out = y.mean()
# print(out)
# out.backward()
# print(y.grad)
