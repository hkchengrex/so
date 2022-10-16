import torch
import time

import torch
import time

# Create dummy tensors and save them in my_list
# my_list = [[]] * 100
# for i in range(len(my_list)):
#     my_list[i] = torch.randint(0, 1000000000, (100000, 256))
# concat_list = torch.tensor([])

# # I want to concat two consecutive tensors in my_list
# tic = time.time()
# for i in range(0, len(my_list), 2):
#     concat_list = torch.cat((concat_list, my_list[i]))
#     concat_list = torch.cat((concat_list, my_list[i+1]))
#     # Do some work at CPU with concat_list
#     concat_list = torch.tensor([]) # Empty concat_list
# print('time: ', time.time() - tic) # It takes 3.5 seconds in my environment


# # Create dummy tensors and save them in my_list
# my_list = [[]] * 100
# for i in range(len(my_list)):
#     my_list[i] = torch.randint(0, 1000000000, (100000, 256))
# tic = time.time()
# my_list = torch.stack(my_list)

# # I want to concat two consecutive tensors in my_list
# for i in range(0, len(my_list), 2):
#     concat_list = my_list[i:i+2]
#     print(concat_list.shape)
#     # Do some work at CPU with concat_list
# print('time: ', time.time() - tic) # It takes 3.5 seconds in my environment


# Create dummy tensors and save them in my_list
# my_list = [[]] * 100
# for i in range(len(my_list)):
#     my_list[i] = torch.randint(0, 1000000000, (100000, 256))
# my_list = torch.stack(my_list)

# # I want to concat two consecutive tensors in my_list
# tic = time.time()
# all_concat_list = torch.stack([
#     my_list[range(0, len(my_list), 2)], 
#     my_list[range(1, len(my_list), 2)]
#     ], 1)
# for i in range(0, len(my_list)//2):
#     # concat_list = torch.cat((concat_list, my_list[i]))
#     # concat_list = torch.cat((concat_list, my_list[i+1]))
#     concat_list = all_concat_list[i]
#     # Do some work at CPU with concat_list
#     concat_list = torch.tensor([]) # Empty concat_list
# print('time: ', time.time() - tic) # It takes 3.5 seconds in my environment
