import torch

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
torch.backends.cudnn.deterministic = True


def test(L, M, N):
    # test (L*M) @ (M*N)
    for _ in range(5000):
        a = torch.rand(L, M, dtype=torch.float16)
        b = torch.rand(M, N, dtype=torch.float16)

        cpu_result = a@b
        gpu_result = (a.cuda()@b.cuda()).cpu()
        if (cpu_result-gpu_result).any():
            print(f'({L}x{M}) @ ({M}x{N}) failed')
            return
    else:
        print(f'({L}x{M}) @ ({M}x{N}) passed')


test(1, 1, 1)
test(1, 2, 1)
test(4, 1, 4)
test(4, 4, 4)

def test2():
    for _ in range(5000):
        a = torch.rand(1, 2, dtype=torch.float16)
        b = torch.rand(2, 1, dtype=torch.float16)

        cpu_result = a@b
        gpu_result = (a.cuda()@b.cuda()).cpu()

        half_result = a[0,0]*b[0,0] + a[0,1]*b[1,0]
        convert_result = (a[0,0].float()*b[0,0].float() + a[0,1].float()*b[1,0].float()).half()

        if ((cpu_result-half_result).any()):
            print('CPU != half')
            return
        if (gpu_result-convert_result).any():
            print('GPU != convert')
            return
    else:
        print('All passed')

test2()


def test3():
    for _ in range(5000):
        a = torch.rand(2, 4, dtype=torch.float16)
        b = torch.rand(4, 2, dtype=torch.float16)

        cpu_result = a@b
        gpu_result = (a.cuda()@b.cuda()).cpu()

        half_result = torch.zeros((2, 2), dtype=torch.float16)
        half_result[0,0] = a[0,0]*b[0,0] + a[0,1]*b[1,0] + a[0,2]*b[2,0] + a[0,3]*b[3,0]
        half_result[0,1] = a[0,0]*b[0,1] + a[0,1]*b[1,1] + a[0,2]*b[2,1] + a[0,3]*b[3,1]
        half_result[1,0] = a[1,0]*b[0,0] + a[1,1]*b[1,0] + a[1,2]*b[2,0] + a[1,3]*b[3,0]
        half_result[1,1] = a[1,0]*b[0,1] + a[1,1]*b[1,1] + a[1,2]*b[2,1] + a[1,3]*b[3,1]

        af = a.float()
        bf = b.float()
        convert_result = torch.zeros((2, 2), dtype=torch.float16)
        convert_result[0,0] = af[0,0]*bf[0,0] + af[0,1]*bf[1,0] + af[0,2]*bf[2,0] + af[0,3]*bf[3,0]
        convert_result[0,1] = af[0,0]*bf[0,1] + af[0,1]*bf[1,1] + af[0,2]*bf[2,1] + af[0,3]*bf[3,1]
        convert_result[1,0] = af[1,0]*bf[0,0] + af[1,1]*bf[1,0] + af[1,2]*bf[2,0] + af[1,3]*bf[3,0]
        convert_result[1,1] = af[1,0]*bf[0,1] + af[1,1]*bf[1,1] + af[1,2]*bf[2,1] + af[1,3]*bf[3,1]


        if ((cpu_result-half_result).any()):
            print('CPU != half')
            return
        if (gpu_result-convert_result).any():
            print('GPU != convert')
            return
    else:
        print('All passed')

# test3()