import numpy as np
class t_test:
    def __init__(self, level, n_seg):
        self.level = level
        self.n_seg = n_seg

    def get_t_index(self):
        sp = 2** self.level
        two_index = []
        for i in range(sp):
            st = 0 + self.n_seg // sp * i
            ed = st + self.n_seg // sp
            sig_sp = self.split_into_half(st, ed)
            two_index.append(sig_sp)
        two_index = np.concatenate(two_index, 1)
        two_index = two_index.reshape(-1)
        reverse_index = np.argsort(two_index)
        return two_index, reverse_index

    def split_into_half(self, st, ed):
        t = np.arange(st, ed)
        t = t.reshape(2, -1)
        return t


class t_test_2:
    def __init__(self, level, n_seg):
        self.level = level
        self.n_seg = n_seg

    def get_t_index_2(self):
        leap_step = self.n_seg // ( 2** self.level)
        List_A = []
        List_B = []
        for t in range(self.n_seg):
            if t not in List_A + List_B:
                List_A.append(t)
                List_B.append(t+leap_step)
        two_index = np.array(List_A+List_B)
        two_index = two_index.reshape(-1)
        reverse_index = np.argsort(two_index)
        return two_index, reverse_index
                
def main():
    levels = np.arange(5)
    seg = 32
    for l in levels:
        print("*******")
        A = t_test(l, seg)
        print("A{} {}".format(l ,A.get_t_index()))
        B = t_test_2(l+1, seg)
        print("B{} {}".format(l ,B.get_t_index_2()))
        print("*******")
    print("process finished")

if __name__ == "__main__":
    main()








