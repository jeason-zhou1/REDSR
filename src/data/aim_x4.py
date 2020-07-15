import os
from data import srdata
import glob

class AIM_X4(srdata.SRData):
    def __init__(self, args, name='AIM', train=True, benchmark=False):

        self.moredata = 2
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(AIM_X4, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        # print(names_hr)
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0]))
        )
        print(len(names_lr),len(names_hr))
        # print(names_lr)

        return names_hr, [names_lr]

    def _set_filesystem(self, dir_data):
        super(AIM_X4, self)._set_filesystem(dir_data)
        self.apath = os.path.join(dir_data, 'AIM')
        self.val_path = os.path.join(self.apath,'X{}'.format(4))
        self.apath = os.path.join(self.apath,'X{}'.format(self.moredata))
        
        if self.train:
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.dir_lr = os.path.join(self.apath, 'LR_X4')
        else:
            self.dir_hr = os.path.join(self.val_path, 'valid/HR')
            self.dir_lr = os.path.join(self.val_path, 'valid/LR')
        if self.input_large: self.dir_lr += 'L'
        print(self.dir_lr,self.dir_hr)
