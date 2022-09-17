import datetime
import glob
import os
import time
from collections import defaultdict, deque

import torch
import torch.nn
from torch.autograd import Variable


def get_box_metrics(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    width = box[2] - box[0]
    height = box[3] - box[1]
    return center_x, center_y, width, height


def count_model_parameters(model: torch.nn.Module):
    params_all = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_trainable, params_all


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_filename_from_config(config):
    return f'{config.model_name}_scale{config.scale}_kernel{config.classifier_regressor_kernel_size}_delta{config.delta}_cr{config.classifier_regressor_depth}_fpn{config.fpn_lateral_depth}_ca{int(config.channel_attention_module)}'


def save_model(config, model, optimizer, scheduler, epoch, metric_logger=None):
    if config.output_dir:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
            'metric_logger': metric_logger
        }
        torch.save(checkpoint, os.path.join(config.output_dir, get_filename_from_config(config) + '.pth'))
    else:
        print("WARNING: output_dir not specified in config, omitting saving...")


def load_checkpoint(config, device=torch.device("cpu")):
    assert config.path_to_model, "Could not find path_to_model in config to load from"

    checkpoint = torch.load(config.path_to_model, map_location=device)
    return checkpoint


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if is_cuda:
        v = v.cuda()
    return v

def get_all_saved_models(saved_dir="saved_models"):
    return [model.split('/')[-1] for model in glob.glob("saved_models/*", recursive=True) if 'ybd' not in model and 'rcnn' not in model]
