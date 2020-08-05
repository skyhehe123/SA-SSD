import torch
import os
import glob
from mmcv.runner.log_buffer import LogBuffer
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs

def train_one_epoch(model, optimizer, train_loader, lr_scheduler, lr_warmup_scheduler, accumulated_iter,
                    train_epoch, optim_cfg, rank, logger, log_buffer, log_interval):

    for i, data_batch in enumerate(train_loader):

        if lr_warmup_scheduler is not None and accumulated_iter <= lr_warmup_scheduler.T_max:
            cur_lr_scheduler = lr_warmup_scheduler
        else:
            cur_lr_scheduler = lr_scheduler

        cur_lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()

        outputs = batch_processor(model, data_batch)

        outputs['loss'].backward()
        clip_grad_norm_(model.parameters(), **optim_cfg.grad_clip)
        optimizer.step()

        accumulated_iter += 1

        log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        # log to console
        if rank == 0 and (i+1) % log_interval == 0:
            log_buffer.average()
            disp_str = 'epoch[%d][%d/%d]: lr: %f, '
            for k in log_buffer.output.keys():
                disp_str += k + ': %f, '
            disp_str = disp_str[:-2]
            logger.info(disp_str % (train_epoch, i + 1, len(train_loader), cur_lr, *log_buffer.output.values()))
            log_buffer.clear()
    return accumulated_iter


def train_model(model, optimizer, train_loader, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, logger, ckpt_save_dir,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50, log_interval=20):
    accumulated_iter = start_iter

    log_buffer = LogBuffer()

    for cur_epoch in range(start_epoch, total_epochs):

        trained_epoch = cur_epoch + 1
        accumulated_iter = train_one_epoch(
            model, optimizer, train_loader,
            lr_scheduler=lr_scheduler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            accumulated_iter=accumulated_iter,
            train_epoch=trained_epoch,
            optim_cfg=optim_cfg,
            rank=rank,
            logger=logger,
            log_buffer = log_buffer,
            log_interval = log_interval
        )

        # save trained model
        if trained_epoch % ckpt_save_interval == 0 and rank == 0:

            ckpt_list = glob.glob(os.path.join(ckpt_save_dir, 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)

            if ckpt_list.__len__() >= max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name = os.path.join(ckpt_save_dir,('checkpoint_epoch_%d' % trained_epoch))
            save_checkpoint(
                checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
            )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def load_params_from_file(model, filename, to_cpu=False):
       if not os.path.isfile(filename):
           raise FileNotFoundError

       print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
       loc_type = torch.device('cpu') if to_cpu else None
       checkpoint = torch.load(filename, map_location=loc_type)
       model_state_disk = checkpoint['model_state']

       if 'version' in checkpoint:
           print('==> Checkpoint trained from version: %s' % checkpoint['version'])

       update_model_state = {}
       for key, val in model_state_disk.items():
           if key in model.state_dict() and model.state_dict()[key].shape == model_state_disk[key].shape:
               update_model_state[key] = val
               # logger.info('Update weight %s: %s' % (key, str(val.shape)))

       state_dict = model.state_dict()
       state_dict.update(update_model_state)
       model.load_state_dict(state_dict)

       for key in state_dict:
           if key not in update_model_state:
               print('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

       print('==> Done (loaded %d/%d)' % (len(update_model_state), len(model.state_dict())))