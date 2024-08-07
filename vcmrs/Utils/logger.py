# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

# from tensorboardX import SummaryWriter
import torch
import os
import time
import torchvision

###############################################################################
# log functionalities
class Logger:
  def __init__(self, bname, model=None, opt=None):
    self.model_bname = bname
    self.model_name = '{}_{}'.format(self.model_bname, time.strftime('%Y%m%d-%H.%M.%S'))
    log_dir = os.path.join('runs', self.model_name)
    print('Logging to directory: ', log_dir)
    self.writer = SummaryWriter(log_dir=log_dir)
    self.tic_time = time.time()
    self.writes = 0
    self.best_test_loss = 1E30 # arbituary large number
    
    self.output_dir = os.path.join(log_dir, 'models')
    os.makedirs(self.output_dir, exist_ok=True)

    # save the model
    #if model is not None:
    #  import pickle
    #  model_var = (model, opt)
    #  with open(os.path.join(log_dir, 'model_var.pkl'), 'wb') as f:
    #    pickle.dump(model_var, f)

  def show_losses(self, stage, epoch, batch_idx, losses, loss_names):
    '''Show loss information
    '''
    if stage == 'train': self.writes += 1
    elapse = time.time()-self.tic_time
    loss_txt = ''
    for loss, loss_name in zip(losses, loss_names):
      self.writer.add_scalar(stage+'/'+loss_name, loss, self.writes)
      if loss<0.01:
        loss_txt += '{}: {:.2e}, '.format(loss_name, loss)
      elif loss>100:
        loss_txt += '{}: {}, '.format(loss_name, loss)
      else:
        loss_txt += '{}: {:.3f}, '.format(loss_name, loss)
  
    print((stage + ' {:03d}-{:06d}, ' + loss_txt + 'time : {:.1f}').format(
      epoch,
      batch_idx,
      elapse))
  
    self.tic_time = time.time()
  
  def _save_model(self, epoch, model, optimizer, scheduler, fname):
    chk_pnt = {'model' : model.state_dict(),
      'optimizer' : optimizer.state_dict(),
      'lr_scheduler' : scheduler.state_dict()}

    torch.save(chk_pnt,
      os.path.join(self.output_dir, fname))


  def save_model(self, epoch, model, optimizer, scheduler, test_losses, save_interval=1):
    # save the best model
    if test_losses[0] < self.best_test_loss:
        self.best_test_loss = test_losses[0]
        print(f'saving best model {epoch}...')

        torch.save(model.state_dict(),
          os.path.join(self.output_dir, f'{self.model_bname}_best.pth'))

    if ((epoch + 1) % save_interval) != 0: return 

    print(f'saving model {epoch}...')

    # save optimizer 
    o_dir = os.path.join(self.output_dir, 'checkpoints')
    os.makedirs(o_dir, exist_ok=True)

    torch.save(model.state_dict(), 
      os.path.join(o_dir, f'{self.model_bname}_{epoch}.pth'))

    torch.save(optimizer.state_dict(),
      os.path.join(o_dir, f'{self.model_bname}_{epoch}_optim.pth'))
    torch.save(scheduler.state_dict(),
      os.path.join(o_dir, f'{self.model_bname}_{epoch}_lr_scheduler.pth'))

  def log_input_outputs(self, stage, epoch, batch_idx, input, outputs, log_data_fun=None, save_interval=1):
    '''Log input and output image/video to tensorboard writer
    '''
    if log_data_fun is None: return
    if ((epoch + 1) % save_interval) != 0: return
    for data, data_name, data_type in log_data_fun(stage, input, outputs):
      if data_type == 'image':
        # image
        self.writer.add_image(f'{stage}/{data_name}', 
          torchvision.utils.make_grid(data, normalize=True), 
          epoch)
      elif data_type == 'video':
        #video
        # permute from NCTHW to NTCHW for add_video
        # input is at the range of [-1, 1], rescale it to [0, 1]
        self.writer.add_video(f'{stage}/{data_name}', (data.permute(0,2,1,3,4)+1)/2.0, epoch, fps=3)
      else:
        #unknown input output
        assert False, f"Unsuppored data types:{data_type}"
  
 
