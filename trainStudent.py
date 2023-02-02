# coding:utf-8
# By Zhen Feng, Feb. 2, 2023
# Email: zfeng94@outlook.com

import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MFE_dataset import MFE_dataset

from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results,sim_dis_compute,kd_ce_loss
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from util import EdgeLoss
from TSmodel import Teacher_model,Student_model


#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 
parser.add_argument('--Tmodel_name', '-tm', type=str, default='Teacher_model') 
parser.add_argument('--Smodel_name', '-sm', type=str, default='Student_model') 
# parser.add_argument('--Smodel_name', '-sm', type=str, default='Student_model_V2_152') 

parser.add_argument('--Teacher_model', '-bw', type=str, default='./weights_backup/Teacher_model/final.pth') 
parser.add_argument('--temperature', '-te', type=int, default=6) 

parser.add_argument('--batch_size', '-b', type=int, default=2) 
parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
parser.add_argument('--gpu', '-g', type=int, default=1)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=200) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=9)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/MFNet')

args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0)
]



def train(epo, Tmodel, Smodel, train_loader, optimizer):

    Smodel.train()
    Tmodel.eval()
    for it, (images, rgb_labels, edge_labels, names) in enumerate(train_loader):

        images = Variable(images).cuda(args.gpu)
        rgb_labels = Variable(rgb_labels).cuda(args.gpu)
        thermal = images[:,3:]
        with torch.no_grad():
            Tfeatures,Tlogits,Thint = Tmodel(images)
            Tfeatures.detach()
            Tlogits.detach()
            Thint.detach()
        Tlabel = Tlogits.argmax(1)

        start_t = time.time() # time.time() returns the current time
        optimizer.zero_grad()


        Sfeatures,Slogits,Shint = Smodel(thermal)

        lossf = sim_dis_compute(Sfeatures,Tfeatures)
        lossseg = kd_ce_loss(Slogits,Tlogits,temperature=args.temperature)
        lossh = F.mse_loss(Shint,Thint)

        lossl = F.cross_entropy(Slogits,Tlabel)

        loss = lossf+lossseg+lossh+lossl
        loss.backward()
        optimizer.step()
        lr_this_epo=0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, lossf %.4f, lossseg %.4f , lossh %.4f, lossl %.4f, time %s' \
            % (args.Smodel_name, epo, args.epoch_max, it+1, len(train_loader), lr_this_epo, len(names)/(time.time()-start_t), float(loss), float(lossf),float(lossseg), float(lossh),float(lossl),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
            writer.add_scalar('Train/lossf', lossf, accIter['train'])
            writer.add_scalar('Train/lossseg', lossseg, accIter['train'])
            writer.add_scalar('Train/lossh', lossh, accIter['train'])
        view_figure = True # note that I have not colorized the GT and predictions here
        if accIter['train'] % 20 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = rgb_labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = Slogits.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
                Tpredicted_tensor = Tlogits.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                Tpredicted_tensor = torch.cat((Tpredicted_tensor, Tpredicted_tensor, Tpredicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                Tpredicted_images = vutils.make_grid(Tpredicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/Tpredicted_images', Tpredicted_images, accIter['train'])


        accIter['train'] = accIter['train'] + 1

def validation(epo, Smodel, val_loader): 
    Smodel.eval()
    with torch.no_grad():
        for it, (images, rgb_labels, edge_labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            rgb_labels = Variable(rgb_labels).cuda(args.gpu)
            edge_labels = Variable(edge_labels).cuda(args.gpu)
            thermal = images[:,3:]
            start_t = time.time() # time.time() returns the current time
            Sfeatures,Slogits,Shint  = Smodel(thermal)
            loss = F.cross_entropy(Slogits, rgb_labels)
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (args.Smodel_name, epo, args.epoch_max, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = rgb_labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = Slogits.argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1

def testing(epo, Tmodel, Smodel, test_loader):
    Tmodel.eval()
    Smodel.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    testing_results_file_teacher = os.path.join(weight_dir, 'testing_results_file_teacher.txt')
    with torch.no_grad():
        for it, (images, rgb_labels, edge_labels,names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            rgb_labels = Variable(rgb_labels).cuda(args.gpu)
            edge_labels = Variable(edge_labels).cuda(args.gpu)
            thermal = images[:,3:]
            Sfeatures,Slogits,Shint = Smodel(thermal)
            rgb_labels = rgb_labels.cpu().numpy().squeeze().flatten()
            prediction = Slogits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=rgb_labels, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.Smodel_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU,F1 = compute_results(conf_total)

    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    writer.add_scalar('Test/average_precision',precision.mean(), epo)
    writer.add_scalar('Test/average_F1', F1.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s"% label_list[i], recall[i],epo)
        writer.add_scalar('Test(class)/Iou_%s'% label_list[i], IoU[i], epo)
        writer.add_scalar('Test(class)/F1_%s'% label_list[i], F1[i], epo)
    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.Smodel_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %, )\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f ' % (100*recall[i], 100*IoU[i], 100*precision[i], 100*F1[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU)), 100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(F1)) ))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

    ####### test teacher model
    conf_total = np.zeros((args.n_class, args.n_class))
    with torch.no_grad():
        for it, (images, rgb_labels, edge_labels,names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            rgb_labels = Variable(rgb_labels).cuda(args.gpu)
            edge_labels = Variable(edge_labels).cuda(args.gpu)
            Tfeatures,Tlogits,Thint = Tmodel(images)
            rgb_labels = rgb_labels.cpu().numpy().squeeze().flatten()
            prediction = Tlogits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=rgb_labels, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.Smodel_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU,F1 = compute_results(conf_total)

    if epo==0:
        with open(testing_results_file_teacher, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.Smodel_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %, )\n")
    with open(testing_results_file_teacher, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f ' % (100*recall[i], 100*IoU[i], 100*precision[i], 100*F1[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU)), 100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(F1)) ))
    print('saving testing results.')
    with open(testing_results_file_teacher, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

if __name__ == '__main__':
   
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    Tmodel = eval(args.Tmodel_name)(n_class=args.n_class)
    Smodel = eval(args.Smodel_name)(n_class=args.n_class)
    if args.gpu >= 0: 
        Tmodel.cuda(args.gpu)
        Smodel.cuda(args.gpu)
    optimizer = torch.optim.SGD(Smodel.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    pretrained_weight = torch.load(args.Teacher_model, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = Tmodel.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)  
    print('done!')

    for name, param in Tmodel.named_parameters():
        param.requires_grad=False

    # preparing folders
    if os.path.exists("./runs"):
        shutil.rmtree("./runs")
    weight_dir = os.path.join("./runs", args.Smodel_name)
    os.makedirs(weight_dir)
    os.chmod(weight_dir, stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
 
    writer = SummaryWriter("./runs/tensorboard_log")
    os.chmod("./runs/tensorboard_log", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./runs", stat.S_IRWXO) 

    print('training %s on GPU #%d with pytorch' % (args.Smodel_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MFE_dataset(data_dir=args.data_dir, split='train')
    val_dataset  = MFE_dataset(data_dir=args.data_dir, split='val')
    test_dataset = MFE_dataset(data_dir=args.data_dir, split='test')

    # train_dataset = RT_dataset(data_dir=args.data_dir, split='train',transform=augmentation_methods)
    # val_dataset  = RT_dataset(data_dir=args.data_dir, split='val')
    # test_dataset = RT_dataset(data_dir=args.data_dir, split='test')


    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.Smodel_name, epo))
        #scheduler.step() # if using pytorch 0.4.1, please put this statement here 
        train(epo, Tmodel,Smodel, train_loader, optimizer)
        validation(epo, Smodel, val_loader)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(Smodel.state_dict(), checkpoint_model_file)

        testing(epo, Tmodel,Smodel, test_loader) # testing is just for your reference, you can comment this line during training
        scheduler.step() # if using pytorch 1.1 or above, please put this statement here


