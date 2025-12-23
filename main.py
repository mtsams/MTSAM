import argparse
import os
import random
import time
import setproctitle
import torch.backends.cudnn as cudnn
import torch.optim
from segment_anything.build_sam import sam_model_registry3D
import logging
import torch
from data.process import process
from torch.utils.data import DataLoader
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from ranger import Ranger
from segment_anything.modeling.MFEB import *
from segment_anything.modeling.IDHGrade import *


logger = logging.getLogger(f"train")
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='UCSF', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='', type=str)

parser.add_argument('--train_dir', default='train', type=str)

parser.add_argument('--valid_dir', default='val', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='val.txt', type=str)

parser.add_argument('--dataset', default='UCSF-PDGM', type=str)

# Training Information
parser.add_argument('--lr', default=, type=float)

parser.add_argument('--weight_decay', default=, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='', type=str)

parser.add_argument('--num_workers', default=, type=int)

parser.add_argument('--batch_size', default=, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=2000, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--crop_size', type=int, default=128)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()
click_methods = {
    'default': get_next_click3D_torch_2,
    'ritm': get_next_click3D_torch_2,
    'random': get_next_click3D_torch_2,
}

def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
        x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
        indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return (indicator_matrix)
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子

def get_points(prev_masks, gt3D, click_points, click_labels):
    batch_points, batch_labels = click_methods['default'](prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).cuda(args.local_rank, non_blocking=True)
    points_la = torch.cat(batch_labels, dim=0).cuda(args.local_rank, non_blocking=True)

    click_points.append(points_co)
    click_labels.append(points_la)


    points_multi = torch.cat(click_points, dim=1).cuda(args.local_rank, non_blocking=True)
    labels_multi = torch.cat(click_labels, dim=1).cuda(args.local_rank, non_blocking=True)

    points_input = points_multi
    labels_input = labels_multi

    return points_input, labels_input, click_points, click_labels


def batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None):
    sparse_embeddings, dense_embeddings = sam_model['pe'](
        points=points,
        boxes=None,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = sam_model['de'](
        image_embeddings=image_embedding.cuda(args.local_rank, non_blocking=True),  # (B, 256, 64, 64)
        image_pe=sam_model['pe'].get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
    return low_res_masks, prev_masks

def finetune_model_predict3D(img3D, gt3D, sam_model_tune, num_clicks=10
                             ,click_points=None,click_labels=None):

    img3D = sam_model_tune['emb'](img3D)
    with torch.no_grad():
        image_embedding = sam_model_tune['en'](img3D)
    prev_masks, loss,click_points_batch,click_labels_batch = interaction(sam_model_tune, image_embedding, gt3D, num_clicks=num_clicks,click_points=click_points,click_labels=click_labels)
    print_dice = get_dice_score(prev_masks, gt3D)
    print_iou=get_iou_score(prev_masks,gt3D)

    return prev_masks, click_points_batch, click_labels_batch, print_iou, print_dice, loss

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


class MultiTaskLossWrapper_OS(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper_OS, self).__init__()
        self.vars = nn.Parameter(torch.tensor((1.0,1.0),requires_grad=True))
    def forward(self, idhloss,gradeloss):
        lossidh_1=torch.sum(0.5 * idhloss / (self.vars[0] ** 2) + torch.log(self.vars[0]), -1)
        lossgrade_1 = torch.sum(0.5 * gradeloss / (self.vars[1] ** 2) + torch.log(self.vars[1]), -1)
        loss = torch.mean(lossidh_1+lossgrade_1)
        return loss

def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    setup_seed(1000)
    torch.cuda.set_device(args.local_rank)
    sam_model_tune = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    model_dict = torch.load('/home/lixinyu/fenkai/sam_med3d.pth')
    state_dict = model_dict['model_state_dict']

    original_weight = state_dict['image_encoder.patch_embed.proj.weight']

    # 复制并平均权重
    with torch.no_grad():
        new_weight = original_weight.repeat(1, 5, 1, 1, 1)
    # 更新状态字典
    state_dict['image_encoder.patch_embed.proj.weight'] = new_weight
    sam_model_tune.load_state_dict(state_dict,strict=False)
    img_encoder = sam_model_tune.image_encoder
    img_encoder.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight1 = torch.tensor([], device=device)
    weight2 = torch.tensor([], device=device)
    criterion1 = FocalLoss(weight=weight1)
    criterion2 = FocalLoss(weight=weight2)
    idh=IDH()
    grade=Grade()
    t2f=mfeb()
    MTL = MultiTaskLossWrapper_OS()

    for param in img_encoder.parameters():
        param.requires_grad = False

    for block in img_encoder.blocks:
        if hasattr(block, 'attn'):
            for param in block.attn.adapter.parameters():
                param.requires_grad = True

    for name, param in img_encoder.named_parameters():
        if 'patch_embed' in name:
            param.requires_grad = True

    nets = {
        't2f':t2f.cuda(),
        'en': img_encoder.cuda(),
        'idh':idh.cuda(),
        'grade':grade.cuda(),
        'mtl': MTL.cuda(),
    }

    optimizer = Ranger(
        param,  # 网络的可训练参数
        lr=args.lr,  # 学习率
        weight_decay=args.weight_decay  # 权重衰减
    )

    if args.local_rank == 0:
        roott="/home/model"
        checkpoint_dir = os.path.join(roott, 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = process(train_list, train_root, args.mode)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))
    num_gpu = (len(args.gpu)+1) // 2

    train_loader = DataLoader(dataset=train_set, shuffle=False, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = process(valid_list, valid_root, 'valid')
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    torch.set_grad_enabled(True)
    es=0
    torch.cuda.empty_cache()
    epochs_no_improve = 0
    labels = [0, 1]  # 假设是二分类问题
    idh_names = ['IDH野生型', 'IDH突变型']
    grade_names = ['LGG', 'HGG']
    best_avg_auc = 0.0
    for epoch in range(args.start_epoch, args.end_epoch):
        es += 1
        # 每个epoch开始时调整一次学习率
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        print(f"Epoch {epoch} / {args.end_epoch - 1}")
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        print(optimizer.param_groups[0]['lr'])
        # 设为训练模式
        nets['en'].train()
        nets['mtl'].train()
        nets['t2f'].train()
        nets['idh'].train()
        nets['grade'].train()

        idh_pred_scores = []
        grade_pred_scores = []
        idh_preds = []
        grade_preds = []
        idh_trues = []
        grade_trues = []

        optimizer.zero_grad()
        train_loss_sum = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader):
            x, radiomic, idh, grade = data
            x = x.cuda(args.local_rank, non_blocking=True)
            radiomic = radiomic.cuda(args.local_rank, non_blocking=True)
            idh = idh.cuda(args.local_rank, non_blocking=True)
            grade = grade.cuda(args.local_rank, non_blocking=True)
            flair_image = x[:, 0, :, :, :].unsqueeze(1)
            t2_image = x[:, 3, :, :, :].unsqueeze(1)
            t2f=nets['t2f'](flair_image,t2_image)
            en_feature=nets['en'](x,t2f,radiomic)
            grade_pre=nets['grade'](en_feature)
            idh_pre=nets['idh'](en_feature)
            idh_loss=criterion1(idh_pre,idh)
            grade_loss=criterion2(grade_pre,grade)
            loss=nets['mtl'](idh_loss,grade_loss)
            train_loss_sum += loss.item()
            num_batches += 1
            idh_pred_score = F.softmax(idh_pre, dim=1)[:, 1].detach().cpu().numpy()
            grade_pred_score = F.softmax(grade_pre, dim=1)[:, 1].detach().cpu().numpy()
            idh_pred = torch.argmax(idh_pre, dim=1).detach().cpu().numpy()
            grade_pred = torch.argmax(grade_pre, dim=1).detach().cpu().numpy()
            idh_true = idh.detach().cpu().numpy()
            grade_true = grade.detach().cpu().numpy()
            idh_pred_scores.append(idh_pred_score)
            grade_pred_scores.append(grade_pred_score)
            idh_preds.append(idh_pred)
            grade_preds.append(grade_pred)
            idh_trues.append(idh_true)
            grade_trues.append(grade_true)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        idh_trues = np.concatenate(idh_trues).tolist()
        idh_preds = np.concatenate(idh_preds).tolist()
        idh_pred_scores = np.concatenate(idh_pred_scores).tolist()

        grade_trues = np.concatenate(grade_trues).tolist()
        grade_preds = np.concatenate(grade_preds).tolist()
        grade_pred_scores = np.concatenate(grade_pred_scores).tolist()

        print('idh真实的结果')
        print(idh_trues)
        print('idh预测的结果')
        print(idh_preds)

        print('grade真实的结果')
        print(grade_trues)
        print('grade预测的结果')
        print(grade_preds)

        results_idh_train = evalution_metirc_boostrap(
            y_true=idh_trues,
            y_pred_score=idh_pred_scores,
            y_pred=idh_preds,
            labels=labels,
            target_names=idh_names
        )

        results_grade_train = evalution_metirc_boostrap(
            y_true=grade_trues,
            y_pred_score=grade_pred_scores,
            y_pred=grade_preds,
            labels=labels,
            target_names=grade_names
        )

        epoch_train_loss = train_loss_sum / max(1, num_batches)
        print(f"[Epoch {epoch}] Average Train Loss: {epoch_train_loss:.6f}")

        with torch.no_grad():
            nets['en'].eval()
            nets['t2f'].eval()
            nets['idh'].eval()
            nets['mtl'].eval()
            nets['grade'].eval()

            idh_pred_scores_val = []
            grade_pred_scores_val = []
            idh_preds_val = []
            grade_preds_val = []
            idh_trues_val = []
            grade_trues_val = []

            for i, data in enumerate(valid_loader):
                x, radiomic, idh, grade = data
                x = x.cuda(args.local_rank, non_blocking=True)
                radiomic = radiomic.cuda(args.local_rank, non_blocking=True)
                idh = idh.cuda(args.local_rank, non_blocking=True)
                grade = grade.cuda(args.local_rank, non_blocking=True)
                flair_image = x[:, 0, :, :, :].unsqueeze(1)
                t2_image = x[:, 3, :, :, :].unsqueeze(1)
                t2f = nets['t2f'](flair_image, t2_image)
                en_feature = nets['en'](x, t2f, radiomic)
                grade_pre = nets['grade'](en_feature)
                idh_pre = nets['idh'](en_feature)
                idh_pred_score = F.softmax(idh_pre, dim=1)[:, 1].detach().cpu().numpy()
                grade_pred_score = F.softmax(grade_pre, dim=1)[:, 1].detach().cpu().numpy()
                idh_pred = torch.argmax(idh_pre, dim=1).detach().cpu().numpy()
                grade_pred = torch.argmax(grade_pre, dim=1).detach().cpu().numpy()
                idh_true = idh.detach().cpu().numpy()
                grade_true = grade.detach().cpu().numpy()
                idh_pred_scores_val.append(idh_pred_score)
                grade_pred_scores_val.append(grade_pred_score)
                idh_preds_val.append(idh_pred)
                grade_preds_val.append(grade_pred)
                idh_trues_val.append(idh_true)
                grade_trues_val.append(grade_true)

            idh_trues_val = np.concatenate(idh_trues_val).tolist()
            idh_preds_val = np.concatenate(idh_preds_val).tolist()
            idh_pred_scores_val = np.concatenate(idh_pred_scores_val).tolist()

            grade_trues_val = np.concatenate(grade_trues_val).tolist()
            grade_preds_val = np.concatenate(grade_preds_val).tolist()
            grade_pred_scores_val = np.concatenate(grade_pred_scores_val).tolist()

            print('验证集idh真实的结果')
            print(idh_trues_val)
            print('验证集idh预测的结果')
            print(idh_preds_val)

            print('验证集grade真实的结果')
            print(grade_trues_val)
            print('验证集grade预测的结果')
            print(grade_preds_val)

            results_idh_val = evalution_metirc_boostrap(
                y_true=idh_trues_val,
                y_pred_score=idh_pred_scores_val,
                y_pred=idh_preds_val,
                labels=labels,
                target_names=idh_names
            )

            results_grade_val = evalution_metirc_boostrap(
                y_true=grade_trues_val,
                y_pred_score=grade_pred_scores_val,
                y_pred=grade_preds_val,
                labels=labels,
                target_names=grade_names
            )

            current_idh_auc = results_idh_val['AUC'][0]
            current_grade_auc = results_grade_val['AUC'][0]
            current_avg_auc = (current_idh_auc + current_grade_auc) / 2  # 平均AUC
            print(
                f"当前验证集IDH AUC: {current_idh_auc:.4f}, Grade AUC: {current_grade_auc:.4f}, 平均AUC: {current_avg_auc:.4f}")

            # 核心修改5：当平均AUC为历史最高时保存模型
            if current_avg_auc > best_avg_auc:
                best_avg_auc = current_avg_auc  # 更新最佳平均AUC
                epochs_no_improve = 0  # 重置无提升计数器
                if args.local_rank == 0:  # 仅主进程保存模型
                    model_filename = f"best_model_epoch_{epoch}_avg_auc_{best_avg_auc:.4f}.pth"
                    model_path = os.path.join(checkpoint_dir, model_filename)
                    torch.save({
                        'epoch': epoch,
                        'best_avg_auc': best_avg_auc,
                        'idh_auc': current_idh_auc,
                        'grade_auc': current_grade_auc,
                        't2f': nets['t2f'].state_dict(),
                        'en': nets['en'].state_dict(),
                        'idh': nets['idh'].state_dict(),
                        'grade': nets['grade'].state_dict(),
                        'mtl': nets['mtl'].state_dict(),
                        'optimizer': optimizer.state_dict(),  # 保存优化器状态，方便续训
                    }, model_path)
                    print(f"✅ 平均AUC刷新至{best_avg_auc:.4f}，模型已保存至 {model_path}")
            else:
                epochs_no_improve += 1
                print(f"❌ 平均AUC未提升（当前：{current_avg_auc:.4f}，最高：{best_avg_auc:.4f}）")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
