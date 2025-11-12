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
parser.add_argument('--root', default='/home/lixinyu/UCSF_MTSAM', type=str)

parser.add_argument('--train_dir', default='train', type=str)

parser.add_argument('--valid_dir', default='val', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='val.txt', type=str)

parser.add_argument('--dataset', default='UCSF-PDGM', type=str)

parser.add_argument('--model_name', default='UCSF', type=str)

# Training Information
parser.add_argument('--lr', default=0.00002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=1, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=2000, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--point_method', default='default', type=str)

parser.add_argument('--crop_size', type=int, default=128)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()
click_methods = {
    'default': get_next_click3D_torch_2,
    'ritm': get_next_click3D_torch_2,
    'random': get_next_click3D_torch_2,
}
def CoxLoss(hazard_pred,survtime, censor):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(censor.device)#censor.device
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox
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
def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子

def c_index(pred, ytime, yevent):
    '''Calculate concordance index to evaluate models.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        concordance_index: c-index (between 0 and 1).
    '''
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    ###T_i is uncensored
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    ###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i] = 1
            elif pred[i] == pred[j]:
                pred_matrix[j, i] = 0.5

    concord_matrix = pred_matrix.mul(ytime_matrix)
    ###numerator
    concord = torch.sum(concord_matrix)
    ###denominator
    epsilon = torch.sum(ytime_matrix)
    ###c-index = numerator/denominator
    concordance_index = torch.div(concord, epsilon)
    ###if gpu is being used
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    ###
    return concordance_index
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


def get_iou_score(prev_masks, gt3D):
    def compute_iou(mask_pred, mask_gt):
        in_mask = np.logical_and(mask_gt, mask_pred)
        out_mask = np.logical_or(mask_gt, mask_pred)
        iou = np.sum(in_mask) / np.sum(out_mask)
        return iou

    pred_masks = (prev_masks > 0.5).cpu().numpy()  # Assuming tensor, convert to boolean numpy array
    true_masks = (gt3D > 0).cpu().numpy()          # Assuming tensor, convert to boolean numpy array
    iou_list = []
    for i in range(true_masks.shape[0]):
        iou_list.append(compute_iou(pred_masks[i], true_masks[i]))
    return sum(iou_list) / len(iou_list)
def get_dice_score(prev_masks, gt3D):
    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    pred_masks = (prev_masks > 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
    return (sum(dice_list) / len(dice_list)).item()
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


def _extract_metric_triplet(res_dict, key_aliases):
    """
    从结果字典里取 metric。支持不同命名别名；返回 (mean, low, high)，若无 CI 则 low/high 为 None。
    兼容:
      - res_dict['Accuracy'] = [mean, low, high]
      - res_dict['Accuracy'] = (mean, low, high)
      - res_dict['Accuracy'] = mean
    """
    for k in key_aliases:
        if k in res_dict:
            v = res_dict[k]
            if isinstance(v, (list, tuple)):
                if len(v) >= 3:
                    return float(v[0]), float(v[1]), float(v[2])
                elif len(v) == 2:
                    return float(v[0]), float(v[1]), None
                elif len(v) == 1:
                    return float(v[0]), None, None
            try:
                return float(v), None, None
            except Exception:
                pass
    return None, None, None

def _format_line(name, triplet):
    mean, lo, hi = triplet
    if mean is None:
        return f"{name}: N/A\n"
    if lo is None or hi is None:
        return f"{name}: {mean:.4f}\n"
    return f"{name}: {mean:.4f} (95% CI {lo:.4f}–{hi:.4f})\n"


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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score,accuracy_score
from sklearn.utils import resample
import numpy as np
def evalution_metirc_boostrap(y_true, y_pred_score, y_pred, labels, target_names):
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)
    y_pred = np.array(y_pred)
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))

    auc_ = roc_auc_score(y_true, y_pred_score)
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)

    accuracy_ = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    specificity_ = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity_ = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    F1_score_ = f1_score(y_true, y_pred, labels=labels, pos_label=1)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_AUC = []
    bootstrapped_ACC = []
    bootstrapped_SEN = []
    bootstrapped_SPE = []
    bootstrapped_F1 = []
    rng = np.random.RandomState(rng_seed)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_score), len(y_pred_score))
        if len(np.unique(y_true[indices.astype(int)])) < 2:
            # We need at least one positive and one negative sample for ROC AUC to be defined: reject the sample
            continue
        auc = roc_auc_score(y_true[indices], y_pred_score[indices])
        bootstrapped_AUC.append(auc)

        confusion = confusion_matrix(y_true[indices], y_pred[indices])
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        F1_score = f1_score(y_true[indices], y_pred[indices], labels=labels, pos_label=1)

        bootstrapped_ACC.append(accuracy)
        bootstrapped_SPE.append(specificity)
        bootstrapped_SEN.append(sensitivity)
        bootstrapped_F1.append(F1_score)

    sorted_AUC = np.array(bootstrapped_AUC)
    sorted_AUC.sort()
    sorted_ACC = np.array(bootstrapped_ACC)
    sorted_ACC.sort()
    sorted_SPE = np.array(bootstrapped_SPE)
    sorted_SPE.sort()
    sorted_SEN = np.array(bootstrapped_SEN)
    sorted_SEN.sort()
    sorted_F1 = np.array(bootstrapped_F1)
    sorted_F1.sort()

    results = {
        'AUC': (auc_, sorted_AUC[int(0.05 * len(sorted_AUC))], sorted_AUC[int(0.95 * len(sorted_AUC))]),
        'Accuracy': (accuracy_, sorted_ACC[int(0.05 * len(sorted_ACC))], sorted_ACC[int(0.95 * len(sorted_ACC))]),
        'Specificity': (specificity_, sorted_SPE[int(0.05 * len(sorted_SPE))], sorted_SPE[int(0.95 * len(sorted_SPE))]),
        'Sensitivity': (sensitivity_, sorted_SEN[int(0.05 * len(sorted_SEN))], sorted_SEN[int(0.95 * len(sorted_SEN))]),
        'F1_score': (F1_score_, sorted_F1[int(0.05 * len(sorted_F1))], sorted_F1[int(0.95 * len(sorted_F1))])
    }

    print("Confidence interval for the AUC: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['AUC']))
    print("Confidence interval for the Accuracy: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Accuracy']))
    print("Confidence interval for the Specificity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Specificity']))
    print("Confidence interval for the Sensitivity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Sensitivity']))
    print("Confidence interval for the F1_score: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['F1_score']))

    return results


def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()