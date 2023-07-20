import os
import torch as pt
from dp_mepf_args import get_args, get_imagenet_norm_min_and_range, get_param_group_tuples
from util_logging import configure_logger, log_losses_and_imgs, route_io_to_file, delayed_log, \
  LOG, log_fake_feature_norms, log_synth_data_eval, log_iteration, log_args, BestResult, \
  note_best_scores_column
from util import set_random_seed, create_checkpoint, load_checkpoint, get_optimizers, \
  GeneratorNoiseMaker, LossAccumulator, MovingAverages, Embeddings
from feature_matching import adam_moving_average_update, extract_and_bound_features, \
  compute_data_embedding, sort_feats_by_label, get_test_data_embedding, \
  hybrid_labeled_batch_embedding, l2_norm_feat_analysis, get_static_losses_valid, \
  get_number_of_matching_layers_pytorch
from model_builder import get_generator, get_mean_and_var_nets, get_torchvision_encoders
from torch.utils.tensorboard import SummaryWriter
from dp_functions import dp_dataset_feature_release
from data_loading import load_dataset
from encoders_class import Encoders


def zero_grads(net_gen, m_avg):
  net_gen.zero_grad(set_to_none=True)
  m_avg.mean.zero_grad(set_to_none=True)
  m_avg.var.zero_grad(set_to_none=True)


def get_real_data_embedding(ckpt, encoders, n_matching_layers, device, train_loader, writer,
                            channel_ids_by_enc, dp_params,
                            match_with_top_layers, matched_moments, n_classes, feat_emb,
                            dataset, image_size, center_crop_size, dataroot,
                            batch_size, n_workers, data_scale,
                            val_encoders, val_data, val_dp_params, val_moments,
                            do_embedding_norms_analysis=False):
  if ckpt is None:  # compute true data embeddings
    emb_res = compute_data_embedding(encoders, n_matching_layers, device, train_loader, writer,
                                     channel_ids_by_enc, dp_params,
                                     match_with_top_layers, matched_moments, n_classes)
    feat_emb.real_means, feat_emb.real_vars, n_samples, feat_l2_norms = emb_res
    if isinstance(encoders, Encoders) and do_embedding_norms_analysis:
      l2_norm_feat_analysis(feat_emb.real_means, feat_emb.real_vars, encoders.n_feats_by_layer,
                            encoders.layer_feats, writer)
    if dp_params.noise is not None:
      dp_res = dp_dataset_feature_release(feat_emb, feat_l2_norms, dp_params, matched_moments,
                                          n_samples, writer)
      feat_emb.real_means, feat_emb.real_vars = dp_res

  else:
    feat_emb.real_means = ckpt['feat_means']
    feat_emb.real_vars = ckpt['feat_vars']

  if ckpt is None:
    n_classes_val = None  # don't use labeled embedding to check for fid quality
    labeled_val = False
    test_emb = get_test_data_embedding(val_encoders, n_matching_layers, device,
                                       channel_ids_by_enc, match_with_top_layers,
                                       val_dp_params, val_moments, data_scale, n_classes_val,
                                       dataset, image_size, center_crop_size, dataroot,
                                       batch_size, n_workers, labeled_val, val_data)
    feat_emb.means_valid, feat_emb.vars_valid = test_emb
  else:
    feat_emb.means_valid = ckpt['feat_means_valid']
    feat_emb.vars_valid = ckpt['feat_vars_valid']


def get_fake_data_embedding(fake_data, encoders, feat_emb, dp_params, gen_labels, n_classes,
                            n_matching_layers, match_with_top_layers, channel_ids_by_enc,
                            do_second_moment, tb_log_step, step, writer):
  ef_res = extract_and_bound_features(fake_data, encoders, n_matching_layers,
                                      match_with_top_layers, dp_params,
                                      channel_ids_by_enc, do_second_moment, detach_output=False,
                                      compute_norms=tb_log_step)

  if isinstance(encoders, Encoders) and encoders.n_split_layers is not None:
    feats_batch, feats_sqrd_batch, l2_norms, l2_norms_sqrd = ef_res
    ff, ff_sq = hybrid_labeled_batch_embedding(encoders, feats_batch, feats_sqrd_batch, l2_norms,
                                               do_second_moment, n_classes, gen_labels)
    feat_emb.fake_feats, feat_emb.fake_feats_sqrd = ff, ff_sq

  else:
    feat_emb.fake_feats, feat_emb.fake_feats_sqrd, l2_norms, l2_norms_sqrd = ef_res
    if n_classes is not None:
      feat_emb.fake_feats = sort_feats_by_label(feat_emb.fake_feats, gen_labels)
      if feat_emb.fake_feats_sqrd is not None:
        feat_emb.fake_feats_sqrd = sort_feats_by_label(feat_emb.fake_feats_sqrd, gen_labels)

  if tb_log_step:
    log_fake_feature_norms(l2_norms, l2_norms_sqrd, step, writer)

  return feat_emb


def eval_stop_criterion(stop_criterion_scores, dataset):
  lower_is_better = False if dataset in {'dmnist', 'fmnist'} else True

  if stop_criterion_scores[-1] > stop_criterion_scores[-2]:  # score going up
    stop_now = lower_is_better  # stop if score should go down
  else:
    stop_now = not lower_is_better
  return stop_now


def delete_old_syn_data(syn_data_file, old_syn_data_file, best_proxy_result, step, keep_best_syn_data):
  if keep_best_syn_data:
    # remove syn data files that are not among the best results
    if old_syn_data_file is not None:
      if best_proxy_result.step == step:
        try:
          os.remove(old_syn_data_file)
        except FileNotFoundError as fnf_e:
            LOG.warning(f'failed to delete old syn data file: {fnf_e}')
        old_syn_data_file = syn_data_file
      else:
        try:
          os.remove(syn_data_file)
        except FileNotFoundError as fnf_e:
            LOG.warning(f'failed to delete old syn data file: {fnf_e}')
    else:
      old_syn_data_file = syn_data_file
  else:
    # remove syn data files
    try:
      os.remove(syn_data_file)
    except FileNotFoundError as fnf_e:
        LOG.warning(f'failed to delete old syn data file: {fnf_e}')
  return old_syn_data_file


def update_best_score(eval_score, step, syn_data_file, dataset, best_result: BestResult):
  lower_is_better = False if dataset in {'dmnist', 'fmnist'} else True
  score_is_better = best_result.score is None or ((eval_score < best_result.score) == lower_is_better)

  # is score was not better than before, save last best result as first local optimum
  if best_result.score is None or score_is_better:
    best_result.score = eval_score
    best_result.step = step
    best_result.data_file = syn_data_file


def update_best_proxy_score(proxy_score, step, syn_data_file,
                            best_proxy_result: BestResult):
  if proxy_score is None:
    return

  lower_is_better = True  # always loss based
  score_is_better = best_proxy_result.score is None or (
        (proxy_score < best_proxy_result.score) == lower_is_better)

  if best_proxy_result.score is None or score_is_better:
    best_proxy_result.score = proxy_score
    best_proxy_result.step = step
    best_proxy_result.data_file = syn_data_file


def load_best_result(best_result: BestResult, best_proxy_result: BestResult, ckpt):
  best_result.score = ckpt['best_score']
  best_result.step = ckpt['best_step']
  best_result.data_file = ckpt['best_syn_data_file']

  if best_proxy_result is not None:
    best_proxy_result.score = ckpt['best_proxy_score']
    best_proxy_result.step = ckpt['best_proxy_step']
    best_proxy_result.data_file = ckpt['best_proxy_syn_data_file']


def get_enc_input_scalings(net_enc_type, dataset, image_size, data_scale, extra_input_scaling):
  if extra_input_scaling == 'none':
    return None

  assert data_scale == '0_1'
  scalings_by_dataset = {'imagenet': '0_1_to_IMGNet_Norm',
                         'cifar10': '0_1_to_Cifar10_Norm',
                         'celeba32': '0_1_to_Celeba32_Norm',
                         'celeba64': '0_1_to_Celeba64_Norm'}

  # pick the right scaling name
  if extra_input_scaling == 'dataset_norm':
    if dataset == 'cifar10':
      scaling = scalings_by_dataset['cifar10']
    elif dataset == 'celeba':
      if image_size == 32:
        scaling = scalings_by_dataset['celeba32']
      elif image_size == 64:
        scaling = scalings_by_dataset['celeba64']
      else:
        raise ValueError
    else:
      raise ValueError

  elif extra_input_scaling == 'imagenet_norm':
    scaling = scalings_by_dataset['imagenet']
  else:
    raise ValueError

  scaling_by_model = dict()
  for net_enc_name in net_enc_type:
    scaling_by_model[net_enc_name] = scaling
  return scaling_by_model


def main():
  arg = get_args()
  os.makedirs(arg.log_dir, exist_ok=True)
  os.makedirs(os.path.join(arg.log_dir, 'images/'), exist_ok=True)
  os.makedirs(os.path.join(arg.log_dir, 'tensorboard/'), exist_ok=True)
  log_args(arg)
  device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
  dp_params, val_dp_params, event_steps = get_param_group_tuples(arg)
  if not arg.no_io_files:
    route_io_to_file(arg.log_dir, arg.stdout_file, arg.stderr_file)
  configure_logger(arg.log_importance_level)
  for level, message in arg.log_messages:
    delayed_log(level, message)
  n_classes_val = None  # val loss always unlabeled

  ckpt, arg.first_batch_id = load_checkpoint(arg.log_dir, arg.first_batch_id)

  LOG.info("Args: {}".format(arg))
  LOG.info(f'running rank {device} on device: {pt.cuda.get_device_name(device)}')

  best_result = BestResult()
  best_proxy_result = BestResult()
  if ckpt is not None:
    load_best_result(best_result, best_proxy_result, ckpt)

  writer = None if arg.no_tensorboard else SummaryWriter(log_dir=os.path.join(arg.log_dir,
                                                                              'tensorboard/'))

  set_random_seed(arg.manual_seed)

  if ckpt is not None and arg.n_features_in_enc is not None \
          and arg.n_classes is not None and arg.n_split_layers is None:
    n_classes = arg.n_classes if arg.n_classes > 0 else None
    n_features_in_enc = arg.n_features_in_enc
    train_loader = None
  else:
    train_loader, n_classes = load_dataset(arg.dataset, arg.image_size, arg.center_crop_size,
                                           arg.dataroot, arg.batch_size, arg.n_workers,
                                           arg.data_scale, arg.labeled)
    n_features_in_enc = None

  train_enc_input_scalings = get_enc_input_scalings(arg.net_enc_type, arg.dataset, arg.image_size,
                                                    arg.data_scale, arg.extra_input_scaling)

  encoders = get_torchvision_encoders(arg.net_enc_type, arg.image_size, device,
                                      arg.pretrain_dataset, arg.n_classes_in_enc,
                                      arg.n_split_layers, n_classes, train_enc_input_scalings, arg.dataset)

  if arg.val_enc is not None:
    val_encoders = get_torchvision_encoders(arg.val_enc, arg.image_size, device,
                                            'imagenet', arg.n_classes_in_enc,
                                            arg.n_split_layers, n_classes, None, arg.dataset)
  else:
    val_encoders = encoders

  if n_features_in_enc is None:
    get_number_of_matching_layers_pytorch(encoders, device, train_loader)
    n_features_in_enc = encoders.n_feats_total
  n_matching_layers = None
  channel_ids_by_enc = None

  LOG.info(f'# n_features_total: {n_features_in_enc}')
  # fixed_noise = fixed_noise.to(device)
  gen = get_generator(arg.image_size, arg.z_dim, arg.gen_output, device, ckpt)

  noise_maker = GeneratorNoiseMaker(arg.batch_size, arg.z_dim, device, n_classes, ckpt)
  fixed_noise = noise_maker.get_fixed_noise()

  feat_emb = Embeddings()

  do_second_moment = arg.matched_moments in {'mean_and_var', 'm1_and_m2'}
  do_second_moment_val = do_second_moment
  val_moments = arg.matched_moments

  if arg.val_enc[0] == 'fid_features':
    do_second_moment_val = True
    val_moments = 'm1_and_m2'
  # load train and optionally valid data embedding into feat_emb
  get_real_data_embedding(ckpt, encoders, n_matching_layers, device, train_loader, writer,
                          channel_ids_by_enc, dp_params,
                          arg.match_with_top_layers, arg.matched_moments, n_classes, feat_emb,
                          arg.dataset, arg.image_size, arg.center_crop_size, arg.dataroot,
                          arg.batch_size, arg.n_workers,
                          arg.data_scale, val_encoders, arg.val_data, val_dp_params, val_moments,
                          arg.do_embedding_norms_analysis)

  m_avg = MovingAverages()

  m_avg.mean, m_avg.var = get_mean_and_var_nets(n_features_in_enc, device, ckpt, n_classes)

  optimizers = get_optimizers(gen, arg.lr, m_avg, arg.m_avg_lr, arg.beta1, ckpt)
  del ckpt  # loading done

  acc_losses = LossAccumulator()

  LOG.info('starting training loop')

  old_syn_data_file = best_proxy_result.data_file
  #######################################################
  # GFMN Training Loop.
  #######################################################
  for step in range(arg.first_batch_id + 1, event_steps.final + 1):
    if writer is None or step % event_steps.tb_log != 0:
      tb_log_step = False
    else:
      tb_log_step = True

    zero_grads(gen, m_avg)

    gen_noise, gen_labels = noise_maker.generator_noise()
    fake_data = gen(gen_noise)

    if arg.pretrain_dataset == 'imagenet' and arg.gen_output == 'tanh' \
        and arg.data_scale == 'normed':
      # normalizes the generated images using imagenet min-max ranges
      imagenet_norm_min, imagenet_norm_range = get_imagenet_norm_min_and_range(device)
      fake_data = (((fake_data + 1) * imagenet_norm_range) / 2) + imagenet_norm_min

    # extract features from FAKE data
    feat_emb = get_fake_data_embedding(fake_data, encoders, feat_emb, dp_params, gen_labels,
                                       n_classes, n_matching_layers, arg.match_with_top_layers,
                                       channel_ids_by_enc, do_second_moment, tb_log_step, step,
                                       writer)

    adam_moving_average_update(feat_emb, acc_losses, m_avg, optimizers, arg.matched_moments)

    gen.eval()
    if step % event_steps.eval == 0:  # note this is all train loss, not validation!
      log_iteration(arg.exp_name, step)

      n_val_samples = arg.fid_dataset_size
      lm, lv = get_static_losses_valid(feat_emb.means_valid, feat_emb.vars_valid, gen,
                                       val_encoders,
                                       noise_maker, arg.batch_size, n_val_samples, n_classes_val,
                                       device, n_matching_layers, arg.match_with_top_layers,
                                       dp_params, channel_ids_by_enc, do_second_moment_val,
                                       writer, step)
      static_val_loss = lm + lv

      log_losses_and_imgs(gen, acc_losses, fixed_noise,
                          step, writer, event_steps.final,
                          event_steps.eval, arg.log_dir, arg.exp_name)
      acc_losses = LossAccumulator()

      syn_data_file, eval_score = log_synth_data_eval(gen, writer, step, noise_maker, device,
                                                      arg.dataset, arg.synth_dataset_size,
                                                      arg.batch_size, arg.log_dir,
                                                      n_classes, arg.fid_dataset_size,
                                                      arg.image_size, arg.center_crop_size,
                                                      arg.data_scale, arg.local_fid_eval_storage, arg.skip_prdc,
                                                      final_step=False)

      update_best_score(eval_score, step, syn_data_file, arg.dataset, best_result)
      update_best_proxy_score(static_val_loss, step, syn_data_file, best_proxy_result)

      old_syn_data_file = delete_old_syn_data(syn_data_file, old_syn_data_file, best_proxy_result, step,
                                              arg.keep_best_syn_data)

    if step % event_steps.ckpt == 0:
      create_checkpoint(feat_emb, gen, m_avg, optimizers, step, fixed_noise, arg.log_dir,
                        event_steps.new_ckpt, best_result, best_proxy_result)

    if (event_steps.restart is not None) and (step % event_steps.restart == 0) \
            and event_steps.restart > 0 and step != event_steps.final:
      LOG.info(f'preparing restart at step {step}')
      exit(3)

    gen.train()

  LOG.info('completed training')
  gen.eval()
  if arg.synth_dataset_size is not None:
    syn_data_file, eval_score = log_synth_data_eval(gen, writer, event_steps.final, noise_maker,
                                                    device, arg.dataset, arg.synth_dataset_size,
                                                    arg.batch_size, arg.log_dir, n_classes,
                                                    arg.fid_dataset_size, arg.image_size,
                                                    arg.center_crop_size, arg.data_scale,
                                                    arg.local_fid_eval_storage, arg.skip_prdc, final_step=True)
    update_best_score(eval_score, arg.n_iter, syn_data_file, arg.dataset, best_result)

    n_val_samples = arg.fid_dataset_size
    lm, lv = get_static_losses_valid(feat_emb.means_valid, feat_emb.vars_valid, gen,
                                     val_encoders, noise_maker, arg.batch_size, n_val_samples,
                                     n_classes_val, device, n_matching_layers,
                                     arg.match_with_top_layers, dp_params, channel_ids_by_enc,
                                     do_second_moment_val, writer, arg.n_iter)
    static_val_loss = lm + lv
    update_best_proxy_score(static_val_loss, arg.n_iter, syn_data_file, best_proxy_result)
    os.remove(syn_data_file)

  if writer is not None:
    writer.close()
  LOG.info(f'{best_result}, {best_proxy_result}')
  note_best_scores_column(best_result, arg.log_dir, is_proxy=False)
  if best_proxy_result.score is not None:
    note_best_scores_column(best_proxy_result, arg.log_dir, is_proxy=True)

  log_iteration(arg.exp_name, arg.n_iter, is_final=True)


if __name__ == '__main__':
  main()
