import os
import torch as pt
from dp_mepf_args import get_args, get_imagenet_norm_min_and_range, get_param_group_tuples
from util_logging import configure_logger, log_losses_and_imgs, route_io_to_file, delayed_log, \
  LOG, log_fake_feature_norms, log_synth_data_eval, log_iteration, log_args, BestResult, \
  note_best_scores_column, log_staticdset_synth_data_eval
from util import set_random_seed, create_checkpoint, load_checkpoint, get_optimizers, \
  GeneratorNoiseMaker, LossAccumulator, MovingAverages, Embeddings, get_validation_optimizers
from feature_matching import ma_update, extract_and_bound_features, \
  compute_data_embedding, sort_feats_by_label, get_test_data_embedding, \
  hybrid_labeled_batch_embedding, l2_norm_feat_analysis, get_static_losses_valid, \
  get_staticdset_static_losses_valid
from feature_selection import get_number_of_matching_layers_pytorch, select_pruned_channels, \
  get_number_of_matching_features
from models.model_builder import get_encoders, get_generator, get_mean_and_var_nets, \
  get_torchvision_encoders, StaticDataset
from torch.utils.tensorboard import SummaryWriter
from dp_functions import dp_dataset_feature_release
from data_loading import load_dataset
from models.encoders_class import Encoders
import torch.distributed as dist
import torch.multiprocessing as mp


def zero_grads(net_gen, m_avg, update_type):
  net_gen.zero_grad(set_to_none=True)
  if update_type == 'adam_ma':
    m_avg.mean.zero_grad(set_to_none=True)
    m_avg.var.zero_grad(set_to_none=True)


def get_real_data_embedding(ckpt, encoders, n_matching_layers, device, train_loader, writer,
                            channel_ids_by_enc, dp_params, no_cuda,
                            match_with_top_layers, matched_moments, n_classes, feat_emb,
                            dataset, image_size, center_crop_size, dataroot,
                            batch_size, n_workers, labeled, validation_mode, data_scale,
                            val_encoders, val_data, val_dp_params, val_moments,
                            do_embedding_norms_analysis=False):
  if ckpt is None:  # compute true data embeddings
    emb_res = compute_data_embedding(encoders, n_matching_layers, device, train_loader, writer,
                                     channel_ids_by_enc, dp_params, no_cuda,
                                     match_with_top_layers, matched_moments, n_classes)
    feat_emb.real_means, feat_emb.real_vars, n_samples, feat_l2_norms = emb_res
    if isinstance(encoders, Encoders) and do_embedding_norms_analysis:
      l2_norm_feat_analysis(feat_emb.real_means, feat_emb.real_vars, encoders.n_feats_by_layer,
                            encoders.layer_feats, writer)
    if dp_params.noise is not None:
      dp_res = dp_dataset_feature_release(feat_emb, feat_l2_norms, dp_params, matched_moments,
                                          n_samples, writer)
      feat_emb.real_means, feat_emb.real_vars = dp_res

    dist.broadcast(feat_emb.real_means, src=0, async_op=False)
    if matched_moments != 'mean':
      dist.broadcast(feat_emb.real_vars, src=0, async_op=False)

  else:
    feat_emb.real_means = ckpt['feat_means']
    feat_emb.real_vars = ckpt['feat_vars']

  if validation_mode != 'off':
    if ckpt is None:
      n_classes_val = None  # don't use labeled embedding to check for fid quality
      labeled_val = False
      test_emb = get_test_data_embedding(val_encoders, n_matching_layers, device,
                                         channel_ids_by_enc, no_cuda, match_with_top_layers,
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


def delete_old_syn_data(syn_data_file, old_syn_data_file, best_result, step, keep_best_syn_data):
  if keep_best_syn_data:
    # remove syn data files that are not among the best results
    if old_syn_data_file is not None:
      if best_result.step == step:
        os.remove(old_syn_data_file)
        old_syn_data_file = syn_data_file
      else:
        os.remove(syn_data_file)
    else:
      old_syn_data_file = syn_data_file
  else:
    # remove syn data files
    os.remove(syn_data_file)
  return old_syn_data_file


def update_best_score(eval_score, step, syn_data_file, dataset, best_result: BestResult):
  lower_is_better = False if dataset in {'dmnist', 'fmnist'} else True
  score_is_better = best_result.score is None or ((eval_score < best_result.score) == lower_is_better)

  # is score was not better than before, save last best result as first local optimum
  if best_result.first_local_optimum_score is None and not score_is_better:
    best_result.first_local_optimum_score = best_result.score
    best_result.first_local_optimum_step = best_result.step
    best_result.first_local_optimum_data_file = best_result.data_file

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

  if best_proxy_result.first_local_optimum_score is None and not score_is_better:
    best_proxy_result.first_local_optimum_score = best_proxy_result.score
    best_proxy_result.first_local_optimum_step = best_proxy_result.step
    best_proxy_result.first_local_optimum_data_file = best_proxy_result.data_file

  if best_proxy_result.score is None or score_is_better:
    best_proxy_result.score = proxy_score
    best_proxy_result.step = step
    best_proxy_result.data_file = syn_data_file


def load_best_result(best_result: BestResult, best_proxy_result: BestResult, ckpt):
  best_result.score = ckpt['best_score']
  best_result.step = ckpt['best_step']
  best_result.data_file = ckpt['best_syn_data_file']
  best_result.first_local_optimum_score = ckpt['first_best_score']
  best_result.first_local_optimum_step = ckpt['first_best_step']
  best_result.first_local_optimum_data_file = ckpt['first_best_syn_data_file']

  if best_proxy_result is not None:
    best_proxy_result.score = ckpt['best_proxy_score']
    best_proxy_result.step = ckpt['best_proxy_step']
    best_proxy_result.data_file = ckpt['best_proxy_syn_data_file']
    best_proxy_result.first_local_optimum_score = ckpt['first_best_proxy_score']
    best_proxy_result.first_local_optimum_step = ckpt['first_best_proxy_step']
    best_proxy_result.first_local_optimum_data_file = ckpt['first_best_proxy_syn_data_file']


def ddp_setup(device, world_size, init_file_path):
  print(f'trying ddp init at {init_file_path}')
  dist.init_process_group(dist.Backend.NCCL, init_method=init_file_path,
                          rank=device, world_size=world_size)


def ddp_run(device, arg):
  dp_params, val_dp_params, event_steps = get_param_group_tuples(arg)
  if not arg.no_io_files:
    route_io_to_file(arg.log_dir, arg.stdout_file, arg.stderr_file)
  configure_logger(arg.log_importance_level)
  for level, message in arg.log_messages:
    delayed_log(level, message)
  ddp_setup(device, arg.n_gpu, arg.ddp_init_file)
  n_classes_val = None  # val loss always unlabeled

  ckpt, arg.first_batch_id = load_checkpoint(arg.log_dir, arg.first_batch_id, device)
  if device == 0:
    LOG.info("Args: {}".format(arg))
  LOG.info(f'running rank {device} on device: {pt.cuda.get_device_name(device)}')

  best_result = BestResult()
  best_proxy_result = BestResult()
  if ckpt is not None:
    load_best_result(best_result, best_proxy_result, ckpt)

  if arg.no_tensorboard or device > 0:
    writer = None
  else:
    writer = SummaryWriter(log_dir=os.path.join(arg.log_dir, 'tensorboard/'))

  set_random_seed(arg.manual_seed)

  train_loader, n_classes = load_dataset(arg.dataset, arg.image_size, arg.center_crop_size,
                                         arg.dataroot, arg.batch_size, arg.n_workers,
                                         arg.data_scale, arg.labeled)

  if arg.pytorch_encoders:
    encoders = get_torchvision_encoders(arg.net_enc_type, arg.image_size, device,
                                        arg.pretrain_dataset, arg.n_classes_in_enc,
                                        arg.n_split_layers, n_classes)
    if arg.val_enc is not None:
      val_encoders = get_torchvision_encoders(arg.val_enc, arg.image_size, device,
                                              arg.pretrain_dataset, arg.n_classes_in_enc,
                                              arg.n_split_layers, n_classes)
    else:
      val_encoders = encoders
  else:
    encoders = get_encoders(arg.net_enc_type, arg.net_enc, arg.image_size, arg.z_dim,
                            arg.n_classes_in_enc, device, arg.pretrain_dataset)
    val_encoders = encoders

  if arg.channel_filter_rate < 1.:
    need_labels = True if arg.channel_filter_mode == 'hsic' else False
    select_dataset = arg.pretrain_dataset if not arg.select_on_train_data else arg.dataset
    select_loader, _ = load_dataset(select_dataset, arg.image_size, arg.center_crop_size,
                                    arg.dataroot, arg.batch_size, arg.n_workers, arg.data_scale,
                                    labeled=need_labels)
  else:
    select_loader = None

  if isinstance(encoders, list):
    mf_res = get_number_of_matching_features(encoders, arg.n_matching_layers,
                                             arg.match_with_top_layers, device, writer,
                                             arg.channel_filter_rate, select_loader,
                                             arg.channel_filter_mode)
    n_features_in_enc, n_matching_layers, channel_ids_by_enc = mf_res
  else:
    get_number_of_matching_layers_pytorch(encoders, device, train_loader)
    if select_loader is not None:
      select_pruned_channels(encoders, device, writer, arg.channel_filter_rate, select_loader,
                             arg.channel_filter_mode, arg.n_filter_select_batches, ckpt,
                             arg.hsic_block_size, arg.hsic_reps, arg.hsic_max_samples,
                             arg.n_pca_iter)
    n_features_in_enc = encoders.n_feats_total
    n_matching_layers = None
    channel_ids_by_enc = None

  LOG.info(f'# n_features_total: {n_features_in_enc}')
  # fixed_noise = fixed_noise.to(device)
  gen = get_generator(arg.net_gen_type, arg.image_size, arg.z_dim, arg.gen_output, device,
                      ckpt, device)

  if isinstance(gen, StaticDataset):
    noise_maker = None
    fixed_noise = None
    n_dset_samples = gen.weights.shape[0]
    assert n_dset_samples % n_classes == 0
    one_hots = pt.eye(n_classes, dtype=pt.float32, device=device)
    gen.labels = one_hots.repeat_interleave(n_dset_samples // n_classes, dim=0)
  else:
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
                          channel_ids_by_enc, dp_params, arg.no_cuda,
                          arg.match_with_top_layers, arg.matched_moments, n_classes, feat_emb,
                          arg.dataset, arg.image_size, arg.center_crop_size, arg.dataroot,
                          arg.batch_size, arg.n_workers, arg.labeled, arg.validation_mode,
                          arg.data_scale, val_encoders, arg.val_data, val_dp_params, val_moments,
                          arg.do_embedding_norms_analysis)

  m_avg = MovingAverages()
  m_avg_valid = MovingAverages()
  if not arg.update_type == 'no_ma':
    m_avg.mean, m_avg.var = get_mean_and_var_nets(n_features_in_enc, device, ckpt, device,
                                                  n_classes)
    if arg.validation_mode == 'mavg':
      m_avg_valid.mean, m_avg_valid.var = get_mean_and_var_nets(n_features_in_enc, device, ckpt,
                                                                device, n_classes,
                                                                mean_ckpt_name='net_mean_valid',
                                                                var_ckpt_name='net_var_valid')

  optimizers = get_optimizers(gen, arg.lr, m_avg, arg.m_avg_lr, arg.beta1, ckpt)
  if arg.validation_mode == 'mavg':
    optimizers_valid = get_validation_optimizers(m_avg_valid, arg.m_avg_lr, arg.beta1, ckpt)
  else:
    optimizers_valid = None

  del ckpt  # loading done

  acc_losses = LossAccumulator()
  acc_losses_valid = LossAccumulator() if arg.validation_mode == 'mavg' else None

  if device == 0:
    LOG.info('starting training loop')

  old_syn_data_file = best_result.data_file
  static_val_loss = None
  #######################################################
  # GFMN Training Loop.
  #######################################################
  for step in range(arg.first_batch_id + 1, event_steps.final + 1):
    if writer is None or step % event_steps.tb_log != 0:
      tb_log_step = False
    else:
      tb_log_step = True

    zero_grads(gen, m_avg, arg.update_type)

    if isinstance(gen, StaticDataset):
      gen_labels = gen.labels
      fake_data = gen(None)
    else:
      gen_noise, gen_labels = noise_maker.generator_noise()
      fake_data = gen(gen_noise)

    do_rescaling = arg.gen_output == 'tanh' and arg.data_scale == 'normed'
    if arg.pretrain_dataset == 'imagenet' and do_rescaling:
      # normalizes the generated images using imagenet min-max ranges
      imagenet_norm_min, imagenet_norm_range = get_imagenet_norm_min_and_range(device)
      fake_data = (((fake_data + 1) * imagenet_norm_range) / 2) + imagenet_norm_min

    # extract features from FAKE data
    feat_emb = get_fake_data_embedding(fake_data, encoders, feat_emb, dp_params, gen_labels,
                                       n_classes, n_matching_layers, arg.match_with_top_layers,
                                       channel_ids_by_enc, do_second_moment, tb_log_step, step,
                                       writer)

    ma_update(feat_emb, m_avg, optimizers, acc_losses, m_avg_valid, optimizers_valid,
              acc_losses_valid, arg.update_type, arg.validation_mode == 'mavg',
              arg.matched_moments, arg.m_avg_alpha, gen)

    gen.eval()
    if device == 0:  # only log in first gpu
      if step % event_steps.valid == 0:  # note this is all train loss, not validation!
        log_iteration(arg.exp_name, step)
        if arg.validation_mode == 'static':
          n_val_samples = arg.fid_dataset_size
          if isinstance(gen, StaticDataset):
            lm, lv = get_staticdset_static_losses_valid(feat_emb.means_valid, feat_emb.vars_valid,
                                                        gen, val_encoders, n_matching_layers,
                                                        arg.match_with_top_layers, dp_params,
                                                        channel_ids_by_enc, do_second_moment_val,
                                                        writer, step)
          else:
            lm, lv = get_static_losses_valid(feat_emb.means_valid, feat_emb.vars_valid, gen,
                                             val_encoders, noise_maker, arg.batch_size,
                                             n_val_samples, n_classes_val,
                                             device, n_matching_layers, arg.match_with_top_layers,
                                             dp_params, channel_ids_by_enc, do_second_moment_val,
                                             writer, step)
          static_val_loss = lm + lv

        log_losses_and_imgs(gen, acc_losses, acc_losses_valid, fixed_noise,
                            step, writer, event_steps.final,
                            event_steps.valid, arg.log_dir, arg.exp_name)
        acc_losses = LossAccumulator()
        acc_losses_valid = LossAccumulator() if arg.validation_mode == 'mavg' else None

      if step % event_steps.syn_eval == 0 and step < arg.n_iter:
        if isinstance(gen, StaticDataset):
          ev_res = log_staticdset_synth_data_eval(gen, writer, step, device, arg.dataset,
                                                  arg.synth_dataset_size, arg.batch_size,
                                                  arg.log_dir, n_classes, arg.fid_dataset_size,
                                                  arg.image_size, arg.center_crop_size,
                                                  arg.data_scale, final_step=False)
          syn_data_file, eval_score = ev_res

        else:
          syn_data_file, eval_score = log_synth_data_eval(gen, writer, step, noise_maker, device,
                                                          arg.dataset, arg.synth_dataset_size,
                                                          arg.batch_size, arg.log_dir,
                                                          n_classes, arg.fid_dataset_size,
                                                          arg.image_size, arg.center_crop_size,
                                                          arg.data_scale, final_step=False)

        update_best_score(eval_score, step, syn_data_file, arg.dataset, best_result)
        update_best_proxy_score(static_val_loss, step, syn_data_file, best_proxy_result)

        old_syn_data_file = delete_old_syn_data(syn_data_file, old_syn_data_file, best_result, step,
                                                arg.keep_best_syn_data)

      if step % event_steps.ckpt == 0:
        create_checkpoint(feat_emb, gen, encoders, m_avg, optimizers,
                          step, fixed_noise, arg.log_dir, arg.update_type,
                          event_steps.new_ckpt, m_avg_valid, optimizers_valid,
                          best_result, best_proxy_result)

    if (event_steps.restart is not None) and (step % event_steps.restart == 0) \
            and event_steps.restart > 0 and step != event_steps.final:
      LOG.info(f'preparing restart at step {step} on device {device}')
      dist.barrier()
      dist.destroy_process_group()
      exit(3)

    gen.train()

  if device == 0:
    LOG.info('completed training')
  gen.eval()
  if arg.synth_dataset_size is not None and device == 0:
    if isinstance(gen, StaticDataset):
      syn_data_file, eval_score = log_staticdset_synth_data_eval(gen, writer, event_steps.final,
                                                                 device, arg.dataset,
                                                                 arg.synth_dataset_size,
                                                                 arg.batch_size, arg.log_dir,
                                                                 n_classes, arg.fid_dataset_size,
                                                                 arg.image_size,
                                                                 arg.center_crop_size,
                                                                 arg.data_scale, final_step=False)

    else:
      syn_data_file, eval_score = log_synth_data_eval(gen, writer, event_steps.final, noise_maker,
                                                      device, arg.dataset, arg.synth_dataset_size,
                                                      arg.batch_size, arg.log_dir, n_classes,
                                                      arg.fid_dataset_size, arg.image_size,
                                                      arg.center_crop_size, arg.data_scale,
                                                      final_step=True)
    update_best_score(eval_score, arg.n_iter, syn_data_file, arg.dataset, best_result)

    if arg.validation_mode == 'static':
      n_val_samples = arg.fid_dataset_size
      if isinstance(gen, StaticDataset):
        lm, lv = get_staticdset_static_losses_valid(feat_emb.means_valid, feat_emb.vars_valid,
                                                    gen, val_encoders, n_matching_layers,
                                                    arg.match_with_top_layers, dp_params,
                                                    channel_ids_by_enc, do_second_moment_val,
                                                    writer, arg.n_iter)
      else:
        lm, lv = get_static_losses_valid(feat_emb.means_valid, feat_emb.vars_valid, gen,
                                         val_encoders, noise_maker, arg.batch_size, n_val_samples,
                                         n_classes_val, device, n_matching_layers,
                                         arg.match_with_top_layers, dp_params, channel_ids_by_enc,
                                         do_second_moment_val, writer, arg.n_iter)
      static_val_loss = lm + lv
      update_best_proxy_score(static_val_loss, arg.n_iter, syn_data_file, best_proxy_result)
    if not arg.keep_best_syn_data:
      os.remove(syn_data_file)

  if writer is not None:
    writer.close()
  LOG.info(f'{best_result}, {best_proxy_result}')
  note_best_scores_column(best_result, arg.log_dir, is_proxy=False)
  if best_proxy_result.score is not None:
    note_best_scores_column(best_proxy_result, arg.log_dir, is_proxy=True)

  log_iteration(arg.exp_name, arg.n_iter, is_final=True)
  dist.destroy_process_group()


def main():
  arg = get_args()
  os.makedirs(arg.log_dir, exist_ok=True)
  os.makedirs(os.path.join(arg.log_dir, 'images/'), exist_ok=True)
  os.makedirs(os.path.join(arg.log_dir, 'tensorboard/'), exist_ok=True)
  log_args(arg)
  log_str = '_'.join(arg.log_dir.split('/'))
  ddp_init_file = f'{os.getcwd()}/init_file_{log_str}'
  arg.ddp_init_file = f'file://{ddp_init_file}'

  if os.path.exists(ddp_init_file):  # remove eventual leftover init file
    print(f'removing pre-existing {ddp_init_file}')
    os.remove(ddp_init_file)
  try:
    mp.spawn(ddp_run, args=(arg,), nprocs=arg.n_gpu, join=True)
  except pt.multiprocessing.ProcessExitedException as exit_err:
    LOG.info(f'caught process trying to exit with code {exit_err.exit_code}, exiting in turn')
    exit(exit_err.exit_code)





if __name__ == '__main__':
  main()
