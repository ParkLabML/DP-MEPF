from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.transformer_zoo import Composition, AmplificationBySampling


def find_sigma(dp_mechanism, target_eps, eps_tol=0.01, sigma_tol=0.001,
               sigma_init=1., verbose=False):
  """
  searches for a sigma that produces the desired eps,delta bound for a given dp mechanism.
  :param dp_mechanism: function that takes sigma and returns epsilon
  :param target_eps: epsilon that should be met
  :param eps_tol: terminate search if epsilon is in [target_eps * (1 - eps_tol), target_eps]
  :param sigma_tol: terminate search if sigma is in [sigma_lower, sigma_lower * (1 + sigma_tol)]
  :return: (sigma, eps) tuple
  """
  # Find initial interval of [sigma_lower, sigma_upper] containing the target
  sigma_lower, sigma_upper = sigma_init, sigma_init
  eps = dp_mechanism(sigma_init)
  if eps <= target_eps:
    while eps <= target_eps:
      sigma_upper = sigma_lower
      sigma_lower /= 2.
      eps = dp_mechanism(sigma_lower)
  else:
    while eps > target_eps:
      sigma_lower = sigma_upper
      sigma_upper *= 2.
      eps = dp_mechanism(sigma_upper)

  if verbose:
    print(f'starting with sigma lower = {sigma_lower}, sigma upper = {sigma_upper}')

  # search the computed interval until sigma or epsilon are within tolerance intervals to the target
  while True:
    sigma = (sigma_upper + sigma_lower) / 2
    eps = dp_mechanism(sigma)
    if verbose:
      print(f'tested sigma = {sigma:3.3f}, got eps = {eps}')

    if eps > target_eps:
      sigma_lower = sigma
    else:
      sigma_upper = sigma

      if eps > target_eps * (1 - eps_tol) or sigma < sigma_lower * (1 + sigma_tol):
        if verbose:
          tol_str = "epsilon" if eps > target_eps * (1 - eps_tol) else "sigma"
          print(f'search terminated by reaching {tol_str} tolerance interval')
        return eps, sigma


def find_single_release_sigma(target_eps, target_delta, neighbouring_relation='swap'):
  assert neighbouring_relation in {'swap', 'add_remove'}

  def dp_mechanism(sigma):
    gm = ExactGaussianMechanism(sigma)
    gm.replace_one = neighbouring_relation == 'swap'
    return gm.get_approxDP(target_delta)

  return find_sigma(dp_mechanism, target_eps, verbose=False)


def find_two_release_sigma(target_eps, target_delta, second_sigma_scale=1.,
                           neighbouring_relation='swap'):
  assert neighbouring_relation in {'swap', 'add_remove'}

  def dp_mechanism(sig):
    gm1 = ExactGaussianMechanism(sig)
    gm2 = ExactGaussianMechanism(sig * second_sigma_scale)
    composed_gm = Composition()([gm1, gm2], [1, 1])
    composed_gm.replace_one = neighbouring_relation == 'swap'
    return composed_gm.get_approxDP(target_delta)

  eps, sigma = find_sigma(dp_mechanism, target_eps, verbose=False)
  return eps, sigma


def find_train_val_sigma_m1(target_eps, target_delta, val_noise_scaling=1.,
                            neighbouring_relation='swap'):
  eps, sigma_train = find_two_release_sigma(target_eps, target_delta, val_noise_scaling,
                                            neighbouring_relation)
  sigma_val = sigma_train * val_noise_scaling
  return eps, sigma_train, sigma_val


def find_train_val_sigma_m1m2(target_eps, target_delta, m2_scaling, val_noise_scaling,
                              neighbouring_relation='swap'):
  assert neighbouring_relation in {'swap', 'add_remove'}

  def dp_mechanism(sig):
    gm1 = ExactGaussianMechanism(sig)
    gm2 = ExactGaussianMechanism(sig * m2_scaling)
    gm3 = ExactGaussianMechanism(sig * val_noise_scaling)
    gm4 = ExactGaussianMechanism(sig * m2_scaling * val_noise_scaling)
    composed_gm = Composition()([gm1, gm2, gm3, gm4], [1, 1, 1, 1])
    composed_gm.replace_one = neighbouring_relation == 'swap'
    return composed_gm.get_approxDP(target_delta)

  eps, sigma_train = find_sigma(dp_mechanism, target_eps, verbose=False)
  sigma_val = sigma_train * val_noise_scaling
  return eps, sigma_train, sigma_val


def find_dpsgd_sigma(target_eps, target_delta, batch_size, n_samples, n_iter,
                     neighbouring_relation='swap'):
  assert neighbouring_relation in {'swap', 'add_remove'}

  def dp_mechanism(sig):
    subsample = AmplificationBySampling(PoissonSampling=False)
    mech = ExactGaussianMechanism(sigma=sig)
    mech.replace_one = neighbouring_relation == 'swap'
    # Create subsampled Gaussian mechanism
    prob = batch_size / n_samples
    SubsampledGaussian_mech = subsample(mech, prob, improved_bound_flag=True)

    composed_gm = Composition()([SubsampledGaussian_mech], [n_iter])
    # composed_gm.replace_one = neighbouring_relation == 'swap'
    # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
    return composed_gm.get_approxDP(target_delta)

  eps, sigma = find_sigma(dp_mechanism, target_eps, verbose=True)
  return eps, sigma


def rebuttal_delta_debate():
  epsilons = [10., 5., 2., 1., .5, .2]
  wrong_delta = 1e-5
  right_delta = 1e-6 # 1/203_000
  m2_scaling = 1.
  val_noise_scaling = 10.
  neighbouring_relation = 'swap'

  def right_mechanism(sig):
    gm1 = ExactGaussianMechanism(sig)
    gm2 = ExactGaussianMechanism(sig * m2_scaling)
    gm3 = ExactGaussianMechanism(sig * val_noise_scaling)
    gm4 = ExactGaussianMechanism(sig * m2_scaling * val_noise_scaling)
    composed_gm = Composition()([gm1, gm2, gm3, gm4], [1, 1, 1, 1])
    composed_gm.replace_one = neighbouring_relation == 'swap'
    return composed_gm.get_approxDP(right_delta)

  for wrong_eps in epsilons:
    _, sigma_train, _ = find_train_val_sigma_m1m2(wrong_eps, wrong_delta, m2_scaling,
                                                  val_noise_scaling, neighbouring_relation)
    _, sigma_new, _ = find_train_val_sigma_m1m2(wrong_eps, right_delta, m2_scaling,
                                                  val_noise_scaling, neighbouring_relation)
    right_eps = right_mechanism(sigma_train)
    print(f'for m1m2+val, ({wrong_eps}, {wrong_delta})-DP is also ({right_eps}, {right_delta})-DP')
    # print(f'epsilon{right_eps/wrong_eps}')
    print(f'sigma old={sigma_train}, sigma new={sigma_new}, frac={sigma_new/sigma_train}')




def main():
  target_eps = 1.
  target_delta = 1e-6
  m2_scaling = 1.
  val_noise_scaling = 10.
  _, sig1 = find_single_release_sigma(target_eps, target_delta)
  _, sig2 = find_two_release_sigma(target_eps, target_delta, second_sigma_scale=1.)
  _, sig3, _ = find_train_val_sigma_m1(target_eps, target_delta, val_noise_scaling)
  _, sig4, _ = find_train_val_sigma_m1m2(target_eps, target_delta, m2_scaling, val_noise_scaling)

  print(sig1, sig2, sig3, sig4)


def tmlr_rebuttal_check():
  for eps in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
    _, sig1 = find_single_release_sigma(eps, target_delta=1e-5)
    # _, sig1, _ = find_train_val_sigma_m1m2(eps, target_delta=1e-5, m2_scaling=1., val_noise_scaling=10.)
    print(eps, sig1)


if __name__ == '__main__':
  # main()
  # rebuttal_delta_debate()
  tmlr_rebuttal_check()
