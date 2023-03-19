import numpy as np

def padding_list(l0, num, return_ndarray=True):
    l_pad = [l0_i for l0_i in l0 for j in range(num)]
    if return_ndarray:
        return np.array(l_pad)
    else:
        return l_pad

def get_loc_DP_p_list(DP_p_list, k):
    '''
    A single depolarizing channel is decomposed into $4^k - 1$ stochastic Pauli noise channels (See [Lemma 1, Ref. 1]).
    This function converts a given list of the error probability $p$ of each $k$-qubit depolarizing channel $\mathcal{D}_{k, p}$ in the circuit (notation follows Ref. [1])
    into a list of the error probability $\tilde{p}$ of each single stochastic Pauli noise channel in the decomposition of the depolarizing channel.
    For example, let us consider the case where three single-qubit depolarizing channels are present in the circuit with error probabilities p_0, p_1, p_2.
    Let each $\mathcal{D}_{1, p_i}$ be decomposed into three stochastic Pauli noise channels with the error probability q_i.
    Then, [p_0, p_1, p_2] is converted into [q_0, q_0, q_0, q_1, q_1, q_1, q_2, q_2, q_2].
    -------------------------------
    Reference [1] arXiv:2106.03390
    -------------------------------
    Args:
        DP_p_list (list of float) : list of the error probabilities of k-qubit depolarizing channels in the circuit
        k (int) : number of qubits
    Returns:
       list of float : list of error probability of stochastic Pauli noise
    '''
    ind = 1 / (2. * 4.**(k-1))
    factor = 4.**k / (4.**k - 1.)
    p_list = 0.5 * (1. - (1. - factor * DP_p_list)**ind)
    p_tot_list = padding_list(p_list, 4**k - 1)
    return p_tot_list

def rough_LB(error_prob_list, gap, G_list=None, c=1.):
    '''
    If G_list is given, this function calculates rough lower bound in [(36), Ref. 1] of the minimization error of the noisy cost function,
    otherwise extremely rough lower bound [(40), Ref. 1] is calculated instead.
    G_list stands for the quantity $G_i(\theta^*) = |<\phi(\theta^* + \pi e_i)|\phi(\theta^*)>|^2$ at a minimal point $\theta^*$ (see Eq. (29) in Ref. [1] for detail).
    See Ref. [1] for detail and definition of the notations used in the following documentation.
    The rough bound assumes that the noiseless precision of the minimization is good enough.
    The extremely rough bound further assumes $G_{i_1,i_2,...i_k}(\theta) \approx 1$.
    Hence, these bounds can be violated if the assumptions are not appropriate.
    -------------------------------
    Reference [1] arXiv:2106.03390
    -------------------------------
    Args:
        error_prob_list (list of float) : list of error probabilities of stochastic noise channels
        gap (float) : estimation or lower bound of the spectral gap $E_1 - E_0$ of the target operator $H$
        G_list (list of float, optional) : list of the quantities $G_i(\theta^*)$. If this is None, the extremely rough lower bound is calculated instead.
        c (float, optional) : You can use a constant correction factor c for the extremely rough lower bound, if you have $G_{i_1,i_2,...i_k}(\theta) > c$.
            If G_list is given, c is not used.
    Returns:
        float : rough lower bound
    '''
    if G_list is None:
        return _extremely_rough_LB(error_prob_list, gap, c=c)
    else:
        return _rough_LB(error_prob_list, gap, G_list)

def rough_UB(error_prob_list, spec_width, G_list=None):
    '''
    If G_list is given, this function calculates rough upper bound in [(36), Ref. 1] of the minimization error of the noisy cost function,
    otherwise rougher upper bound [(38) or (39), Ref. 1] is calculated instead.
    G_list stands for the quantity $G_i(\theta^*) = |<\phi(\theta^* + \pi e_i)|\phi(\theta^*)>|^2$ at a minimal point $\theta^*$ (see Eq. (29) in Ref. [1] for detail).
    See Ref. [1] for detail and definition of the notations used in the following documentation.
    Both bounds assume that the noiseless precision of the minimization is good enough.
    Hence, these bounds can be violated if the assumption is not appropriate.
    -------------------------------
    Reference [1] arXiv:2106.03390
    -------------------------------
    Args:
        error_prob_list (list of float) : list of error probabilities of stochastic noise channels
        spec_width (float) : estimation or upper bound of the width $E_{\max} - E_0$ of the spectrum of the target operator $H$
            Especially, you can use $2\|H\|$ or its upper bound instead, where $\|H\|$ is the operator norm of $H$.
        G_list (list of float, optional) : list of the quantities $G_i(\theta^*)$. If this is None, the rougher upper bound is calculated instead.
    Returns:
        float : rough upper bound
    '''
    if G_list is None:
        return _rougher_UB(error_prob_list, spec_width)
    else:
        return _rough_UB(error_prob_list, spec_width, G_list)
    
##############################

def _rough_LB(error_prob_list, gap, G_list):
    '''
    Calculates rough lower bound in [(36), Ref. 1] of the minimization error of the noisy cost function.
    See Ref. [1] for detail and definition of the notations used in the following documentation.
    This bound assumes that the noiseless precision of the minimization is good enough.
    Hence, it can be violated if the assumption is not appropriate.
    You need the quantity $G_i(\theta^*) = |<\phi(\theta^* + \pi e_i)|\phi(\theta^*)>|^2$ at a minimal point $\theta^*$ (see Eq. (29) in Ref. [1] for detail).
    Ref. [1] arXiv:2106.03390
    Args:
        error_prob_list (list of float) : list of error probabilities
        gap (float) : estimation or lower bound of the spectral gap $E_1 - E_0$ of the target operator $H$
        G_list (list of float) : list of the quantities $G_i(\theta^*)$
    Returns:
        float : rough lower bound
    '''
    p_list = np.array(error_prob_list)
    ind_list = np.arange(len(p_list))
    coef_list = [p_list[i] * np.prod(1. - p_list, where=ind_list!=i) for i in range(len(p_list))]
    LB = gap * np.inner(coef_list, G_list)
    return LB

def _rough_UB(error_prob_list, spec_width, G_list):
    '''
    Calculates rough upper bound in [(36), Ref. 1] of the minimization error of the noisy cost function.
    See Ref. [1] for detail and definition of the notations used in the following documentation.
    This bound assumes that the noiseless precision of the minimization is good enough.
    Hence, it can be violated if the assumptions is not appropriate.
    You need the quantity $G_i(\theta^*) = |<\phi(\theta^* + \pi e_i)|\phi(\theta^*)>|^2$ at a minimal point $\theta^*$ (see Eq. (29) in Ref. [1] for detail).
    Ref. [1] arXiv:2106.03390
    Args:
        error_prob_list (list of float) : list of error probabilities
        spec_width (float) : estimation or upper bound of the width $E_{\max} - E_0$ of the spectrum of the target operator $H$
            Especially, you can use $2\|H\|$ or its upper bound instead, where $\|H\|$ is the operator norm of $H$.
        G_list (list of float) : list of the quantities $G_i(\theta^*)$
    Returns:
        float : rough upper bound
    '''
    p_list = np.array(error_prob_list)
    ind_list = np.arange(len(p_list))
    coef_list = [p_list[i] * np.prod(1. - p_list, where=ind_list!=i) for i in range(len(p_list))]
    p_no_error = np.prod(1. - p_list)
    ResU = 1. - p_no_error - np.sum(coef_list)
    UB = spec_width * (np.inner(coef_list, G_list) + ResU)
    return UB

def _rougher_UB(error_prob_list, spec_width):
    '''
    Calculates "rougher upper bound" [(38) or (39), Ref. 1] of the minimization error of the noisy cost function.
    See Ref. [1] for detail and definition of the notations used in the following documentation.
    This bound assumes that the noiseless precision of the minimization is good enough.
    Hence, it can be violated if the assumption is not appropriate.
    Ref. [1] arXiv:2106.03390
    Args:
        error_prob_list (list of float) : list of error probabilities
        spec_width (float) : estimation or upper bound of the width $E_{\max} - E_0$ of the spectrum of the target operator $H$
            Especially, you can use $2\|H\|$ or its upper bound instead, where $\|H\|$ is the operator norm of $H$.
    Returns:
        float : rougher upper bound
    '''
    error_prob_list = np.array(error_prob_list)
    p_no_error = np.prod(1. - error_prob_list)
    UB = spec_width * (1. - p_no_error)
    return UB

def _extremely_rough_LB(error_prob_list, gap, c=1.):
    '''
    Calculates extremely rough lower bound [(40), Ref. 1] of the minimization error of the noisy cost function.
    See Ref. [1] for detail and definition of the notations used in the following documentation.
    This bound is based on a rough assumption $G_{i_1,i_2,...i_k}(\theta) \approx 1$ as well as assuming that the noiseless precision of the minimization is good enough.
    Hence, it can be violated if the assumptions are not appropriate.
    Ref. [1] arXiv:2106.03390
    Args:
        error_prob_list (list of float) : list of error probabilities
        gap (float) : estimation or lower bound of the spectral gap $E_1 - E_0$ of the target operator $H$
        c (float, optional) : You can use a constant correction factor c, if you have $G_{i_1,i_2,...i_k}(\theta) > c$.
    Returns:
        float : extremely rough lower bound
    '''
    error_prob_list = np.array(error_prob_list)
    p_no_error = np.prod(1. - error_prob_list)
    LB = gap * (1. - p_no_error) * c
    return LB