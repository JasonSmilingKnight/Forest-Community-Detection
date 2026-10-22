"""Fixed-K sampler support, adapted from Min Xu (2020–2021) and project updates."""
import numpy as np

def estimateAlphaEM(graf, K=1, maxiter=40, alpha_init=1, display=True):
    """Approximate EM for alpha >= 0, with beta fixed at one."""
    m = len(graf.es)
    n = len(graf.vs)
    degseq = graf.degree()
    max_deg = max(degseq)
    thetahat = (m - (n - K)) / (n * (n - 1) / 2 - (n - K))
    alphap = alpha_init
    for i in range(maxiter):
        condprob = np.zeros((max_deg + 1, max_deg + 1))
        cumprob = np.zeros((max_deg + 1, max_deg + 1))
        for k in range(1, max_deg + 1):
            if k > 5 * n * thetahat:
                condprob[k, k] = 1
                for s in range(k - 1, 0, -1):
                    condprob[k, s] = condprob[k, s + 1] * (n - s) * thetahat * (1 / (k - s)) * (s + 3 + 2 * alphap) / (s + alphap)
            else:
                condprob[k, 1] = 1
                for s in range(2, k + 1):
                    condprob[k, s] = condprob[k, s - 1] / ((n - (s - 1)) * thetahat) * (k - (s - 1)) * (s - 1 + alphap) / (s - 1 + 3 + 2 * alphap)
            condprob[k,] = condprob[k,] / sum(condprob[k,])
            cumprob[k, k] = 0
            for s in range(k - 1, 0, -1):
                cumprob[k, s] = cumprob[k, s + 1] + condprob[k, s + 1]
        Mterm = [0] * max_deg
        for j in range(max_deg):
            for ii in range(n):
                Mterm[j] = Mterm[j] + cumprob[degseq[ii], j + 1]
        Mterm = np.array(Mterm)
        Mterm = Mterm * n / sum(Mterm)
        alpha_next = optimizeAlpha(Mterm, n, alphap)
        if abs(alpha_next - alphap) < 0.0001:
            break
        alphap = alpha_next
    return alphap

def optimizeAlpha(Mterm, n, alphainit=1):
    """Projected gradient ascent with backtracking for the M-step."""
    max_deg = len(Mterm)
    maxiter = 50
    alpha = alphainit
    ls_a = 0.1
    ls_b = 0.5
    stepsize = 1
    ks = np.array(range(3, n + 1))
    js = np.array(range(1, max_deg + 1))
    for i in range(maxiter):
        grad1 = sum(1 / (js + alpha) * Mterm)
        grad1 = grad1 - sum((ks - 1) / (2 * (ks - 2) + (ks - 1) * alpha))
        if abs(grad1) < 0.0001:
            break
        if alpha < 0.0001 and grad1 < 0.0001:
            break
        for ii in range(maxiter):
            alphap = max(alpha + stepsize * grad1, 0)
            diff = alphap - alpha
            old_val = sum(np.log(js + alpha) * Mterm) - sum(np.log(2 * (ks - 2) + (ks - 1) * alpha))
            new_val = sum(np.log(js + alphap) * Mterm) - sum(np.log(2 * (ks - 2) + (ks - 1) * alphap))
            if new_val > old_val + ls_a * stepsize * grad1 * diff:
                alpha = alphap
                break
            else:
                stepsize = stepsize * ls_b
    return alpha
