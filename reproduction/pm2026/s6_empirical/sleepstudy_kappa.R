# Supplement S6.2: sleepstudy source decomposition and nuisance capacity (public data).
# Primary model: Reaction ~ Days + (1 + Days | Subject), REML, independent equal-variance
# residuals, Days 0-9. Computes per-participant quantities with the general matrix formulas and
# checks them against the closed form of Corollary S1. Output names keep the original s7_ prefix.
# Usage (from the repository root): Rscript reproduction/pm2026/s6_empirical/sleepstudy_kappa.R <out_dir>
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[1] else "s7_sleepstudy"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fit <- lmer(Reaction ~ Days + (1 + Days | Subject), data = sleepstudy, REML = TRUE)
G <- as.matrix(VarCorr(fit)$Subject)
s2 <- sigma(fit)^2
tau2 <- G[2, 2]; g01 <- G[1, 2]; G00 <- G[1, 1]
S <- G00 - g01^2 / tau2

rows <- lapply(split(sleepstudy, sleepstudy$Subject), function(d) {
  t <- d$Days; n <- length(t)
  Z <- cbind(1, t)
  V <- Z %*% G %*% t(Z) + s2 * diag(n)
  w <- solve(V, Z %*% G[, 2])                     # s_hat = w' (y - X beta)
  D2 <- as.numeric(t(w) %*% Z %*% G[, 2]) / tau2
  J <- sum((t - mean(t))^2) / s2
  R <- tau2 * J / (1 + tau2 * J)
  kappa <- sum(w)^2 * S / (tau2 * D2)             # Var(nu) / Var(s_hat), nu = w' 1 u
  resid_share <- s2 * sum(w^2) / (tau2 * D2)
  kappa_cf <- (D2 - R) * (1 - D2) / (D2 * (1 - R)) * S / (S + s2 / n)
  data.frame(Subject = d$Subject[1], n = n, D2 = D2, R_direct = R, Delta_fit = D2 - R,
             kappa = kappa, sqrt_kappa = sqrt(kappa), residual_share = resid_share,
             composition_check = D2 + kappa + resid_share, closed_form_error = kappa - kappa_cf,
             loading_sign = sign(sum(w)))
})
per <- do.call(rbind, rows)
summ <- data.frame(
  N = nrow(per), G00 = G00, tau2 = tau2, cov01 = g01, rho = g01 / sqrt(G00 * tau2), sigma2 = s2, S = S,
  mean_D2 = mean(per$D2), mean_Rdirect = mean(per$R_direct), mean_Delta_fit = mean(per$Delta_fit),
  mean_kappa = mean(per$kappa), mean_sqrt_kappa = mean(per$sqrt_kappa),
  max_abs_closed_form_error = max(abs(per$closed_form_error)),
  max_abs_composition_error = max(abs(per$composition_check - 1)),
  identical_across_subjects = diff(range(per$kappa)) < 1e-12)
write.csv(per, file.path(out_dir, "s7_sleepstudy_per_subject.csv"), row.names = FALSE)
write.csv(summ, file.path(out_dir, "s7_sleepstudy_summary.csv"), row.names = FALSE)
print(t(round(summ, 6)))
cat("wrote", out_dir, "\n")
