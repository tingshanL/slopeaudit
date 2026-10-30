Interpretation notes
all_fits includes strict-invalid fits; strict_valid reproduces the archived strict-fit restriction.
The per-cell tables retain q0 and all other design distinctions. t4 pools designs descriptively.
q_direct_index is odds of mean direct reliability, not mean participant information.
boundary_or_singular_proxy means archive boundary flag OR 1-rho_hat^2 <= 1e-4;
near_correlation means remaining .95 <= abs(rho_hat); neither class proves exact numerical rank.
Missing correlations are unknown; exact zero fitted slope variance has its own class.
Normalized oracle D2 is undefined under true-zero variance. Archive zeros are not used as an oracle error target.
Positive-error shares use max(D2_hat-D2_oracle,0) and are undefined if the denominator is zero.
share_within_parent is within cell for t1/t3, within cell x q bin for t2, and within family for t4.
Q1_evaluation_rate uses only recorded strict-valid evaluation replicates (true-zero cells here).
archived_Q1_decision_rate describes all rows in a stratum and is not held-out null calibration.
Optimizer agreement rates are conditional on a recorded value; see optimizer_agreement_n.
These tables do not validate REML maxima or prove that boundary behavior is an optimizer artifact.
