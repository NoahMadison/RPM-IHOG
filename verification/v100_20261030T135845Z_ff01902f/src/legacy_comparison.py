"""User-provided printed legacy values: historical claims, not independent evidence."""
KEYS=['final_train_loss','val_loss_fp16','val_ppl_fp16','val_loss_pure_kron_053bpw','val_loss_rpm_ihog_212bpw','token_match_pure_kron_053bpw_pct','token_match_rpm_ihog_212bpw_pct','shannon_entropy_body_bits','svd_stable_rank','kronecker_stable_rank','sqnr_pure_kronecker_db','sqnr_rpm_ihog_ternary_db']
EXPECTED={
 '1_Standard_AdamW':[2.7560,2.7448,15.56,8.7492,3.0993,2.67,62.24,7.448,25.45,64.95,.52,7.74],
 '2_EntropyConstrained_Kronecker_AdamW':[3.3257,3.2800,26.58,3.2738,3.2758,90.10,94.40,7.412,47.50,6.99,24.03,31.30],
 '3_Native_RPM_IHOG_Generator':[3.3727,3.3066,27.29,3.3094,3.3070,97.39,99.24,7.409,50.63,5.41,24.80,32.01]}
def compare(runs):
 return {arm:{key:{'historical_printed':expected,'observed_printed':runs[arm][key],'delta_printed':runs[arm][key]-expected,'matches_printed_value':abs(runs[arm][key]-expected)<1e-8} for key,expected in zip(KEYS,values)} for arm,values in EXPECTED.items()}
