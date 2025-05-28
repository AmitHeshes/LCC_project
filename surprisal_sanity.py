import kenlm

model = kenlm.Model('model_3gram.binary')

log_prob_not_regular = model.score("Cheese polar red lamp", bos=True, eos=True)
log_prob_regular = model.score("It is best known for", bos=True, eos=True)

print(f"Log probability of normal sentence: {log_prob_regular}")
print(f"Log probability of non-regular sentence: {log_prob_not_regular}")