import nlu_module.nlu as nlu

test = nlu.nlu_pipeline("test", "", "SLURP", "gpt-oss:20b", 2)

print(test["base_intent"])