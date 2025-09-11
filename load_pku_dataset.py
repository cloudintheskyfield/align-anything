from datasets import load_dataset

ds = load_dataset("PKU-Alignment/Align-Anything-TI2T-Instruction-100K",
                  cache_dir=r"C:\Users\shuang_wang\PyCharmProjects\ws_01_meita_training\remote\align-anything\data"
                  )
print('ok')