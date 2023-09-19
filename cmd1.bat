@echo off
for %%e in (3 10) do (
  for %%b in (16) do (
    for %%l in (0.01) do (
      python run.py --epoch %%e --batch_size %%b --lr %%l
    )
  )
)